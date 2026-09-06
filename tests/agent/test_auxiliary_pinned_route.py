"""Call-scoped pinning must fail without switching the selected route."""

import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import openai
import pytest

from agent import auxiliary_client as aux


@pytest.fixture
def pinned_route(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    # Keep registry-wide credential discovery off the operator's gh login.
    from hermes_cli import copilot_auth
    monkeypatch.setattr(copilot_auth, "_try_gh_cli_token", lambda: None)
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  completion_judge:\n"
        "    provider: openrouter\n    model: unrelated-model\n"
        "    base_url: https://unrelated.example/v1\n"
        "    fallback_chain:\n      - provider: nous\n        model: another-model\n",
        encoding="utf-8",
    )
    aux.shutdown_cached_clients()
    calls = []
    clients = []
    handler_result = {"status": 200}

    def handle(request):
        calls.append((str(request.url), json.loads(request.content)))
        if handler_result["status"] == "connection":
            raise httpx.ConnectError("test connection unavailable", request=request)
        if handler_result["status"] != 200:
            return httpx.Response(handler_result["status"], json={"error": {"message": "test refusal"}})
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion", "created": 0,
            "model": calls[-1][1]["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Reviewed result",
            }}],
        })

    def create_client(**kwargs):
        client = openai.OpenAI(
            api_key=kwargs["api_key"], base_url=kwargs["base_url"],
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
            max_retries=0,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(aux, "_create_openai_client", create_client)
    for name in (
        "_try_configured_fallback_for_unavailable_client",
        "_try_configured_fallback_chain", "_try_main_agent_model_fallback",
    ):
        monkeypatch.setattr(aux, name, Mock(side_effect=AssertionError("unexpected fallback")))
    try:
        yield calls, handler_result
    finally:
        aux.shutdown_cached_clients()
        for client in clients:
            client.close()


@pytest.mark.parametrize("task", ["completion_judge", "vision"])
def test_explicit_route_ignores_task_route_and_preserves_budget(pinned_route, task, tmp_path):
    calls, _ = pinned_route
    if task == "vision":
        (tmp_path / "config.yaml").write_text(
            "auxiliary:\n  vision:\n    provider: openrouter\n"
            "    base_url: https://unrelated.example/v1\n", encoding="utf-8",
        )
    route_info = {}
    result = aux.call_llm(
        task, provider="openrouter", model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Review"}], max_tokens=4096,
        allow_fallback=False, route_info=route_info,
    )
    assert result.choices[0].message.content == "Reviewed result"
    assert len(calls) == 1
    assert calls[0][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0][1]["model"] == "anthropic/claude-sonnet-4-6"
    assert calls[0][1]["max_tokens"] == 4096
    assert route_info["provider"] == "openrouter"
    assert route_info["model"] == calls[0][1]["model"]


@pytest.mark.parametrize("status,error_type", [
    (402, openai.APIStatusError),
    (401, openai.AuthenticationError),
    (429, openai.RateLimitError),
    (503, openai.InternalServerError),
    ("connection", openai.APIConnectionError),
])
def test_first_provider_error_propagates_without_auxiliary_retry(pinned_route, status, error_type):
    calls, result = pinned_route
    result["status"] = status
    with pytest.raises(error_type):
        aux.call_llm(
            "completion_judge", provider="openrouter", model="test-model",
            messages=[{"role": "user", "content": "Review"}], allow_fallback=False,
        )
    assert len(calls) == 1
    assert calls[0][1]["model"] == "test-model"


@pytest.mark.parametrize("provider", ["openrouter", "antigravity", "custom"])
def test_missing_client_never_enters_configured_or_auto_fallback(pinned_route, monkeypatch, provider):
    calls, _ = pinned_route
    resolve = Mock(return_value=(None, None))
    monkeypatch.setattr(aux, "_get_cached_client", resolve)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        aux.call_llm(
            "completion_judge", provider=provider, model="test-model",
            messages=[], allow_fallback=False,
            **({"base_url": "https://selected.example/v1"} if provider == "custom" else {}),
        )
    assert resolve.call_count == 1
    assert resolve.call_args.args[:2] == (provider, "test-model")
    assert calls == []


@pytest.mark.parametrize("provider,model", [
    (None, "test-model"), ("auto", "test-model"),
    ("main", "test-model"), ("actual", "test-model"), ("moa", "preset"),
    ("openrouter", None), ("openrouter", "auto"), ("openrouter", "  "),
])
def test_pinning_requires_concrete_explicit_route(monkeypatch, provider, model):
    resolve = Mock(side_effect=AssertionError("invalid route reached resolution"))
    monkeypatch.setattr(aux, "_get_cached_client", resolve)
    with pytest.raises(ValueError, match="explicit provider and model"):
        aux.call_llm(provider=provider, model=model, messages=[], allow_fallback=False)
    resolve.assert_not_called()


def test_default_call_still_uses_unavailable_provider_fallback(monkeypatch):
    response = Mock(choices=[Mock(message=Mock(content="Fallback result"))])
    fallback = Mock()
    fallback.chat.completions.create.return_value = response
    fallback.base_url = "https://fallback.example/v1"
    monkeypatch.setattr(aux, "_get_cached_client", Mock(return_value=(None, None)))
    recovery = Mock(return_value=(fallback, "fallback-model", "openrouter"))
    monkeypatch.setattr(aux, "_try_configured_fallback_for_unavailable_client", recovery)
    result = aux.call_llm(provider="antigravity", model="test-model", messages=[])
    assert result is response
    recovery.assert_called_once()
    assert fallback.chat.completions.create.call_args.kwargs["model"] == "fallback-model"


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("provider", ["custom", "custom:", "custom:custom"])
def test_strict_bare_custom_cannot_borrow_available_provider(pinned_route, monkeypatch, provider, warm_cache):
    calls, _ = pinned_route
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    if warm_cache:
        aux.call_llm(provider=provider, model="deepseek-chat", main_runtime={}, messages=[])
        assert len(calls) == 1
        assert calls[0][0] == "https://api.deepseek.com/v1/chat/completions"
        calls.clear()
    with pytest.raises(ValueError, match="explicit base_url"):
        aux.call_llm(
            provider=provider, model="deepseek-chat", main_runtime={},
            messages=[], allow_fallback=False,
        )
    assert calls == []


@pytest.mark.parametrize("provider", [
    "actual-computer", "actualcomputer", "aci", "custom:main", "custom:auto", "custom:moa",
])
def test_strict_route_rejects_aliases_to_implicit_routes_before_cache(pinned_route, monkeypatch, provider):
    calls, _ = pinned_route
    resolve = Mock(side_effect=AssertionError("implicit route reached cache"))
    monkeypatch.setattr(aux, "_get_cached_client", resolve)
    with pytest.raises(ValueError, match="explicit provider and model"):
        aux.call_llm(provider=provider, model="test-model", messages=[], allow_fallback=False)
    resolve.assert_not_called()
    assert calls == []


@pytest.mark.parametrize("provider", ["custom", "openai", "ollama", "custom:ollama"])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_strict_explicit_custom_endpoint_is_used_without_discovery(pinned_route, monkeypatch, provider, warm_cache):
    calls, _ = pinned_route
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    if warm_cache:
        aux.call_llm(provider="custom", model="test-model", main_runtime={}, messages=[])
        assert calls[0][0] == "https://api.deepseek.com/v1/chat/completions"
        calls.clear()
    aux.call_llm(
        provider=provider, model="test-model", base_url="https://selected.example/v1",
        messages=[], allow_fallback=False,
    )
    assert len(calls) == 1
    assert calls[0][0] == "https://selected.example/v1/chat/completions"


@pytest.mark.parametrize("provider", ["myrelay", "custom:myrelay"])
def test_strict_named_custom_provider_uses_its_declared_endpoint(pinned_route, tmp_path, monkeypatch, provider):
    calls, _ = pinned_route
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    (tmp_path / "config.yaml").write_text(
        "custom_providers:\n  - name: myrelay\n"
        "    base_url: https://named.example/v1\n    api_key: test-only-named-key\n",
        encoding="utf-8",
    )
    aux.call_llm(provider=provider, model="test-model", messages=[], allow_fallback=False)
    assert len(calls) == 1
    assert calls[0][0] == "https://named.example/v1/chat/completions"
    assert calls[0][1]["model"] == "test-model"


def test_strict_unknown_provider_does_not_borrow_available_provider(pinned_route, monkeypatch):
    calls, _ = pinned_route
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-deepseek-key")
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        aux.call_llm(provider="unknown-test-provider", model="test-model", messages=[], allow_fallback=False)
    assert calls == []


@pytest.mark.parametrize("provider", ["ollama", "custom:ollama"])
def test_strict_aliases_to_custom_require_explicit_endpoint(pinned_route, provider):
    calls, _ = pinned_route
    with pytest.raises(ValueError, match="explicit base_url"):
        aux.call_llm(provider=provider, model="test-model", messages=[], allow_fallback=False)
    assert calls == []
