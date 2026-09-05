"""Compatibility admission is not proof of provider context fit."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent.compression_v3 import (
    ContextProjectionUnfit,
    prepare_api_request,
    provider_request_budget,
)


@pytest.mark.parametrize("window", [None, 0, -1, "unknown"])
def test_unknown_window_is_indeterminate_but_compatible(window):
    agent = SimpleNamespace(_config_context_length=window)
    request = {
        "input": [{"role": "user", "content": "task " * 100}],
        "max_output_tokens": 16,
    }

    budget = provider_request_budget(agent, request)

    assert budget.certainty == "estimated"
    assert budget.fit_status == "unknown_window"
    assert budget.fits is None
    assert budget.allows_dispatch is True
    assert budget.safe_input_budget is None
    assert prepare_api_request(agent, request) == request


@pytest.mark.parametrize("key", ["messages", "input"])
def test_unknown_window_does_not_mutate_history_or_inject_budget_into_wire(key):
    agent = SimpleNamespace()
    request = {key: [{"role": "user", "content": "exact task", "_row_id": 7}]}
    original = deepcopy(request)

    assert prepare_api_request(agent, request) == {
        key: [{"role": "user", "content": "exact task"}]
    }
    assert request == original


@pytest.mark.parametrize(
    "content",
    [
        "plain text",
        "中文 русский 😀",
        [{"type": "input_image", "image_url": "https://example.invalid/image.png"}],
    ],
)
def test_known_window_and_model_name_do_not_verify_a_heuristic(content):
    agent = SimpleNamespace(_config_context_length=100_000, model="gpt-4o")
    request = {"input": [{"role": "user", "content": content}], "max_output_tokens": 16}

    budget = provider_request_budget(agent, request)

    assert budget.certainty == "estimated"
    assert budget.fit_status == "estimated_fit"
    assert budget.fits is True
    assert budget.allows_dispatch is True
    assert budget.safety_margin == 1024


def test_estimated_overflow_is_identified_in_error_without_request_body():
    agent = SimpleNamespace(_config_context_length=100, _compression_safety_margin=10)
    request = {
        "input": [{"role": "user", "content": "private-body-" * 100}],
        "max_output_tokens": 16,
    }
    budget = provider_request_budget(agent, request)

    assert budget.certainty == "estimated"
    assert budget.fit_status == "estimated_overflow"
    assert budget.fits is False
    assert budget.allows_dispatch is False
    with pytest.raises(ContextProjectionUnfit) as raised:
        prepare_api_request(agent, request)
    assert raised.value.budget_certainty == "estimated"
    assert raised.value.budget_status == "estimated_overflow"
    assert "estimated" in str(raised.value)
    assert "private-body" not in str(raised.value)


def test_unknown_window_reaches_transport_exactly_once():
    calls = []
    response = object()

    def run(request, **_kwargs):
        calls.append(request)
        return response

    agent = SimpleNamespace(
        api_mode="codex_responses", provider="openai-codex", _run_codex_stream=run
    )
    request = {"input": [{"role": "user", "content": "task"}]}

    assert (
        _dispatch_nonstreaming_api_request(
            agent, request, make_client=lambda *_args, **_kwargs: object()
        )
        is response
    )
    assert calls == [request]


def test_budget_diagnostics_have_no_prompt_or_credentials(caplog):
    agent = SimpleNamespace(_config_context_length=100_000)
    request = {
        "input": [{"role": "user", "content": "private-prompt"}],
        "api_key": "private-key",
    }

    with caplog.at_level("DEBUG", logger="agent.compression_v3"):
        prepare_api_request(agent, request)

    assert "certainty=estimated" in caplog.text
    assert "status=estimated_fit" in caplog.text
    assert "private-prompt" not in caplog.text
    assert "private-key" not in caplog.text
