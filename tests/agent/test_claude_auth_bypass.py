"""Regression tests for the integrated Claude OAuth bypass boundaries."""

from types import SimpleNamespace

import pytest

from agent import claude_auth_bypass as bypass


class _RateLimitError(Exception):
    status_code = 429
    response = SimpleNamespace(
        status_code=429,
        headers={
            "retry-after": "21600",
            "anthropic-ratelimit-unified-5h-status": "throttled",
            "anthropic-ratelimit-unified-5h-reset": "1890000000",
        },
    )


def _adapter_module():
    def build_anthropic_kwargs(*, is_oauth=False):
        return {"is_oauth": is_oauth}

    return SimpleNamespace(
        _OAUTH_ONLY_BETAS=[],
        build_anthropic_kwargs=build_anthropic_kwargs,
        __version__="test",
    )


def test_apply_patches_leaves_both_interruptible_calls_and_429s_to_core(
    monkeypatch,
):
    """OAuth patching must not install a sleep/retry layer around AIAgent."""
    calls = []

    class FakeAgent:
        def _interruptible_api_call(self, *_args, **_kwargs):
            calls.append("non-streaming")
            raise _RateLimitError("quota exhausted")

        def _interruptible_streaming_api_call(self, *_args, **_kwargs):
            calls.append("streaming")
            raise _RateLimitError("quota exhausted")

    original_api = FakeAgent._interruptible_api_call
    original_stream = FakeAgent._interruptible_streaming_api_call
    monkeypatch.setattr(bypass, "_install_thinking_replay_classifier_patch", lambda: None)
    monkeypatch.setattr(bypass, "_install_response_pascalcase_unhook", lambda _aa: None)
    monkeypatch.setattr(bypass, "_install_pool_select_hook", lambda: None)
    monkeypatch.setattr(bypass, "_get_version_safely", lambda _aa: "test")

    import run_agent

    original_agent_api = run_agent.AIAgent._interruptible_api_call
    original_agent_stream = run_agent.AIAgent._interruptible_streaming_api_call
    assert bypass.apply_patches(_adapter_module()) is True
    assert not hasattr(bypass, "install_rate_limit_autowait")
    assert run_agent.AIAgent._interruptible_api_call is original_agent_api
    assert run_agent.AIAgent._interruptible_streaming_api_call is original_agent_stream
    assert FakeAgent._interruptible_api_call is original_api
    assert FakeAgent._interruptible_streaming_api_call is original_stream

    agent = FakeAgent()
    with pytest.raises(_RateLimitError):
        agent._interruptible_api_call({})
    with pytest.raises(_RateLimitError):
        agent._interruptible_streaming_api_call({})
    assert calls == ["non-streaming", "streaming"]
