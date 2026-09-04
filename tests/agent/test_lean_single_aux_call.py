"""Behavior contracts for the one-request lean compaction path."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    ContextCompressor,
    _LEAN_ANCHOR_HEADING,
    _LEAN_RECOVERY_HEADING,
    _LEAN_SESSION_LOG_HEADING,
)


def _compressor(**overrides):
    kwargs = {
        "model": "test/model",
        "threshold_percent": 0.85,
        "protect_first_n": 2,
        "protect_last_n": 2,
        "quiet_mode": True,
        "tail_mode": "lean",
    }
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        compressor = ContextCompressor(**kwargs)
        _ = compressor.context_length
    compressor._session_id = "single-call-session"
    return compressor


def _response(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def _turns(rounds=8, tool_chars=6_000):
    messages = [
        {
            "role": "user",
            "content": "Please fix PR #12345 in agent/foo.py",
            "origin_kind": "human_user",
            "turn_kind": "prompt",
            "trust_kind": "user_authorized",
        }
    ]
    for index in range(rounds):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"Editing agent/foo.py in step {index}",
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "tool_name": "terminal",
                    "content": f"step {index}: " + "x" * tool_chars,
                },
            ]
        )
    return messages


_SUMMARY = (
    "## Historical Task Snapshot\n"
    "User asked: 'Please fix PR #12345 in agent/foo.py'\n\n"
    "## Goal\nFix the bug.\n\n"
    f"{_LEAN_SESSION_LOG_HEADING}\n"
    "- Edited agent/foo.py for PR #12345.\n"
)


def test_lean_compaction_uses_exactly_one_auxiliary_request():
    compressor = _compressor()
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(_SUMMARY),
    ) as summary_call, patch(
        "agent.auxiliary_client.call_llm",
        return_value=_response(_SUMMARY),
    ) as sibling_call:
        summary = compressor._generate_summary(_turns())

    assert summary is not None
    assert summary_call.call_count + sibling_call.call_count == 1
    assert _LEAN_SESSION_LOG_HEADING in summary
    assert _LEAN_ANCHOR_HEADING in summary
    assert _LEAN_RECOVERY_HEADING in summary


def test_lean_prompt_requests_detailed_log_in_same_response():
    compressor = _compressor()
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(_SUMMARY),
    ) as summary_call:
        compressor._generate_summary(_turns())

    prompt = summary_call.call_args.kwargs["messages"][0]["content"]
    assert _LEAN_SESSION_LOG_HEADING in prompt
    assert "PRESERVE EXACTLY" in prompt


def test_oversized_lean_input_is_evenly_sampled_in_one_request():
    compressor = _compressor()
    turns = _turns(rounds=120)
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response(_SUMMARY),
    ) as summary_call:
        compressor._generate_summary(turns)

    assert summary_call.call_count == 1
    prompt = summary_call.call_args.kwargs["messages"][0]["content"]
    assert len(prompt) <= compressor._SUMMARY_INPUT_MAX_CHARS + 20_000
    assert "chars elided" in prompt


def test_sampling_is_bounded_ordered_and_reaches_latest_input():
    content = "".join(
        f"<segment-{index:02d}>" + "x" * 50_000
        for index in range(10)
    )
    sampled = ContextCompressor._sample_summary_input(content)
    seen = [
        index
        for index in range(10)
        if f"<segment-{index:02d}>" in sampled
    ]

    assert len(sampled) <= ContextCompressor._SUMMARY_INPUT_MAX_CHARS
    assert "chars elided" in sampled
    assert seen == sorted(seen)
    assert any(index >= 5 for index in seen)
    assert content[-500:] in sampled


def test_legacy_mode_retains_head_tail_input_bound():
    compressor = _compressor(tail_mode="legacy")
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("## Historical Task Snapshot\nNone."),
    ) as summary_call:
        compressor._generate_summary(_turns(rounds=120))

    prompt = summary_call.call_args.kwargs["messages"][0]["content"]
    assert "summary input truncated" in prompt
