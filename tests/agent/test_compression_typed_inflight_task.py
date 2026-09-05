"""Upstream in-flight replay must retain trusted non-human task identity."""
from copy import deepcopy

import pytest

from agent.context_compressor import (
    ContextCompressor, SUMMARY_PREFIX, _SUMMARY_END_MARKER,
    _INFLIGHT_TASK_REPLAY_HEADER,
)
from agent.message_provenance import (
    SEMANTIC_FIELDS, is_human_intent, provenance_for_runtime_turn,
)


def _task(platform, source):
    return {
        "role": "user", "content": "Finish the scheduled audit",
        **provenance_for_runtime_turn(
            platform=platform, metadata={"source": source},
        ).as_message_fields(),
    }


def _pending():
    return [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call1", "type": "function",
            "function": {"name": "terminal", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call1", "content": "partial result"},
    ]


@pytest.mark.parametrize("platform,source", [
    ("cron", "cron"), ("cron", ""), ("subagent", "delegation"), ("subagent", ""),
])
@pytest.mark.parametrize("merge_on_carrier", [False, True])
def test_typed_task_survives_repeated_handoffs(platform, source, merge_on_carrier):
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.quiet_mode = True
    task = _task(platform, source)
    original = deepcopy(task)
    transcript = [task, *_pending()]
    for _ in range(3):
        inflight = compressor._find_inflight_user_task(transcript)
        assert inflight is not None
        carrier = {"role": "user", "content": SUMMARY_PREFIX + "\nNotes\n" + _SUMMARY_END_MARKER}
        tail = _pending() if merge_on_carrier else [{"role": "assistant", "content": "Progress"}]
        compressed = compressor._reappend_inflight_user_task([carrier, *tail], inflight)
        replay = compressed[0] if merge_on_carrier else compressed[-1]
        assert replay["content"].count(_INFLIGHT_TASK_REPLAY_HEADER) == 1
        assert "Finish the scheduled audit" in replay["content"]
        assert not is_human_intent(replay)
        assert {k: replay[k] for k in SEMANTIC_FIELDS if k in replay} == {
            k: original[k] for k in SEMANTIC_FIELDS if k in original
        }
        # Re-running assembly must not duplicate an already-visible task.
        before = deepcopy(compressed)
        assert compressor._reappend_inflight_user_task(compressed, inflight) == before
        from agent.conversation_compression import _ensure_compressed_has_user_turn
        assert _ensure_compressed_has_user_turn(transcript, compressed) == "already_present"
        assert compressed == before
        transcript = [*compressed, *_pending()]
    assert task == original


@pytest.mark.parametrize("platform,source", [
    ("cron", "cron"), ("subagent", "delegation"),
])
def test_completed_trusted_task_is_not_reanimated(platform, source):
    assert ContextCompressor._find_inflight_user_task([
        _task(platform, source), {"role": "assistant", "content": "Audit complete"},
    ]) is None


@pytest.mark.parametrize("source", ["external_webhook", "plugin"])
def test_untrusted_event_is_not_promoted_to_live_task(source):
    assert ContextCompressor._find_inflight_user_task([_task("cron", source), *_pending()]) is None
