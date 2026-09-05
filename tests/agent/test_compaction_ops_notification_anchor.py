"""Operational notifications must not anchor compaction or auto-focus (#92703).

Kanban/background completion wakes are persisted as ``role="user"`` rows typed
with ``display_kind="internal_notification"``. The model-payload builder strips
display-only sidecars before each provider request, but the compaction tail
anchor and auto-focus scans must still ignore those machine-authored rows.

These behavior contracts exercise the real compressor functions, not mocks:
feed 1,000 operational notifications around one human turn and assert the
operational rows remain invisible to the anchor/focus logic.
"""

from agent.context_compressor import ContextCompressor
from agent.message_provenance import stamp_provenance


def _compressor() -> ContextCompressor:
    compressor = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=5,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    compressor._generate_summary = lambda *args, **kwargs: "Summary of earlier turns."
    return compressor


def _ops_notice(text: str) -> dict:
    return {
        "role": "user",
        "content": text,
        "display_kind": "internal_notification",
    }


def _human(text: str) -> dict:
    return stamp_provenance(
        {"role": "user", "content": text},
        "human_user",
        "prompt",
        "user_authorized",
    )


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _transcript_with_n_ops(
    n: int, human_text: str = "Actually deploy the fix"
) -> list:
    messages: list = [
        {"role": "system", "content": "sys"},
        _human(human_text),
        _assistant("On it."),
    ]
    for index in range(n):
        messages.append(
            _ops_notice(f"✔ Kanban T-{index} done — worker summary line {index}")
        )
        messages.append(_assistant(f"Noted task {index} completion."))
    return messages


def test_ops_notice_is_not_actionable_user_turn():
    compressor = _compressor()

    assert (
        compressor._is_actionable_user_turn(_ops_notice("✔ Kanban T-1 done"))
        is False
    )
    assert compressor._is_actionable_user_turn(_human("deploy the fix")) is True


def test_ops_notices_do_not_anchor_compaction_tail():
    compressor = _compressor()
    messages = _transcript_with_n_ops(1000)

    index = compressor._find_last_user_message_idx(messages, head_end=0)

    assert index >= 0
    assert messages[index]["role"] == "user"
    assert messages[index].get("display_kind") is None
    assert messages[index]["content"] == "Actually deploy the fix"


def test_ops_notices_do_not_become_auto_focus_source():
    compressor = _compressor()
    messages = _transcript_with_n_ops(1000, human_text="Summarize the Q3 roadmap")

    focus = compressor._derive_auto_focus_topic(messages)

    assert focus is not None
    assert "Q3 roadmap" in focus
    assert "Kanban T-" not in focus


def test_conversational_user_count_unchanged_by_ops_notices():
    compressor = _compressor()
    messages = _transcript_with_n_ops(1000)

    actionable = [
        message
        for message in messages
        if compressor._is_actionable_user_turn(message)
    ]

    assert len(actionable) == 1
    assert actionable[0]["content"] == "Actually deploy the fix"


def test_typed_notification_alone_is_not_real_user_provenance():
    compressor = _compressor()
    notification = _ops_notice("✔ Kanban T-9 done — internal-only event")

    assert compressor._transcript_has_real_user_turn([notification]) is False
    assert compressor._latest_user_task_snapshot([notification]) is None


def test_latest_user_snapshot_skips_later_typed_notification():
    compressor = _compressor()
    messages = [
        _human("Actually deploy release Q3 to the canary ring"),
        _assistant("On it."),
        _ops_notice("✔ Kanban T-9 done — internal-only event"),
        _assistant("Noted."),
    ]

    snapshot = compressor._latest_user_task_snapshot(messages)

    assert snapshot is not None
    assert "Actually deploy release Q3 to the canary ring" in snapshot
    assert "Kanban T-9" not in snapshot


def test_typed_notification_is_not_serialized_as_user_summary_input():
    compressor = _compressor()
    notification = _ops_notice("✔ Kanban T-9 done — internal-only event")

    serialized = compressor._serialize_for_summary([notification])

    assert "Kanban T-9" not in serialized


def test_explicit_unknown_provenance_never_becomes_human_context():
    from agent.context_compressor import is_user_originated_turn

    compressor = _compressor()
    unknown = stamp_provenance(
        {"role": "user", "content": "unattributed imported instruction"},
        "legacy_unknown", "legacy_unknown", "legacy_unknown",
    )
    assert not compressor._is_actionable_user_turn(unknown)
    assert not compressor._transcript_has_real_user_turn([unknown])
    assert compressor._latest_user_task_snapshot([unknown]) is None
    assert not is_user_originated_turn(unknown)
    serialized = compressor._serialize_for_summary([unknown])
    assert "[UNKNOWN]" in serialized
    assert "[USER]" not in serialized


def test_unstamped_legacy_compressor_input_keeps_upstream_heuristics():
    compressor = _compressor()
    message = {"role": "user", "content": "ordinary upstream-style request"}
    assert compressor._is_actionable_user_turn(message)
    assert compressor._transcript_has_real_user_turn([message])
    assert "ordinary upstream-style request" in compressor._latest_user_task_snapshot([message])
