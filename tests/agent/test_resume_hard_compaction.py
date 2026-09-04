import json
from types import SimpleNamespace

import pytest

import agent.resume_hard_compaction as hard_compaction
from agent.compression_v3 import estimate_provider_wire_tokens
from agent.resume_hard_compaction import (
    ResumeHardCompactionUnsafe,
    ResumeHardCompactionPolicy,
    _LeaseRefresher,
    compact_oversized_resume,
)
from hermes_state import SessionCompactionSourceChangedError, SessionDB


HUMAN_PROVENANCE = {
    "origin_kind": "human_user",
    "turn_kind": "prompt",
    "trust_kind": "user_authorized",
}


def _seed(db: SessionDB, count: int) -> None:
    db.create_session("chat", source="tui")
    db.append_messages_batch(
        "chat",
        [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"message-{index}",
                **(HUMAN_PROVENANCE if index % 2 == 0 else {}),
            }
            for index in range(count)
        ],
    )


def test_hard_compaction_streams_and_preserves_lossless_archive(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 1_000)
    seen_batches = []

    def summarize(batch, previous):
        seen_batches.append(
            (
                len(batch),
                previous is not None,
                batch[0]["role"],
                batch[0]["_compressed_summary_has_user_turn"],
            )
        )
        return f"rolling summary through {batch[-1]['content']}"

    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=21, page_rows=37, summary_chunk_rows=41
        ),
        summarize=summarize,
    )

    assert result.outcome == "committed"
    assert result.source_rows == 1_000
    assert result.active_rows <= 21
    assert result.used_generated_summary is True
    assert seen_batches == [(1, False, "assistant", True)]
    active = db.get_model_resume_conversation("chat")
    assert active[0]["_compressed_summary"] is True
    assert active[-1]["content"] == "message-999"
    raw = db.get_messages("chat", include_inactive=True)
    assert len(raw) >= 1_000 + result.active_rows
    assert sum(1 for row in raw if row["active"] == 0 and row["compacted"] == 1) >= 1_000


def test_hard_compaction_keeps_concurrent_append_after_watermark(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 80)
    appended = False

    def summarize(batch, previous):
        nonlocal appended
        if not appended:
            appended = True
            db.append_message("chat", "user", "arrived during compaction")
        return "bounded summary"

    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=11, page_rows=13, summary_chunk_rows=17
        ),
        summarize=summarize,
    )

    assert result.outcome == "committed"
    active = db.get_model_resume_conversation("chat")
    assert active[-1]["content"] == "arrived during compaction"
    assert sum(
        row["content"] == "arrived during compaction"
        for row in db.get_messages("chat", include_inactive=True)
    ) == 2


def test_hard_compaction_uses_bounded_deterministic_fallback(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 60)

    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=9, page_rows=7, summary_chunk_rows=8
        ),
        summarize=lambda *_args: None,
    )

    assert result.outcome == "committed"
    assert result.used_generated_summary is False
    active = db.get_model_resume_conversation("chat")
    assert len(active) <= 9
    assert active[0]["_compressed_summary"] is True
    assert "deterministic fallback" in active[0]["content"]


def test_tail_alignment_feeds_displaced_prefix_back_into_summary(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 23)
    summarized = []

    def summarize(batch, previous):
        summarized.append(batch[0]["content"])
        return "bounded summary"

    compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=6, page_rows=7, summary_chunk_rows=20
        ),
        summarize=summarize,
    )

    # The four-row ring begins at assistant message-19, then aligns to the
    # following human boundary. That displaced assistant must be summarized,
    # not silently discarded from both the summary and exact tail.
    assert "message-19" in summarized[0]
    active = db.get_model_resume_conversation("chat")
    assert active[0]["content"].endswith("message-20")


def test_committed_publication_survives_receipt_write_failure(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 40)
    real_finish = db.finish_context_compaction

    def fail_committed_receipt(session_id, *, owner, outcome):
        if outcome == "committed":
            raise RuntimeError("journal unavailable")
        return real_finish(session_id, owner=owner, outcome=outcome)

    monkeypatch.setattr(db, "finish_context_compaction", fail_committed_receipt)
    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(target_rows=8),
        summarize=lambda *_args: "bounded summary",
    )

    assert result.outcome == "committed"
    assert len(db.get_model_resume_conversation("chat")) <= 8


def test_legacy_unknown_user_tail_stays_exact_and_visible(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("chat", source="tui")
    for index in range(20):
        db.append_message(
            "chat",
            "user" if index % 2 == 0 else "assistant",
            f"legacy-{index}",
        )

    compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(target_rows=7),
        summarize=lambda *_args: "bounded summary",
    )

    active = db.get_model_resume_conversation("chat")
    assert len(active) <= 7
    assert active[0]["display_kind"] == "hidden"
    assert active[1]["_compressed_summary"] is True
    assert active[2]["content"] == "legacy-16"
    assert active[-1]["content"] == "legacy-19"


def test_single_huge_tail_payload_becomes_bounded_archive_reference(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 20)
    huge = "H" * 2_000_000
    huge_id = db.append_message("chat", "assistant", huge)

    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=8,
            max_tail_message_chars=4_000,
        ),
        summarize=lambda *_args: "bounded summary",
    )

    assert result.outcome == "committed"
    active = db.get_model_resume_conversation("chat")
    assert len(active[-1]["content"]) <= 4_000
    assert str(huge_id) in active[-1]["content"]
    assert "archived at message rows" in active[-1]["content"]
    raw_huge = next(
        row
        for row in db.get_messages("chat", include_inactive=True)
        if row["id"] == huge_id
    )
    assert raw_huge["content"] == huge
    assert raw_huge["active"] == 0
    assert raw_huge["compacted"] == 1


def test_single_huge_api_sidecar_cannot_bypass_tail_bound(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 20)
    huge_api = "ephemeral-context:" + "E" * 2_000_000
    source_id = db.append_message(
        "chat",
        "user",
        "visible request",
        api_content=huge_api,
        **HUMAN_PROVENANCE,
    )

    compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=8,
            max_tail_message_chars=4_000,
        ),
        summarize=lambda *_args: "bounded summary",
    )

    active = db.get_model_resume_conversation("chat")
    retained = active[-1]
    assert "api_content" not in retained
    assert "visible request" in retained["content"]
    assert f"archived at message row {source_id}" in retained["content"]
    raw_source = next(
        row
        for row in db.get_messages("chat", include_inactive=True)
        if row["id"] == source_id
    )
    assert raw_source["api_content"] == huge_api


def test_generated_summary_and_progress_are_independently_bounded(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 40)

    def broken_progress(_phase, _count):
        raise RuntimeError("renderer disconnected")

    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=8,
            max_summary_chars=1_000,
        ),
        summarize=lambda *_args: "S" * 100_000,
        progress=broken_progress,
    )

    assert result.outcome == "committed"
    summary = db.get_model_resume_conversation("chat")[0]
    assert summary["_compressed_summary"] is True
    assert len(summary["content"]) < 1_100
    assert "summary middle truncated" in summary["content"]


def test_invalid_candidate_preserves_original_rows(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 40)
    before = db.get_messages("chat", include_inactive=True)
    monkeypatch.setattr(
        hard_compaction,
        "validate_projection",
        lambda _candidate: SimpleNamespace(valid=False, reason="broken tool pair"),
    )

    with pytest.raises(ResumeHardCompactionUnsafe, match="broken tool pair"):
        compact_oversized_resume(
            db,
            "chat",
            policy=ResumeHardCompactionPolicy(target_rows=8),
            summarize=lambda *_args: "bounded summary",
        )

    assert db.get_messages("chat", include_inactive=True) == before


def test_large_tool_call_group_is_reduced_with_raw_archive_preserved(
    tmp_path, monkeypatch
):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 20)
    tool_call_id = "call-large"
    call_row_id = db.append_message(
        "chat",
        "assistant",
        "",
        tool_calls=[
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": "A" * 100_000,
                },
            }
        ],
    )
    db.append_message(
        "chat",
        "tool",
        "tool result",
        tool_call_id=tool_call_id,
    )

    reduced_rows = []
    real_add = hard_compaction._RollingReducer.add

    def observed_add(self, messages):
        reduced_rows.extend(dict(message) for message in messages)
        return real_add(self, messages)

    monkeypatch.setattr(hard_compaction._RollingReducer, "add", observed_add)

    compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=8,
            max_tail_message_chars=2_000,
        ),
        summarize=lambda *_args: "bounded summary",
    )

    active = db.get_model_resume_conversation("chat")
    assert not any(row.get("role") == "tool" for row in active)
    reduced_call = next(row for row in reduced_rows if row.get("tool_calls"))
    reduced_tool = next(row for row in reduced_rows if row.get("role") == "tool")
    assert reduced_call["tool_calls"][0]["id"] == tool_call_id
    assert reduced_tool["tool_call_id"] == tool_call_id
    assert str(call_row_id) in reduced_call["tool_calls"][0]["function"]["arguments"]
    raw_call = next(
        row
        for row in db.get_messages("chat", include_inactive=True)
        if row["id"] == call_row_id
    )
    assert raw_call["tool_calls"][0]["function"]["arguments"] == "A" * 100_000


def test_tool_group_wider_than_tail_is_reduced_without_orphan_loss(
    tmp_path, monkeypatch
):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 12)
    call_ids = [f"wide-call-{index}" for index in range(4)]
    db.append_message("chat", "user", "wide tool turn", **HUMAN_PROVENANCE)
    db.append_message(
        "chat",
        "assistant",
        "issuing wide tool group",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    )
    for index, call_id in enumerate(call_ids):
        db.append_message(
            "chat",
            "tool",
            f"wide-result-{index}",
            tool_call_id=call_id,
        )
    db.append_message("chat", "assistant", "wide tool turn complete")
    db.append_message("chat", "user", "latest request", **HUMAN_PROVENANCE)
    db.append_message("chat", "assistant", "latest answer")

    reduced_rows = []
    real_add = hard_compaction._RollingReducer.add

    def observed_add(self, messages):
        reduced_rows.extend(dict(message) for message in messages)
        return real_add(self, messages)

    monkeypatch.setattr(hard_compaction._RollingReducer, "add", observed_add)
    compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(target_rows=6),
        summarize=lambda *_args: "bounded summary",
    )

    reduced_contents = {row.get("content") for row in reduced_rows}
    assert {f"wide-result-{index}" for index in range(4)} <= reduced_contents
    assert any(row.get("tool_calls") for row in reduced_rows)
    active = db.get_model_resume_conversation("chat")
    assert [row["content"] for row in active[-2:]] == [
        "latest request",
        "latest answer",
    ]


def test_clipped_tool_call_list_reduces_every_result(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 120)
    call_ids = [f"clipped-call-{index}" for index in range(40)]
    db.append_message("chat", "user", "many tool calls", **HUMAN_PROVENANCE)
    db.append_message(
        "chat",
        "assistant",
        "issuing many calls",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "A" * 200,
                },
            }
            for call_id in call_ids
        ],
    )
    for index, call_id in enumerate(call_ids):
        db.append_message(
            "chat",
            "tool",
            f"clipped-result-{index}",
            tool_call_id=call_id,
        )
    db.append_message("chat", "assistant", "many calls complete")
    db.append_message("chat", "user", "latest request", **HUMAN_PROVENANCE)
    db.append_message("chat", "assistant", "latest answer")

    reduced_rows = []
    real_add = hard_compaction._RollingReducer.add

    def observed_add(self, messages):
        reduced_rows.extend(dict(message) for message in messages)
        return real_add(self, messages)

    monkeypatch.setattr(hard_compaction._RollingReducer, "add", observed_add)
    compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=100,
            max_tail_message_chars=2_000,
        ),
        summarize=lambda *_args: "bounded summary",
    )

    reduced_contents = {row.get("content") for row in reduced_rows}
    assert {f"clipped-result-{index}" for index in range(40)} <= reduced_contents
    assert any(row.get("tool_calls") for row in reduced_rows)
    active = db.get_model_resume_conversation("chat")
    assert [row["content"] for row in active[-2:]] == [
        "latest request",
        "latest answer",
    ]


def test_lease_refresher_tolerates_one_blip_but_bounds_persistent_failure():
    class RefreshDB:
        def __init__(self, results):
            self.results = list(results)
            self.calls = 0
            self.refresher = None

        def refresh_compression_lock(self, *_args, **_kwargs):
            self.calls += 1
            result = self.results.pop(0) if self.results else True
            if result and self.refresher is not None:
                self.refresher._stop.set()
            return result

    transient = RefreshDB([False, True])
    transient_refresher = _LeaseRefresher(
        transient, "chat", "holder", ttl=0.3
    )
    transient.refresher = transient_refresher
    transient_refresher._stop.wait = (
        lambda _interval: transient_refresher._stop.is_set()
    )
    transient_refresher._run()
    assert transient.calls == 2
    assert transient_refresher.lost is False

    persistent = RefreshDB([False] * 20)
    persistent_refresher = _LeaseRefresher(
        persistent, "chat", "holder", ttl=0.3
    )
    persistent_refresher._stop.wait = lambda _interval: False
    persistent_refresher._run()
    assert persistent.calls == persistent_refresher._max_consecutive_failures
    assert persistent_refresher.lost is True


def test_giant_prefix_and_tail_fields_are_bounded_before_python_reduction(
    tmp_path, monkeypatch
):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("chat", source="tui")
    huge = "Z" * 2_000_000
    prefix_id = db.append_message(
        "chat",
        "user",
        huge,
        api_content=huge,
        **HUMAN_PROVENANCE,
    )
    db.append_message(
        "chat",
        "assistant",
        "prefix answer",
        reasoning=huge,
        reasoning_details=[{"text": huge}],
    )
    structured = [{"type": "input_text", "text": huge}]
    structured_id = db.append_message("chat", "assistant", structured)
    for index in range(40):
        db.append_message(
            "chat",
            "user" if index % 2 == 0 else "assistant",
            f"middle-{index}",
            **(HUMAN_PROVENANCE if index % 2 == 0 else {}),
        )
    tail_user_id = db.append_message(
        "chat",
        "user",
        "latest exact request",
        api_content=huge,
        **HUMAN_PROVENANCE,
    )
    tail_assistant_id = db.append_message(
        "chat",
        "assistant",
        huge,
        tool_name=huge,
        reasoning_content=huge,
        codex_message_items=[{"payload": huge}],
    )

    materialized_page_bytes = []
    real_page = db.get_compaction_source_page

    def observed_page(*args, **kwargs):
        page = real_page(*args, **kwargs)
        materialized_page_bytes.append(
            len(json.dumps(page, ensure_ascii=False, default=str).encode("utf-8"))
        )
        return page

    monkeypatch.setattr(db, "get_compaction_source_page", observed_page)
    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=8,
            page_rows=20,
            max_summary_chars=1_000,
            max_tail_message_chars=2_000,
            max_projection_tokens=3_000,
            max_page_materialized_chars=100_000,
        ),
        summarize=lambda *_args: "bounded summary",
    )

    assert result.outcome == "committed"
    assert materialized_page_bytes
    assert max(materialized_page_bytes) < 150_000
    active = db.get_model_resume_conversation("chat")
    assert (
        estimate_provider_wire_tokens({"messages": active})
        <= 3_000
    )
    raw = {
        row["id"]: row
        for row in db.get_messages("chat", include_inactive=True)
        if row["id"]
        in {prefix_id, structured_id, tail_user_id, tail_assistant_id}
    }
    assert raw[prefix_id]["content"] == huge
    assert raw[prefix_id]["api_content"] == huge
    assert raw[structured_id]["content"] == structured
    assert raw[tail_user_id]["api_content"] == huge
    assert raw[tail_assistant_id]["content"] == huge
    assert raw[tail_assistant_id]["tool_name"] == huge
    assert raw[tail_assistant_id]["reasoning_content"] == huge


def test_source_mutation_during_paging_rejects_stale_publication(
    tmp_path, monkeypatch
):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 40)
    before_count = len(db.get_messages("chat"))
    real_page = db.get_compaction_source_page
    changed = False

    def mutate_after_first_page(*args, **kwargs):
        nonlocal changed
        page = real_page(*args, **kwargs)
        if page and not changed:
            changed = True
            db._execute_write(
                lambda conn: conn.execute(
                    "UPDATE messages SET api_content = ? "
                    "WHERE session_id = ? AND id = ("
                    "SELECT MIN(id) FROM messages WHERE session_id = ?)",
                    ("changed-after-read", "chat", "chat"),
                )
            )
        return page

    monkeypatch.setattr(
        db,
        "get_compaction_source_page",
        mutate_after_first_page,
    )
    with pytest.raises(SessionCompactionSourceChangedError):
        compact_oversized_resume(
            db,
            "chat",
            policy=ResumeHardCompactionPolicy(target_rows=8, page_rows=5),
            summarize=lambda *_args: "stale summary",
        )

    active = db.get_messages("chat")
    assert len(active) == before_count
    assert all(row["compacted"] == 0 for row in active)
    assert active[0]["api_content"] == "changed-after-read"

    retried = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(target_rows=8, page_rows=5),
        summarize=lambda *_args: "fresh summary",
    )
    assert retried.outcome == "committed"


def test_page_budget_must_cover_one_complete_bounded_row():
    with pytest.raises(ValueError, match="one worst-case bounded row"):
        ResumeHardCompactionPolicy(
            max_tail_message_chars=2_000,
            max_page_materialized_chars=20_000,
        ).validate()


def test_changed_source_projection_rearms_terminal_receipt(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 40)
    with monkeypatch.context() as patch:
        patch.setattr(
            hard_compaction,
            "validate_projection",
            lambda _candidate: SimpleNamespace(valid=False, reason="reject once"),
        )
        with pytest.raises(ResumeHardCompactionUnsafe):
            compact_oversized_resume(
                db,
                "chat",
                policy=ResumeHardCompactionPolicy(target_rows=8),
                summarize=lambda *_args: "bounded summary",
            )

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = ? WHERE session_id = ? AND id = ("
            "SELECT MIN(id) FROM messages WHERE session_id = ?)",
            ("changed-in-place", "chat", "chat"),
        )
    )
    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(target_rows=8),
        summarize=lambda *_args: "bounded summary",
    )
    assert result.outcome == "committed"


def test_changed_route_strategy_rearms_terminal_receipt(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    _seed(db, 40)
    with monkeypatch.context() as patch:
        patch.setattr(
            hard_compaction,
            "validate_projection",
            lambda _candidate: SimpleNamespace(valid=False, reason="reject once"),
        )
        with pytest.raises(ResumeHardCompactionUnsafe):
            compact_oversized_resume(
                db,
                "chat",
                policy=ResumeHardCompactionPolicy(
                    target_rows=8,
                    strategy_identity="route-a",
                ),
                summarize=lambda *_args: "bounded summary",
            )

    result = compact_oversized_resume(
        db,
        "chat",
        policy=ResumeHardCompactionPolicy(
            target_rows=8,
            strategy_identity="route-b",
        ),
        summarize=lambda *_args: "bounded summary",
    )
    assert result.outcome == "committed"
