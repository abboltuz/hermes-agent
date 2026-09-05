"""Snapshot publication, bounded reads and legacy-writer coherence."""

from contextlib import contextmanager
import json

import pytest

from hermes_state import SessionDB
from hermes_state_snapshots import (
    WorkingContextSnapshotError,
    WorkingContextTooLargeError,
)


@pytest.fixture
def db(tmp_path):
    with SessionDB(tmp_path / "state.db") as database:
        database.create_session("chat", "test")
        database.append_messages_batch(
            "chat",
            [
                {"role": "user", "content": "original task"},
                {"role": "assistant", "content": "original answer"},
            ],
        )
        yield database


def compact(db, text="summary", **kwargs):
    return db.archive_and_compact(
        "chat",
        [
            {
                "role": "user",
                "content": text,
                "_compressed_summary": True,
                "api_content": " exact wire " + text,
            }
        ],
        **kwargs,
    )


def head(db):
    with db._read_ctx() as conn:
        row = conn.execute(
            "SELECT s.* FROM working_context_heads h JOIN working_context_snapshots s "
            "ON s.id = h.snapshot_id WHERE h.session_id = 'chat'",
        ).fetchone()
        return dict(row) if row is not None else None


def test_publication_has_only_references_and_survives_restart(db):
    assert db.has_working_context_snapshot("chat") is False
    original_ids = [row["id"] for row in db.get_messages("chat")]
    compact(db)
    assert db.has_working_context_snapshot("chat") is True
    snapshot = head(db)
    active = db.get_messages("chat")
    assert snapshot["generation"] == 1
    assert json.loads(snapshot["message_ids"]) == [row["id"] for row in active]
    assert snapshot["row_count"] == 1
    assert "summary" not in json.dumps(snapshot)
    assert set(original_ids) <= {
        row["id"] for row in db.get_messages("chat", include_inactive=True)
    }
    path = db.db_path
    expected = db.get_messages_as_conversation("chat", include_row_ids=True)
    with SessionDB(path) as resumed:
        assert (
            resumed.get_messages_as_conversation("chat", include_row_ids=True)
            == expected
        )
        model, display = resumed.get_resume_conversations("chat")
        assert model[0]["_compressed_summary"] is True
        assert model[0]["api_content"] == " exact wire summary"
        assert display[0]["_row_id"] == expected[0]["_row_id"]
        model_only = resumed.get_model_resume_conversation("chat")
        assert model_only == model


def test_concurrent_append_is_tail_and_is_not_cloned_by_read(db):
    compact(db)
    snapshot = head(db)
    with SessionDB(db.db_path) as sibling:
        sibling.append_message("chat", "assistant", "new answer")
        tail_id = sibling.get_messages("chat")[-1]["id"]
    rows = db.get_messages_as_conversation("chat", include_row_ids=True)
    assert [row["content"] for row in rows] == ["summary", "new answer"]
    assert rows[-1]["_row_id"] == tail_id
    assert head(db) == snapshot
    assert len(db.get_messages("chat", include_inactive=True)) == 4


def test_new_generation_and_concurrent_compaction_tail_are_published_together(db):
    compact(db)
    first = head(db)
    watermark = db.get_active_message_watermark("chat")
    db.append_message("chat", "assistant", "arrived during summary")
    compact(db, "next summary", watermark=watermark)
    second = head(db)
    assert second["generation"] == first["generation"] + 1
    expected = db.get_messages("chat")
    assert json.loads(second["message_ids"]) == [row["id"] for row in expected]
    assert [row["content"] for row in db.get_messages_as_conversation("chat")] == [
        "next summary",
        "arrived during summary",
    ]


def test_publication_from_empty_manifest_clears_the_now_covered_tail(db):
    db.archive_and_compact("chat", [])
    assert json.loads(head(db)["message_ids"]) == []
    assert db.get_messages_as_conversation("chat") == []
    compact(db)
    assert len(db.get_messages_as_conversation("chat")) == 1
    with db._read_ctx() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM working_context_tail").fetchone()[0] == 0
        )


def test_old_manifest_format_never_assumes_an_empty_new_tail_registry(db):
    compact(db)
    db.append_message("chat", "assistant", "unregistered old-writer tail")

    def old_format(conn):
        conn.execute("UPDATE working_context_snapshots SET format_version = 1")
        conn.execute("DELETE FROM working_context_tail")

    db._execute_write(old_format)
    with SessionDB(db.db_path) as reopened:
        assert reopened._read_working_context_rows("chat") is None
        assert [
            row["content"] for row in reopened.get_messages_as_conversation("chat")
        ] == [
            "summary",
            "unregistered old-writer tail",
        ]
        compact(reopened)
        assert head(reopened)["format_version"] == 2


@pytest.mark.parametrize(
    "limit_name,limit",
    [("MAX_WORKING_CONTEXT_ROWS", 0), ("MAX_WORKING_CONTEXT_BYTES", 1)],
)
def test_rejected_candidate_rolls_back_rows_and_previous_head(
    db, monkeypatch, limit_name, limit
):
    compact(db)
    before = db.get_messages("chat", include_inactive=True)
    previous = head(db)
    monkeypatch.setattr("hermes_state_snapshots." + limit_name, limit)
    with pytest.raises(WorkingContextTooLargeError):
        compact(db, "cannot publish")
    assert db.get_messages("chat", include_inactive=True) == before
    assert head(db) == previous


def test_tail_row_overflow_is_explicit_not_truncated(db, monkeypatch):
    compact(db)
    db.append_message("chat", "assistant", "tail")
    monkeypatch.setattr("hermes_state_snapshots.MAX_WORKING_CONTEXT_ROWS", 1)
    with pytest.raises(WorkingContextTooLargeError):
        db.get_resume_conversations("chat")
    assert len(db.get_messages("chat")) == 2


def test_tail_byte_overflow_is_rejected_before_body_materialization(db, monkeypatch):
    compact(db)
    db.append_message("chat", "assistant", "huge tail " * 1000)
    monkeypatch.setattr("hermes_state_snapshots.MAX_WORKING_CONTEXT_BYTES", 1000)
    queries = []
    original = db._read_ctx

    @contextmanager
    def traced():
        with original() as conn:
            conn.set_trace_callback(queries.append)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    monkeypatch.setattr(db, "_read_ctx", traced)
    with pytest.raises(WorkingContextTooLargeError):
        db.get_messages_as_conversation("chat")
    assert not any("SELECT m.id, m.role, m.content" in query for query in queries)


@pytest.mark.parametrize("operation", ["replace", "deactivate", "delete"])
def test_legacy_membership_mutation_invalidates_head_atomically(db, operation):
    compact(db)
    if operation == "replace":
        db.replace_messages(
            "chat", [{"role": "user", "content": "replacement"}], archive_dropped=True
        )
    elif operation == "deactivate":
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET active = 0 WHERE session_id = 'chat'"
            )
        )
    else:
        db._execute_write(
            lambda conn: conn.execute(
                "DELETE FROM messages WHERE session_id = 'chat' AND active = 1"
            )
        )
    assert head(db) is None
    assert db._read_working_context_rows("chat") is None
    assert len(db.get_messages_as_conversation("chat")) == (
        1 if operation == "replace" else 0
    )


def test_content_sidecar_and_presentation_updates_are_read_through(db):
    compact(db)
    previous = head(db)
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = 'edited', api_content = ' exact edited wire ', "
            "display_kind = 'hidden' WHERE session_id = 'chat' AND active = 1",
        )
    )
    assert head(db) == previous  # manifest membership is unchanged
    model, _ = db.get_resume_conversations("chat")
    assert model[0]["content"] == "edited"
    assert model[0]["api_content"] == " exact edited wire "
    assert model[0]["display_kind"] == "hidden"


@pytest.mark.parametrize("manifest", ["not json", "[1,1]", "[true]", "[999999999]"])
def test_corrupt_manifest_never_falls_back_to_partial_replay(db, manifest):
    compact(db)
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE working_context_snapshots SET message_ids = ?",
            (manifest,),
        )
    )
    with pytest.raises(WorkingContextSnapshotError):
        db.get_messages_as_conversation("chat")


def test_publication_failure_restores_previous_head(db, monkeypatch):
    compact(db)
    previous = head(db)
    original = db._publish_working_context_snapshot

    def fail_after_publication(conn, session_id):
        original(conn, session_id)
        raise RuntimeError("interrupted before commit")

    monkeypatch.setattr(db, "_publish_working_context_snapshot", fail_after_publication)
    with pytest.raises(RuntimeError, match="interrupted"):
        compact(db, "uncommitted")
    assert head(db) == previous
    assert db.get_messages_as_conversation("chat")[0]["content"] == "summary"


def test_fast_resume_reads_by_primary_key_and_bounded_tail_not_history_scan(db):
    compact(db)
    snapshot = head(db)
    with db._read_ctx() as conn:
        reference_plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT m.content FROM json_each(?) refs "
            "CROSS JOIN messages m NOT INDEXED "
            "WHERE m.id = refs.value AND m.session_id = ? AND m.active = 1",
            (snapshot["message_ids"], "chat"),
        ).fetchall()
        tail_plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT message_id FROM working_context_tail "
            "WHERE session_id = ? AND message_id > ? ORDER BY message_id LIMIT ?",
            ("chat", snapshot["watermark"], 8193),
        ).fetchall()
    assert any("INTEGER PRIMARY KEY" in row[3] for row in reference_plan)
    assert any(
        "COVERING INDEX" in row[3] and "(session_id=? AND message_id>?)" in row[3]
        for row in tail_plan
    )


def test_read_transaction_is_released_on_failure(db, monkeypatch):
    compact(db)
    monkeypatch.setattr("hermes_state_snapshots.MAX_WORKING_CONTEXT_BYTES", 1)
    with pytest.raises(WorkingContextTooLargeError):
        db.get_messages_as_conversation("chat")
    with db._read_ctx() as conn:
        assert not conn.in_transaction


def test_sibling_rewrite_cannot_mix_manifest_and_new_rows(db, monkeypatch):
    compact(db)
    if not db._wal_active:
        pytest.skip("concurrent writer/read snapshot requires WAL")
    original = db._working_context_rows_by_id
    with SessionDB(db.db_path) as sibling:

        def rewrite_during_read(conn, session_id, ids, **kwargs):
            sibling.replace_messages(
                "chat",
                [{"role": "user", "content": "new revision"}],
                archive_dropped=True,
            )
            return original(conn, session_id, ids, **kwargs)

        monkeypatch.setattr(db, "_working_context_rows_by_id", rewrite_during_read)
        assert db.get_messages_as_conversation("chat")[0]["content"] == "summary"
        monkeypatch.setattr(db, "_working_context_rows_by_id", original)
        assert db.get_messages_as_conversation("chat")[0]["content"] == "new revision"


def test_stale_compression_owner_cannot_publish_snapshot(db):
    from hermes_state import SessionCompressionInProgressError

    compact(db)
    previous = head(db)
    before = db.get_messages("chat", include_inactive=True)
    assert db.try_acquire_compression_lock("chat", "winner")
    with pytest.raises(SessionCompressionInProgressError):
        compact(db, "stale candidate", lock_holder="loser")
    assert head(db) == previous
    assert db.get_messages("chat", include_inactive=True) == before
    db.release_compression_lock("chat", "winner")


def test_legacy_readonly_database_remains_readable_without_snapshot_tables(db):
    def remove_additive_schema(conn):
        for name in (
            "working_context_message_delete",
            "working_context_message_update",
            "working_context_message_insert",
            "working_context_tail_insert",
        ):
            conn.execute("DROP TRIGGER " + name)
        conn.execute("DROP TABLE working_context_tail")
        conn.execute("DROP TABLE working_context_heads")
        conn.execute("DROP TABLE working_context_snapshots")

    db._execute_write(remove_additive_schema)
    with SessionDB(db.db_path, read_only=True) as legacy:
        assert [
            row["content"] for row in legacy.get_messages_as_conversation("chat")
        ] == [
            "original task",
            "original answer",
        ]
    with SessionDB(db.db_path) as upgraded:
        assert upgraded.get_messages("chat") == db.get_messages("chat")
        compact(upgraded)
        assert head(upgraded)["generation"] == 1


@pytest.mark.parametrize("archive_after_publication", [False, True])
def test_large_archived_history_does_not_scale_public_resume_query_work(
    db, monkeypatch, archive_after_publication
):
    # Synthetic rows only. The archive is much larger than the working set;
    # instruction counts avoid hardware-dependent wall-clock assertions.
    if archive_after_publication:
        compact(db)
    db._execute_write(
        lambda conn: conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp, active, compacted) "
            "VALUES ('chat', 'assistant', ?, 1, 0, 1)",
            [("archived exact result " + str(i),) for i in range(10_000)],
        )
    )
    if not archive_after_publication:
        compact(db)
    assert head(db) is not None
    db.append_message("chat", "assistant", "new active tail")
    callbacks = []
    original = db._read_ctx

    @contextmanager
    def bounded_work():
        with original() as conn:

            def progress():
                callbacks.append(1)
                return 0

            conn.set_progress_handler(progress, 100)
            try:
                yield conn
            finally:
                conn.set_progress_handler(None, 0)

    monkeypatch.setattr(db, "_read_ctx", bounded_work)
    model, display = db.get_resume_conversations("chat")
    assert model[0]["content"] == display[0]["content"] == "summary"
    assert model[-1]["content"] == display[-1]["content"] == "new active tail"
    assert len(callbacks) < 100
    assert len(db.get_messages("chat", include_inactive=True)) == 10_004
