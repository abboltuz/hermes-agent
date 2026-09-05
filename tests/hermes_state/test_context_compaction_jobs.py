"""Durable admission contracts, using independent real SQLite connections."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("conversation", "test")
    return path


def claim(db, *, source="source", strategy="strategy", owner="worker", **kwargs):
    return db.claim_context_compaction(
        "conversation", source, strategy, owner=owner, **kwargs,
    )


@pytest.mark.parametrize("outcome", ["no_progress", "timed_out", "aborted", "committed"])
def test_terminal_survives_database_and_worker_restart(db_path, outcome):
    with SessionDB(db_path) as db:
        assert claim(db) == "admitted"
        assert db.finish_context_compaction("conversation", owner="worker", outcome=outcome)
    with SessionDB(db_path) as resumed:
        assert claim(resumed, owner="new-worker") == "no_progress_suppressed"
        assert claim(resumed, source="new-source", owner="new-worker") == "admitted"


def test_changed_strategy_and_explicit_force_rearm_terminal(db_path):
    with SessionDB(db_path) as db:
        assert claim(db) == "admitted"
        db.finish_context_compaction("conversation", owner="worker", outcome="no_progress")
        assert claim(db, strategy="new-policy") == "admitted"
        db.finish_context_compaction("conversation", owner="worker", outcome="no_progress")
        assert claim(db, strategy="new-policy", force=True) == "admitted"


@pytest.mark.parametrize("outcome", ["cooldown", "deferred_lock", "native_delegated"])
def test_nonexecution_does_not_exhaust_attempt(db_path, outcome):
    with SessionDB(db_path) as db:
        assert claim(db) == "admitted"
        db.finish_context_compaction("conversation", owner="worker", outcome=outcome)
        assert claim(db, owner="new-worker") == "admitted"


def test_only_one_connection_admitted_and_force_cannot_steal_live_work(db_path):
    with SessionDB(db_path) as first, SessionDB(db_path) as second:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda pair: claim(pair[0], owner=pair[1]), [
                (first, "first"), (second, "second"),
            ]))
        assert sorted(results) == ["admitted", "joined"]
        assert claim(first, source="other-source", force=True) == "deferred_lock"
        assert claim(first, force=True) == "joined"


def test_expired_attempt_is_terminal_and_stale_owner_cannot_finish_replacement(db_path):
    with SessionDB(db_path) as db:
        assert claim(db, now=100, lease_seconds=10) == "admitted"
        assert claim(db, owner="new", now=111) == "no_progress_suppressed"
        assert claim(db, owner="new", now=112, force=True) == "admitted"
        assert not db.finish_context_compaction("conversation", owner="worker", outcome="committed")
        assert claim(db, owner="third", now=113) == "joined"


def test_profile_and_conversation_isolation(db_path, tmp_path):
    with SessionDB(db_path) as first, SessionDB(tmp_path / "second.db") as second:
        second.create_session("conversation", "test")
        first.create_session("other", "test")
        assert claim(first) == "admitted"
        assert claim(second) == "admitted"
        assert first.claim_context_compaction("other", "source", "strategy", owner="worker") == "admitted"


def test_missing_session_is_explicit_and_invalid_finish_cannot_release_owner(db_path):
    with SessionDB(db_path) as db:
        assert db.claim_context_compaction("absent", "source", "strategy", owner="worker") == "unpersisted"
        assert claim(db) == "admitted"
        with pytest.raises(ValueError):
            db.finish_context_compaction("conversation", owner="worker", outcome="invented")
        assert claim(db, owner="another") == "joined"


def test_writable_open_migrates_legacy_store_without_rewriting_transcript(db_path):
    with SessionDB(db_path) as db:
        db.append_messages_batch("conversation", [{"role": "user", "content": "original"}])
        before = db.get_messages("conversation")
        # Simulate a database from before the additive lifecycle table.
        db._execute_write(lambda conn: conn.execute("DROP TABLE context_compaction_jobs"))
    with SessionDB(db_path) as migrated:
        assert migrated.get_messages("conversation") == before
        assert claim(migrated) == "admitted"
