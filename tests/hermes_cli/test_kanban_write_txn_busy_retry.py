"""write_txn BUSY-retry behaviour.

These tests target the transaction boundary (BEGIN IMMEDIATE / COMMIT) only.
On unmodified main write_txn has no application-level retry, so the
"transient BUSY is absorbed" and "persistent BUSY is bounded" cases fail until
the fix lands. No real DB is touched: a fake connection records and replays
scripted boundary outcomes.
"""

import sqlite3

import pytest

from hermes_cli import kanban_db as kb


class _FakeConn:
    """Records execute() calls and replays a scripted result per SQL statement.

    script maps an uppercased SQL prefix to a list of outcomes consumed in
    order. An outcome is either an Exception (raised) or None (success).
    """

    def __init__(self, script):
        self._script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def execute(self, sql, *args):
        self.calls.append(sql)
        key = sql.strip().split()[0].upper()
        outcomes = self._script.get(key)
        if outcomes:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return None

    def count(self, prefix):
        prefix = prefix.upper()
        return sum(1 for c in self.calls if c.strip().upper().startswith(prefix))


def _busy():
    return sqlite3.OperationalError("database is locked")


def _other():
    return sqlite3.OperationalError("no such table: tasks")


@pytest.fixture(autouse=True)
def _no_file_check(monkeypatch):
    # Isolate the boundary behaviour from the post-commit invariant.
    monkeypatch.setattr(kb, "_check_file_length_invariant", lambda conn: None)


def test_retry_sleep_respects_floor(monkeypatch):
    # The jitter has a floor so a retry can't busy-spin back into the collision.
    slept = []
    monkeypatch.setattr(kb.time, "sleep", lambda s: slept.append(s))
    conn = _FakeConn({"BEGIN": [_busy(), _busy(), None]})
    with kb.write_txn(conn):
        pass
    assert slept
    assert all(s >= kb._BUSY_RETRY_MIN_S for s in slept)
    assert all(s <= kb._BUSY_RETRY_MAX_S for s in slept)


def test_transient_busy_at_begin_is_absorbed():
    conn = _FakeConn({"BEGIN": [_busy(), None]})
    with kb.write_txn(conn):
        pass
    assert conn.count("BEGIN") == 2
    assert conn.count("COMMIT") == 1


def test_persistent_busy_at_commit_rolls_back():
    # Exhausted COMMIT leaves the txn open; write_txn must ROLLBACK before
    # re-raising so the connection isn't poisoned for the next transaction.
    conn = _FakeConn({"COMMIT": [_busy()] * 50})
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with kb.write_txn(conn):
            pass
    assert conn.count("ROLLBACK") == 1


def test_on_commit_runs_after_durable_commit_before_integrity_check(monkeypatch):
    conn = _FakeConn({})
    observations = []

    def _on_commit():
        observations.append(("callback", conn.count("COMMIT")))

    monkeypatch.setattr(
        kb,
        "_check_file_length_invariant",
        lambda _conn: observations.append(("integrity", conn.count("COMMIT"))),
    )

    with kb.write_txn(conn, on_commit=_on_commit):
        pass

    assert observations == [("callback", 1), ("integrity", 1)]


def test_on_commit_does_not_run_when_commit_fails():
    conn = _FakeConn({"COMMIT": [_other()]})
    callbacks = []

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        with kb.write_txn(conn, on_commit=lambda: callbacks.append("called")):
            pass

    assert callbacks == []
    assert conn.count("ROLLBACK") == 1


def test_on_commit_rejects_savepoint_semantics():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("BEGIN")
        with pytest.raises(RuntimeError, match="outer durable transaction"):
            with kb.write_txn(conn, allow_nested=True, on_commit=lambda: None):
                pass
    finally:
        conn.rollback()
        conn.close()
