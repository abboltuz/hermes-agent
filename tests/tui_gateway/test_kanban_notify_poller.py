"""Tests for the TUI-side kanban notification poller (issue #59890).

``kanban_create`` auto-subscribes TUI/desktop sessions with
``platform="tui"`` / ``chat_id=HERMES_SESSION_KEY``, but no component ever
read those rows back: the gateway notifier skips them (no "tui" messaging
adapter) and the TUI notification poller only watched process completions.
``last_event_id`` stayed 0 forever and no notification was ever delivered.

These tests cover the delivery half that now lives in tui_gateway/server.py:
``_collect_kanban_notifications`` (cursor claim + formatting + archive-only
unsubscribe) and ``_format_kanban_event_text``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from tui_gateway.server import (
    _collect_kanban_notifications,
    _format_kanban_event_text,
)

SESSION_KEY = "tui-session-key-1"


def _session(key: str = SESSION_KEY) -> dict:
    return {"session_key": key}


def _create_subscribed_task(*, chat_id: str = SESSION_KEY, platform: str = "tui"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify tui", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform=platform, chat_id=chat_id)
        return tid
    finally:
        conn.close()


def _complete(tid: str, summary: str = "all done") -> None:
    conn = kb.connect()
    try:
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()


def _sub_rows(tid: str) -> list:
    conn = kb.connect()
    try:
        return kb.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()


class TestCollectKanbanNotifications:
    def test_zero_sub_board_is_never_opened_writable(self):
        conn = kb.connect()
        conn.close()
        kb.create_board("second-board")

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()

    def test_done_reopen_notifies_once_per_event_until_archive(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="shipped the fix")

        first = _collect_kanban_notifications(_session())

        assert len(first) == 1
        wake = first[0]
        assert wake.task_id == tid
        assert wake.kind == "completed"
        assert wake.event_id > 0
        assert wake.created_at > 0
        assert wake.board_slug == kb.DEFAULT_BOARD
        assert wake.platform == "tui"
        assert wake.chat_id == SESSION_KEY
        assert wake.thread_id == ""
        assert tid in wake.text
        assert "done" in wake.text
        assert "shipped the fix" in wake.text
        rows = _sub_rows(tid)
        assert len(rows) == 1, "done must retain the originating session"
        first_cursor = rows[0]["last_event_id"]

        # The retained subscription must not replay the completed event.
        assert _collect_kanban_notifications(_session()) == []

        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,)
                )
                kb._append_event(conn, tid, "status", {"status": "ready"})
            assert kb.complete_task(conn, tid, summary="review corrections")
        finally:
            conn.close()

        reopened = _collect_kanban_notifications(_session())

        assert len(reopened) == 2
        assert "ready" in reopened[0].text
        assert "review corrections" in reopened[1].text
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == SESSION_KEY
        assert rows[0]["last_event_id"] > first_cursor
        assert _collect_kanban_notifications(_session()) == []

        conn = kb.connect()
        try:
            assert kb.archive_task(conn, tid)
        finally:
            conn.close()

        # Archive is notification-terminal and removes the retained route.
        assert _collect_kanban_notifications(_session()) == []
        assert _sub_rows(tid) == []

    def test_matching_tui_sub_delivers_and_advances_cursor(self):
        tid = _create_subscribed_task()
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        conn = kb.connect()
        try:
            kb.block_task(conn, tid, reason="waiting on review")
        finally:
            conn.close()

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            first = _collect_kanban_notifications(_session())
            second = _collect_kanban_notifications(_session())

        assert len(first) == 1
        assert "blocked" in first[0].text
        assert "waiting on review" in first[0].text
        assert second == []
        assert spy_connect.called
        # Blocked is not a final status -> subscription stays alive so a
        # respawned task's next terminal event still reaches the user.
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] > pre_cursor

    def test_non_tui_subscription_does_not_open_board_writable(self):
        tid = _create_subscribed_task(platform="telegram", chat_id="chat-1")
        # New subs start caught up at creation time (issue #29905); record the
        # pre-completion cursors so we can assert they were never claimed.
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        _complete(tid)

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] == pre_cursor

    def test_other_tui_session_does_not_open_board_writable(self):
        tid = _create_subscribed_task(chat_id="some-other-session")
        pre_cursor = _sub_rows(tid)[0]["last_event_id"]
        _complete(tid)

        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            texts = _collect_kanban_notifications(_session())

        assert texts == []
        spy_connect.assert_not_called()
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["last_event_id"] == pre_cursor

    def test_probe_error_falls_back_to_writable_delivery(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="fallback delivery")

        def fail_probe(*args, **kwargs):
            raise OSError("probe unavailable")

        monkeypatch.setattr(kb, "count_notify_subs", fail_probe)
        with patch.object(kb, "connect", wraps=kb.connect) as spy_connect:
            wakes = _collect_kanban_notifications(_session())

        assert len(wakes) == 1
        assert tid in wakes[0].text
        spy_connect.assert_called_once()

    def test_no_session_key_is_a_noop(self):
        tid = _create_subscribed_task()
        _complete(tid)

        assert _collect_kanban_notifications({"session_key": ""}) == []
        assert _collect_kanban_notifications({"session_key": None}) == []
        assert len(_sub_rows(tid)) == 1

    def test_profile_scoped_session_reads_the_shared_board(self, tmp_path):
        """The kanban board is shared across profiles BY DESIGN (see the
        hermes_cli/kanban_db.py module docstring): ``kanban_home()`` anchors on
        ``get_default_hermes_root()``, which resolves the process env and
        ignores context-local profile overrides. A Desktop session bound to a
        non-launch profile (``session["profile_home"]``) must therefore still
        have its subscription claimed from the one shared board — the poller
        needs no per-profile home binding.
        """
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        tid = _create_subscribed_task()
        _complete(tid, summary="cross-profile delivery")

        other_profile_home = tmp_path / "profiles" / "reviewer"
        other_profile_home.mkdir(parents=True)
        session = {
            "session_key": SESSION_KEY,
            "profile_home": str(other_profile_home),
        }
        # Simulate the strictest case: a context-local profile override is
        # active while the poller collects (as a profile-bound RPC would set).
        token = set_hermes_home_override(str(other_profile_home))
        try:
            wakes = _collect_kanban_notifications(session)
        finally:
            reset_hermes_home_override(token)

        assert len(wakes) == 1
        assert tid in wakes[0].text
        assert "cross-profile delivery" in wakes[0].text
        # Completion is reversible, so the shared-board subscription remains
        # owned by this exact Desktop session until the task is archived.
        rows = _sub_rows(tid)
        assert len(rows) == 1
        assert rows[0]["chat_id"] == SESSION_KEY


class TestFormatKanbanEventText:
    SUB = {"task_id": "t_abc123"}
    TASK = SimpleNamespace(title="build the thing", assignee="worker", result=None)

    def test_silent_kinds_return_none(self):
        for kind in ("archived", "unblocked"):
            ev = SimpleNamespace(kind=kind, payload={})
            assert _format_kanban_event_text(self.SUB, self.TASK, ev, "main") is None

    def test_blocked_includes_reason(self):
        ev = SimpleNamespace(kind="blocked", payload={"reason": "needs creds"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "main")
        assert "t_abc123" in text
        assert "blocked" in text
        assert "needs creds" in text
        assert "[main]" in text
        assert "@worker" in text

    def test_completed_prefers_payload_summary(self):
        ev = SimpleNamespace(kind="completed", payload={"summary": "first line\nsecond"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "done" in text
        assert "first line" in text
        assert "second" not in text

    def test_timed_out_with_bad_payload_does_not_raise(self):
        ev = SimpleNamespace(kind="timed_out", payload={"limit_seconds": "not-a-number"})
        text = _format_kanban_event_text(self.SUB, self.TASK, ev, "")
        assert "timed out" in text


class TestKanbanWakeLiveState:
    def test_current_completed_event_is_actionable(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="still current")
        wake = _collect_kanban_notifications(_session())[0]

        conn = kb.connect()
        try:
            assert kb.is_notification_event_current(
                conn,
                task_id=wake.task_id,
                event_id=wake.event_id,
                run_id=wake.run_id,
                kind=wake.kind,
            )
        finally:
            conn.close()

    def test_completed_event_remains_current_after_archive(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="final completion")
        wake = _collect_kanban_notifications(_session())[0]
        conn = kb.connect()
        try:
            assert kb.archive_task(conn, tid)
            assert kb.is_notification_event_current(
                conn,
                task_id=wake.task_id,
                event_id=wake.event_id,
                run_id=wake.run_id,
                kind=wake.kind,
            )
        finally:
            conn.close()

    def test_old_completed_event_is_stale_after_reopen_and_recomplete(self):
        tid = _create_subscribed_task()
        _complete(tid, summary="first completion")
        old_wake = _collect_kanban_notifications(_session())[0]

        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
                kb._append_event(conn, tid, "status", {"status": "ready"})
            assert kb.complete_task(conn, tid, summary="second completion")
            assert not kb.is_notification_event_current(
                conn,
                task_id=old_wake.task_id,
                event_id=old_wake.event_id,
                run_id=old_wake.run_id,
                kind=old_wake.kind,
            )
        finally:
            conn.close()

    def test_old_crash_event_is_stale_after_new_run_starts(self):
        tid = _create_subscribed_task()
        conn = kb.connect()
        try:
            claimed = kb.claim_task(conn, tid, claimer="worker-one")
            assert claimed is not None
            with kb.write_txn(conn):
                old_run_id = kb._end_run(conn, tid, outcome="crashed", status="crashed")
                conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
                    (tid,),
                )
                kb._append_event(conn, tid, "crashed", {}, run_id=old_run_id)
        finally:
            conn.close()
        old_wake = _collect_kanban_notifications(_session())[0]

        conn = kb.connect()
        try:
            assert kb.claim_task(conn, tid, claimer="worker-two") is not None
            assert not kb.is_notification_event_current(
                conn,
                task_id=old_wake.task_id,
                event_id=old_wake.event_id,
                run_id=old_wake.run_id,
                kind=old_wake.kind,
            )
        finally:
            conn.close()


class TestNotificationPollerLoopKanbanWiring:
    """Drive a real TUI subscription through ``_notification_poller_loop``.

    Covers the wiring above ``_collect_kanban_notifications``: status.update
    emission, agent-turn dispatch when the session is idle, and the
    busy-session pending buffer that flushes once the session goes idle.
    """

    def _start_poller(self, session: dict, monkeypatch):
        import threading
        import tui_gateway.server as server

        emits: list = []
        submits: list[dict] = []
        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(
            server, "_emit", lambda event, sid, payload=None: emits.append((event, payload))
        )
        def capture_submit(rid, sid, sess, text, **kwargs):
            submits.append({"text": text, **kwargs})

        monkeypatch.setattr(server, "_run_prompt_submit", capture_submit)
        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        return stop, thread, emits, submits

    @staticmethod
    def _wait_for(predicate, timeout: float = 5.0) -> bool:
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            if predicate():
                return True
            _time.sleep(0.02)
        return False

    def _poller_session(self, *, running: bool = False) -> dict:
        import threading

        return {
            "session_key": SESSION_KEY,
            "history_lock": threading.Lock(),
            "running": running,
        }

    def test_idle_session_gets_status_update_and_agent_turn(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="poller e2e done")
        session = self._poller_session(running=False)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "agent turn was never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        status_texts = [p["text"] for e, p in emits if e == "status.update" and p]
        assert all(isinstance(text, str) for text in status_texts)
        assert any(tid in t for t in status_texts), status_texts
        assert any(e == "message.start" for e, _ in emits)
        assert len(submits) == 1
        submitted = submits[0]
        assert "[INTERNAL KANBAN WAKE — NOT USER-AUTHORED]" in submitted["text"]
        assert tid in submitted["text"]
        assert submitted["display_kind"] == "internal_notification"
        metadata = submitted["display_metadata"]
        assert metadata["source"] == "kanban"
        assert metadata["internal"] is True
        assert metadata["kind"] == "kanban_wake"
        assert metadata["message_id"].startswith("kanban-wake:")
        assert metadata["display_text"] == status_texts[0]
        assert len(metadata["events"]) == 1
        event = metadata["events"][0]
        assert event["board"] == kb.DEFAULT_BOARD
        assert event["task_id"] == tid
        assert event["event_id"] > 0
        assert event["event_kind"] == "completed"
        assert event["occurred_at"] > 0
        assert event["platform"] == "tui"
        assert event["chat_id"] == SESSION_KEY
        assert event["thread_id"] == ""
        assert session["running"] is True  # poller claimed the turn
        assert not session.get("_kanban_pending")

    def test_busy_session_buffers_then_flushes_when_idle(self, monkeypatch):
        tid = _create_subscribed_task()
        _complete(tid, summary="buffered while busy")
        session = self._poller_session(running=True)

        stop, thread, emits, submits = self._start_poller(session, monkeypatch)
        try:
            # Busy: the status line appears and the event is buffered, but no
            # agent turn is dispatched while another turn is running.
            assert self._wait_for(
                lambda: any(e == "status.update" for e, _ in emits)
                and session.get("_kanban_pending")
            )
            assert not submits
            pending = session["_kanban_pending"]
            assert len(pending) == 1
            assert pending[0].task_id == tid
            assert pending[0].kind == "completed"

            with session["history_lock"]:
                session["running"] = False

            assert self._wait_for(lambda: submits), "pending batch never flushed"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert len(submits) == 1
        assert tid in submits[0]["text"]
        assert submits[0]["display_kind"] == "internal_notification"
        assert submits[0]["display_metadata"]["source"] == "kanban"
        assert session["_kanban_pending"] == []
        assert session["running"] is True

    def test_stale_pending_wake_is_suppressed_at_idle_boundary(self, monkeypatch):
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="became stale")
        stale_wake = _collect_kanban_notifications(_session())[0]
        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
                kb._append_event(conn, tid, "status", {"status": "ready"})
        finally:
            conn.close()

        session = self._poller_session(running=False)
        session["_kanban_pending"] = [stale_wake]
        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
        stop, thread, _emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(
                lambda: session.get("_kanban_pending") == []
                and session.get("running") is False
            ), "stale pending wake was not consumed"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert submits == []

    def test_live_state_read_failure_keeps_wake_pending(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="retry validation")
        wake = _collect_kanban_notifications(_session())[0]
        session = self._poller_session(running=False)
        session["_kanban_pending"] = [wake]
        checked = threading.Event()

        def fail_validation(*_args, **_kwargs):
            checked.set()
            raise OSError("board temporarily unavailable")

        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
        monkeypatch.setattr(kb, "is_notification_event_current", fail_validation)
        stop, thread, _emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert checked.wait(timeout=5), "live-state validation was never attempted"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert submits == []
        assert session.get("running") is False
        assert session.get("_kanban_pending") == [wake]

    def test_validation_retry_stays_before_concurrent_new_wake(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="older wake")
        older = _collect_kanban_notifications(_session())[0]
        newer = older._replace(event_id=older.event_id + 100, text="newer wake")
        session = self._poller_session(running=False)
        session["_kanban_pending"] = [older]
        injected = False

        def fail_after_concurrent_enqueue(*_args, **_kwargs):
            nonlocal injected
            if not injected:
                session["_kanban_pending"] = [newer]
                injected = True
            raise OSError("board temporarily unavailable")

        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
        monkeypatch.setattr(kb, "is_notification_event_current", fail_after_concurrent_enqueue)
        stop, thread, _emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(
                lambda: len(session.get("_kanban_pending") or []) == 2
                and session.get("running") is False
            )
        finally:
            stop.set()
            thread.join(timeout=5)

        assert submits == []
        assert session.get("_kanban_pending") == [older, newer]

    def test_submit_failure_requeues_current_wake(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="retry submit")
        wake = _collect_kanban_notifications(_session())[0]
        session = self._poller_session(running=False)
        session["_kanban_pending"] = [wake]
        attempted = threading.Event()

        def fail_submit(*_args, **_kwargs):
            attempted.set()
            raise RuntimeError("submit unavailable")

        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
        monkeypatch.setattr(server, "_run_prompt_submit", fail_submit)
        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        try:
            assert attempted.wait(timeout=5), "submit was never attempted"
            assert self._wait_for(lambda: session.get("running") is False)
        finally:
            stop.set()
            thread.join(timeout=5)

        assert session.get("_kanban_pending") == [wake]

    def test_submit_failure_preserves_mixed_batch_order(self, monkeypatch):
        import threading
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="mixed batch")
        older_retry = _collect_kanban_notifications(_session())[0]
        newer_current = older_retry._replace(
            event_id=older_retry.event_id + 1,
            text="newer current wake",
        )
        session = self._poller_session(running=False)
        session["_kanban_pending"] = [older_retry, newer_current]
        attempted = threading.Event()

        def fail_submit(*_args, **_kwargs):
            attempted.set()
            raise RuntimeError("submit unavailable")

        monkeypatch.setattr(server, "_KANBAN_POLL_SECONDS", 0.01)
        monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
        monkeypatch.setattr(
            server,
            "_partition_current_kanban_wakes",
            lambda _batch: ([newer_current], [older_retry]),
        )
        monkeypatch.setattr(server, "_run_prompt_submit", fail_submit)
        stop = threading.Event()
        thread = threading.Thread(
            target=server._notification_poller_loop,
            args=(stop, "sid-poller-test", session),
            daemon=True,
        )
        thread.start()
        try:
            assert attempted.wait(timeout=5), "submit was never attempted"
            assert self._wait_for(lambda: session.get("running") is False)
        finally:
            stop.set()
            thread.join(timeout=5)

        assert session.get("_kanban_pending") == [older_retry, newer_current]

    def test_duplicate_pending_event_identity_dispatches_once(self, monkeypatch):
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="one durable event")
        wake = _collect_kanban_notifications(_session())[0]
        session = self._poller_session(running=False)
        session["_kanban_pending"] = [wake, wake]
        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])

        stop, thread, _emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "pending wake was never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert len(submits) == 1
        assert len(submits[0]["display_metadata"]["events"]) == 1
        assert submits[0]["display_metadata"]["events"][0]["event_id"] == wake.event_id

    def test_distinct_subscription_identity_is_not_collapsed(self, monkeypatch):
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        _complete(tid, summary="same event, two routes")
        wake = _collect_kanban_notifications(_session())[0]
        routed_copy = wake._replace(thread_id="secondary-thread")
        session = self._poller_session(running=False)
        session["_kanban_pending"] = [wake, routed_copy]
        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])

        stop, thread, _emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "pending wakes were never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert len(submits) == 1
        assert len(submits[0]["display_metadata"]["events"]) == 2

    def test_same_text_with_distinct_event_ids_is_not_collapsed(self, monkeypatch):
        import tui_gateway.server as server

        tid = _create_subscribed_task()
        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "crashed", {})
                kb._append_event(conn, tid, "crashed", {})
        finally:
            conn.close()
        wakes = _collect_kanban_notifications(_session())
        assert len(wakes) == 2
        assert wakes[0].text == wakes[1].text
        assert wakes[0].event_id != wakes[1].event_id

        session = self._poller_session(running=False)
        session["_kanban_pending"] = wakes
        monkeypatch.setattr(server, "_collect_kanban_notifications", lambda _session: [])
        stop, thread, _emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "pending wakes were never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert len(submits[0]["display_metadata"]["events"]) == 2

    @pytest.mark.parametrize(
        ("event_kind", "payload"),
        [
            ("completed", {"summary": "done"}),
            ("blocked", {"reason": "needs input"}),
            ("gave_up", {"error": "spawn failed"}),
            ("crashed", {}),
            ("timed_out", {"limit_seconds": 30}),
            ("status", {"status": "ready"}),
        ],
    )
    def test_existing_wake_kinds_still_dispatch_typed_agent_turn(
        self, monkeypatch, event_kind, payload
    ):
        tid = _create_subscribed_task()
        conn = kb.connect()
        try:
            with kb.write_txn(conn):
                current_status = {
                    "completed": "done",
                    "blocked": "blocked",
                    "gave_up": "blocked",
                }.get(event_kind)
                if current_status:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ?",
                        (current_status, tid),
                    )
                kb._append_event(conn, tid, event_kind, payload, run_id=17)
        finally:
            conn.close()
        session = self._poller_session(running=False)

        stop, thread, _emits, submits = self._start_poller(session, monkeypatch)
        try:
            assert self._wait_for(lambda: submits), "agent turn was never dispatched"
        finally:
            stop.set()
            thread.join(timeout=5)

        assert len(submits) == 1
        assert submits[0]["display_kind"] == "internal_notification"
        metadata = submits[0]["display_metadata"]
        assert metadata["source"] == "kanban"
        assert len(metadata["events"]) == 1
        event = metadata["events"][0]
        assert event["task_id"] == tid
        assert event["event_kind"] == event_kind
        assert event["run_id"] == 17
