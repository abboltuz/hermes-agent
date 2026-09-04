"""Durable working-context lifecycle for SessionDB.

Only hashes and execution receipts belong here, never provider credentials or
transcript copies. The existing compression lease remains the publication fence;
this journal prevents the same failed work from restarting with a new process.
"""

import math
import time


_TERMINAL_OUTCOMES = frozenset({"committed", "no_progress", "timed_out", "aborted"})
_RETRYABLE_OUTCOMES = frozenset({"cooldown", "deferred_lock", "native_delegated"})


class SessionContextMixin:
    """Uses the host's transaction and profile-scoped connection infrastructure."""

    def claim_context_compaction(
        self, session_id: str, source_fingerprint: str, strategy_fingerprint: str,
        *, owner: str, force: bool = False, lease_seconds: float = 3600,
        now: float | None = None,
    ) -> str:
        """Admit one source/strategy attempt, independently of process generations.

        Expiration makes abandoned work terminal, not automatically runnable.
        A changed source/strategy or an explicit force can rearm it. Force never
        steals a live attempt. ``now`` is an injectable wall clock, not a timeout
        for the summarizer (which retains its progress-aware timeout policy).
        """
        if any(not isinstance(value, str) or not value or len(value) > 256 for value in (
            session_id, source_fingerprint, strategy_fingerprint, owner,
        )):
            raise ValueError("context compaction requires bounded nonempty identities")
        now = time.time() if now is None else float(now)
        if not math.isfinite(now) or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("context compaction requires a finite positive lease")

        def claim(conn):
            # The first in-memory turn may not have a durable session yet.
            if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
                return "unpersisted"
            conn.execute(
                "UPDATE context_compaction_jobs SET outcome = 'timed_out', updated_at = ? "
                "WHERE session_id = ? AND outcome = 'running' AND expires_at <= ?",
                (now, session_id, now),
            )
            active = conn.execute(
                "SELECT source_fingerprint, strategy_fingerprint FROM context_compaction_jobs "
                "WHERE session_id = ? AND outcome = 'running'",
                (session_id,),
            ).fetchone()
            if active is not None:
                same = (active[0], active[1]) == (source_fingerprint, strategy_fingerprint)
                return "joined" if same else "deferred_lock"
            previous = conn.execute(
                "SELECT outcome FROM context_compaction_jobs WHERE session_id = ? "
                "AND source_fingerprint = ? AND strategy_fingerprint = ?",
                (session_id, source_fingerprint, strategy_fingerprint),
            ).fetchone()
            if previous is not None and previous[0] in _TERMINAL_OUTCOMES and not force:
                return "no_progress_suppressed"
            conn.execute(
                "INSERT INTO context_compaction_jobs "
                "(session_id, source_fingerprint, strategy_fingerprint, owner, outcome, "
                "attempt_count, expires_at, updated_at) VALUES (?, ?, ?, ?, 'running', 1, ?, ?) "
                "ON CONFLICT(session_id, source_fingerprint, strategy_fingerprint) DO UPDATE SET "
                "owner = excluded.owner, outcome = 'running', "
                "attempt_count = context_compaction_jobs.attempt_count + 1, "
                "expires_at = excluded.expires_at, updated_at = excluded.updated_at",
                (session_id, source_fingerprint, strategy_fingerprint, owner, now + lease_seconds, now),
            )
            return "admitted"

        # Admission must not queue behind long transcript/FTS maintenance.
        return self._execute_write(claim, patience_s=self._ACTIVITY_WRITE_PATIENCE_S)

    def finish_context_compaction(self, session_id: str, *, owner: str, outcome: str) -> bool:
        """Record only the current owner's result; stale workers cannot overwrite it."""
        if outcome not in _TERMINAL_OUTCOMES | _RETRYABLE_OUTCOMES:
            raise ValueError("unknown context compaction outcome")

        def finish(conn):
            return conn.execute(
                "UPDATE context_compaction_jobs SET outcome = ?, updated_at = ? "
                "WHERE session_id = ? AND owner = ? AND outcome = 'running'",
                (outcome, time.time(), session_id, owner),
            ).rowcount == 1

        return self._execute_write(finish, patience_s=self._ACTIVITY_WRITE_PATIENCE_S)
