"""Versioned, bounded working-context manifests, separate from raw history.

Manifests contain row references, not another copy of transcript bodies. The
legacy compaction writer still owns the materialized message projection until
the immutable archive migration replaces it. All reads pin one SQLite snapshot;
concurrent appends are a fenced tail, never silently truncated.
"""

from contextlib import contextmanager
import json
import sqlite3
import time


MAX_WORKING_CONTEXT_ROWS = 8192
MAX_WORKING_CONTEXT_BYTES = 16 * 1024 * 1024


class WorkingContextTooLargeError(RuntimeError):
    """The projection needs compaction/artifact extraction, not data deletion."""


class WorkingContextSnapshotError(RuntimeError):
    """A stored manifest is inconsistent; do not silently replay partial data."""


@contextmanager
def _read_transaction(conn):
    owned = not conn.in_transaction
    if owned:
        conn.execute("BEGIN")
    try:
        yield
    finally:
        if owned:
            conn.rollback()


class SessionSnapshotMixin:
    """Internal publication and indexed read paths for SessionDB."""

    def has_working_context_snapshot(self, session_id):
        """Cheap archive hint for chat-open pagination.

        The head table has one primary-key row per compacted session. Unlike a
        probe over ``messages.compacted``, this lookup never scales with archive
        size. Pre-migration read-only stores simply have no hint.
        """
        with self._read_ctx() as conn:
            try:
                row = conn.execute(
                    "SELECT 1 FROM working_context_heads WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                if "no such table: working_context_heads" in str(exc):
                    return False
                raise
        return row is not None

    def _working_context_rows_by_id(self, conn, session_id, ids, *, materialize=True):
        if len(ids) > MAX_WORKING_CONTEXT_ROWS:
            raise WorkingContextTooLargeError(
                "Working context exceeds the row limit; compact it before resuming. "
                "The complete conversation remains stored."
            )
        # json_each avoids SQLite's variable-count limit. PK probes avoid
        # scanning the session's accumulated inactive compaction generations.
        encoded = json.dumps(ids, separators=(",", ":"))
        columns = [name.strip() for name in self._CONVERSATION_ROW_COLUMNS.split(",")]
        size_sql = " + ".join(
            f"COALESCE(length(CAST(m.{name} AS BLOB)), 0)" for name in columns
        )
        join = (
            # CROSS JOIN pins the tiny manifest as the outer loop. NOT INDEXED
            # still permits INTEGER PRIMARY KEY lookup, but prevents choosing
            # the session/active index and scanning history before filtering IDs.
            "FROM json_each(?) AS refs CROSS JOIN messages AS m NOT INDEXED "
            "WHERE m.id = refs.value AND m.session_id = ? AND m.active = 1"
        )
        count, byte_count = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM({size_sql}), 0) {join}",
            (encoded, session_id),
        ).fetchone()
        if count != len(ids):
            raise WorkingContextSnapshotError(
                "Working context references unavailable rows"
            )
        if byte_count > MAX_WORKING_CONTEXT_BYTES:
            raise WorkingContextTooLargeError(
                "Working context exceeds the byte limit; compact it or extract large "
                "artifacts before resuming. The complete conversation remains stored."
            )
        if not materialize:
            return [], byte_count
        rows = conn.execute(
            f"SELECT {', '.join('m.' + name for name in columns)} {join} "
            "ORDER BY CAST(refs.key AS INTEGER)",
            (encoded, session_id),
        ).fetchall()
        return rows, byte_count

    def _publish_working_context_snapshot(self, conn, session_id):
        """Publish in the SAME transaction as the fenced compaction commit."""
        ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND active = 1 "
                "ORDER BY id LIMIT ?",
                (session_id, MAX_WORKING_CONTEXT_ROWS + 1),
            )
        ]
        _, byte_count = self._working_context_rows_by_id(
            conn,
            session_id,
            ids,
            materialize=False,
        )
        generation = conn.execute(
            "SELECT COALESCE(MAX(generation), 0) + 1 FROM working_context_snapshots "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        # An empty projection still fences older inactive rows. Appended ids
        # are monotonic even when the compaction candidate contains no rows.
        watermark = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        cursor = conn.execute(
            "INSERT INTO working_context_snapshots "
            "(session_id, generation, watermark, message_ids, row_count, payload_bytes, created_at, format_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 2)",
            (
                session_id,
                generation,
                watermark,
                json.dumps(ids, separators=(",", ":")),
                len(ids),
                byte_count,
                time.time(),
            ),
        )
        # Every current tail row is covered by the newly published manifest.
        # Also handles publication after a previously empty manifest.
        conn.execute(
            "DELETE FROM working_context_tail WHERE session_id = ?", (session_id,)
        )
        conn.execute(
            "INSERT INTO working_context_heads (session_id, snapshot_id) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET snapshot_id = excluded.snapshot_id",
            (session_id, cursor.lastrowid),
        )

    def _read_working_context_rows(self, session_id):
        """Return a bounded projection plus append tail, or None for legacy data."""
        with self._read_ctx() as conn, _read_transaction(conn):
            try:
                head = conn.execute(
                    "SELECT s.* FROM working_context_heads h "
                    "JOIN working_context_snapshots s ON s.id = h.snapshot_id "
                    "AND s.session_id = h.session_id WHERE h.session_id = ?",
                    (session_id,),
                ).fetchone()
            except sqlite3.OperationalError as exc:
                # Read-only access to a pre-migration database remains valid.
                # Do not swallow lock, I/O or corruption errors as cache misses.
                if "no such table: working_context_" in str(exc):
                    return None
                raise
            if head is None:
                return None
            # A manifest created before the transactional tail registry cannot
            # prove tail completeness. Read it through the legacy path until a
            # fresh compaction publishes v2; never backfill by scanning at open.
            if "format_version" not in head.keys() or head["format_version"] != 2:
                return None
            encoded = head["message_ids"]
            if (
                not isinstance(encoded, str)
                or len(encoded) > MAX_WORKING_CONTEXT_ROWS * 24 + 2
            ):
                raise WorkingContextSnapshotError(
                    "Working context manifest exceeds its bound"
                )
            try:
                ids = json.loads(encoded)
            except (ValueError, TypeError) as exc:
                raise WorkingContextSnapshotError(
                    "Invalid working context manifest"
                ) from exc
            if (
                not isinstance(ids, list)
                or len(ids) != head["row_count"]
                or any(
                    type(value) is not int or value <= 0 or value > head["watermark"]
                    for value in ids
                )
                or ids != sorted(set(ids))
            ):
                raise WorkingContextSnapshotError(
                    "Invalid working context row identities"
                )
            tail = [
                int(row[0])
                for row in conn.execute(
                    # This small membership index excludes inactive imports,
                    # including those whose IDs exceed the snapshot watermark.
                    "SELECT message_id FROM working_context_tail "
                    "WHERE session_id = ? AND message_id > ? "
                    "ORDER BY message_id LIMIT ?",
                    (session_id, head["watermark"], MAX_WORKING_CONTEXT_ROWS + 1),
                )
            ]
            rows, _ = self._working_context_rows_by_id(conn, session_id, ids + tail)
            return rows
