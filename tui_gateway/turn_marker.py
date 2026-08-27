"""Durable interrupted-turn markers for the desktop/TUI auto-continue path.

A running turn's progress lives only in process memory (the agent flushes to
SQLite at turn end, not mid-turn), so an app/backend/machine death mid-turn
leaves no durable trace of the interrupted prompt. This sidecar is that
trace: a marker is written when a turn starts running and cleared when the
turn concludes — success, handled error, or interrupt all clear it, so only
a process death leaves one behind. ``session.resume`` reads the marker to
decide whether to auto-continue the interrupted turn (see
``_maybe_schedule_auto_continue`` in ``tui_gateway/server.py``).

Markers are stored per ``HERMES_HOME`` (callers pass the session's home so
profile sessions keep their state in their own profile directory) and the
file is bounded: writes prune entries older than ``_MAX_AGE_SECS`` and cap
the total count, so an unlucky streak of crashes can't grow it unboundedly.

Every function is best-effort by design — marker bookkeeping must never
break a turn — so I/O errors degrade to "no marker" instead of raising.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MARKER_DIR = "desktop"
_MARKER_FILE = "interrupted_turns.json"
_MAX_AGE_SECS = 24 * 3600
_MAX_ENTRIES = 32
# Enough to re-submit any realistic prompt; guards the sidecar against a
# pathological multi-megabyte paste being journaled on every turn.
_MAX_PROMPT_CHARS = 64_000
_MAX_DISPLAY_KIND_CHARS = 128
_MAX_METADATA_TEXT_CHARS = 16_000
_MAX_METADATA_ID_CHARS = 512
_MAX_METADATA_EVENTS = 64

_lock = threading.Lock()


def _bounded_string(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:limit] if value else None


def _sanitize_display_metadata(value: Any) -> dict[str, Any] | None:
    """Return the bounded provenance subset safe to journal across restarts."""
    if not isinstance(value, dict):
        return None

    out: dict[str, Any] = {}
    for key in ("source", "kind", "platform", "delegation_id", "event_id", "session_id"):
        item = value.get(key)
        if key == "event_id" and isinstance(item, int) and not isinstance(item, bool):
            out[key] = item
            continue
        bounded = _bounded_string(item, _MAX_METADATA_ID_CHARS)
        if bounded is not None:
            out[key] = bounded

    display_text = _bounded_string(value.get("display_text"), _MAX_METADATA_TEXT_CHARS)
    if display_text is not None:
        out["display_text"] = display_text

    for key in ("internal", "recovered"):
        if isinstance(value.get(key), bool):
            out[key] = value[key]

    for key in ("task_count", "completed_count", "failed_count"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            out[key] = max(0, item)

    duration = value.get("duration_seconds")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        out["duration_seconds"] = max(0.0, float(duration))

    raw_events = value.get("events")
    if isinstance(raw_events, list):
        events: list[dict[str, Any]] = []
        for raw_event in raw_events[:_MAX_METADATA_EVENTS]:
            if not isinstance(raw_event, dict):
                continue
            event: dict[str, Any] = {}
            for key in ("board", "task_id", "event_kind"):
                bounded = _bounded_string(raw_event.get(key), _MAX_METADATA_ID_CHARS)
                if bounded is not None:
                    event[key] = bounded
            for key in ("event_id", "run_id"):
                item = raw_event.get(key)
                if isinstance(item, int) and not isinstance(item, bool):
                    event[key] = item
            occurred_at = raw_event.get("occurred_at")
            if isinstance(occurred_at, (int, float)) and not isinstance(occurred_at, bool):
                event["occurred_at"] = occurred_at
            if event:
                events.append(event)
        if events:
            out["events"] = events

    return out or None


def _marker_path(home: Path | str) -> Path:
    return Path(home) / _MARKER_DIR / _MARKER_FILE


def _load(path: Path) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.debug("unreadable turn-marker file %s; starting fresh", path, exc_info=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _prune(entries: dict[str, dict], now: float) -> dict[str, dict]:
    fresh = {
        key: entry
        for key, entry in entries.items()
        if now - float(entry.get("started_at") or 0) <= _MAX_AGE_SECS
    }
    if len(fresh) <= _MAX_ENTRIES:
        return fresh
    newest = sorted(
        fresh.items(),
        key=lambda item: float(item[1].get("started_at") or 0),
        reverse=True,
    )[:_MAX_ENTRIES]
    return dict(newest)


def _store(path: Path, entries: dict[str, dict]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".turn-marker-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_turn_start(
    home: Path | str,
    session_key: str,
    prompt: str,
    *,
    attempts: int = 0,
    display_kind: str | None = None,
    display_metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the marker for a turn that is about to run.

    ``attempts`` counts how many auto-continues led to this run: 0 for a
    user-initiated turn, N for the Nth automatic re-run — the crash-loop
    breaker reads it back on the next resume.
    """
    if not session_key or not prompt:
        return
    now = time.time()
    entry = {
        "attempts": max(0, int(attempts)),
        "prompt": prompt[:_MAX_PROMPT_CHARS],
        "started_at": now,
    }
    bounded_kind = _bounded_string(display_kind, _MAX_DISPLAY_KIND_CHARS)
    if bounded_kind is not None:
        entry["display_kind"] = bounded_kind
    bounded_metadata = _sanitize_display_metadata(display_metadata)
    if bounded_metadata is not None:
        entry["display_metadata"] = bounded_metadata
    try:
        with _lock:
            path = _marker_path(home)
            entries = _prune(_load(path), now)
            entries[session_key] = entry
            _store(path, entries)
    except Exception:
        logger.debug("failed to record turn marker for %s", session_key, exc_info=True)


def clear_turn_marker(home: Path | str, session_key: str) -> None:
    """Remove the marker once its turn concluded (any outcome the client saw)."""
    if not session_key:
        return
    try:
        with _lock:
            path = _marker_path(home)
            entries = _load(path)
            if session_key not in entries:
                return
            del entries[session_key]
            _store(path, entries)
    except Exception:
        logger.debug("failed to clear turn marker for %s", session_key, exc_info=True)


def read_turn_marker(home: Path | str, session_key: str) -> dict[str, Any] | None:
    """The marker left by a turn that never concluded, or None."""
    if not session_key:
        return None
    try:
        with _lock:
            entry = _load(_marker_path(home)).get(session_key)
    except Exception:
        return None
    if not isinstance(entry, dict):
        return None
    prompt = str(entry.get("prompt") or "")
    if not prompt.strip():
        return None
    try:
        started_at = float(entry.get("started_at") or 0)
        attempts = max(0, int(entry.get("attempts") or 0))
    except (TypeError, ValueError):
        return None
    marker: dict[str, Any] = {
        "attempts": attempts,
        "prompt": prompt,
        "started_at": started_at,
    }
    display_kind = _bounded_string(entry.get("display_kind"), _MAX_DISPLAY_KIND_CHARS)
    if display_kind is not None:
        marker["display_kind"] = display_kind
    display_metadata = _sanitize_display_metadata(entry.get("display_metadata"))
    if display_metadata is not None:
        marker["display_metadata"] = display_metadata
    return marker
