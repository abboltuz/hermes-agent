"""Bounded recovery for legacy sessions whose active tip is too large to resume.

The normal compressor operates on an already-loaded conversation.  This module
is the cold-start counterpart: it pages durable rows up to a watermark, reduces
the old prefix incrementally, retains a small exact tail, and publishes through
``SessionDB.archive_and_compact``.  Raw rows are soft-archived, never deleted.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import json
import os
import secrets
import threading
from typing import Any, Callable, Mapping, Sequence

from agent.compression_v3 import estimate_provider_wire_tokens, validate_projection
from agent.context_compressor import (
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
    _SUMMARY_END_MARKER,
)
from agent.message_provenance import is_human_intent


SummaryCallback = Callable[[Sequence[Mapping[str, Any]], str | None], str | None]
ProgressCallback = Callable[[str, int], None]


class ResumeHardCompactionBusy(RuntimeError):
    """Another process owns compaction for the same logical source."""


class ResumeHardCompactionUnsafe(RuntimeError):
    """No replay-safe bounded projection could be produced."""


@dataclass(frozen=True)
class ResumeHardCompactionPolicy:
    target_rows: int = 512
    page_rows: int = 256
    summary_chunk_rows: int = 256
    max_summary_chars: int = 32_000
    max_tail_message_chars: int = 16_000
    max_projection_tokens: int = 48_000
    max_page_materialized_chars: int = 4_000_000
    strategy_identity: str = ""
    lease_seconds: float = 300.0

    def validate(self) -> None:
        if self.target_rows < 4:
            raise ValueError("target_rows must be at least 4")
        if self.page_rows < 1 or self.summary_chunk_rows < 1:
            raise ValueError("page and summary chunk sizes must be positive")
        if self.max_summary_chars < 1_000:
            raise ValueError("max_summary_chars must be at least 1000")
        if self.max_tail_message_chars < 1_000:
            raise ValueError("max_tail_message_chars must be at least 1000")
        if self.max_projection_tokens < 1_000:
            raise ValueError("max_projection_tokens must be at least 1000")
        if self.max_page_materialized_chars < self.max_tail_message_chars:
            raise ValueError(
                "max_page_materialized_chars must cover at least one bounded row"
            )
        if self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")


@dataclass(frozen=True)
class ResumeHardCompactionResult:
    outcome: str
    source_rows: int
    active_rows: int
    watermark: int
    used_generated_summary: bool = False


class _LeaseRefresher:
    def __init__(self, db: Any, session_id: str, holder: str, ttl: float) -> None:
        self._db = db
        self._session_id = session_id
        self._holder = holder
        self._ttl = ttl
        self._interval = max(0.1, min(60.0, self._ttl / 3.0))
        self._max_consecutive_failures = max(
            1, int(self._ttl / self._interval)
        )
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="resume-hard-compaction-lease",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    @property
    def lost(self) -> bool:
        return self._lost.is_set()

    def _run(self) -> None:
        consecutive_failures = 0
        # Refresh immediately: enumeration can be substantial and must not
        # consume the first lease interval before ownership is re-confirmed.
        first = True
        while first or not self._stop.wait(self._interval):
            if first:
                first = False
                if self._stop.is_set():
                    return
            try:
                if self._db.refresh_compression_lock(
                    self._session_id, self._holder, ttl_seconds=self._ttl
                ):
                    consecutive_failures = 0
                    continue
            except Exception:
                pass
            # SessionDB deliberately reports transient SQLite contention and
            # genuine ownership loss through the same falsy result. Match the
            # normal compressor: tolerate a blip, but never beyond one TTL.
            consecutive_failures += 1
            if consecutive_failures >= self._max_consecutive_failures:
                self._lost.set()
                return


def _report_progress(
    callback: ProgressCallback | None, phase: str, count: int
) -> None:
    if callback is None:
        return
    try:
        callback(phase, count)
    except Exception:
        # UI/observability callbacks are advisory. They must never abort a
        # safe compaction or turn a committed publication into a retry loop.
        pass


def _clean_message(message: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(dict(message))
    for key in (
        "id",
        "session_id",
        "active",
        "compacted",
        "created_at",
        "_row_id",
        "_db_persisted",
        "_resume_archive_rows",
        "_resume_payload_clipped",
        "_resume_source_payload_chars",
    ):
        cleaned.pop(key, None)
    return cleaned


def _bounded_tail_message(
    message: Mapping[str, Any], max_chars: int
) -> dict[str, Any]:
    """Bound one replay row while pointing back to its exact archived source."""
    tracked_rows = message.get("_resume_archive_rows")
    source_rows = [
        int(row_id)
        for row_id in (
            tracked_rows
            if isinstance(tracked_rows, list)
            else [message.get("id") or message.get("_row_id") or 0]
        )
        if int(row_id or 0) > 0
    ]
    source_row = source_rows[0] if source_rows else 0
    source_was_clipped = bool(message.get("_resume_payload_clipped"))
    cleaned = _clean_message(message)
    if len(source_rows) > 1:
        row_label = ", ".join(str(row_id) for row_id in source_rows)
        reference = f"[Full durable payloads archived at message rows {row_label}.]"
    else:
        reference = f"[Full durable payload archived at message row {source_row}.]"

    def bounded_text(text: str) -> str:
        marker = f"\n... {reference}\n"
        available = max(0, max_chars - len(marker))
        head = available * 3 // 4
        tail = available - head
        return text[:head].rstrip() + marker + text[-tail:].lstrip()

    def append_reference(text: str) -> str:
        if reference in text:
            return text
        marker = f"\n{reference}"
        return text[: max(0, max_chars - len(marker))].rstrip() + marker

    content = cleaned.get("content")
    if isinstance(content, str) and len(content) > max_chars:
        cleaned["content"] = bounded_text(content)
        cleaned.pop("api_content", None)
    elif not isinstance(content, str):
        try:
            encoded_content = json.dumps(content, ensure_ascii=False, default=str)
        except Exception:
            encoded_content = str(content)
        if len(encoded_content) > max_chars:
            cleaned["content"] = reference
            cleaned.pop("api_content", None)

    api_content = cleaned.get("api_content")
    if isinstance(api_content, str) and len(api_content) > max_chars:
        # Hard compaction is already a prompt-cache episode boundary. Keeping
        # an unbounded historical wire sidecar would defeat recovery, so retain
        # the human-readable view plus a durable pointer to the exact source.
        cleaned.pop("api_content", None)
        current = cleaned.get("content")
        cleaned["content"] = (
            append_reference(current) if isinstance(current, str) else reference
        )

    if source_was_clipped:
        current = cleaned.get("content")
        cleaned["content"] = (
            append_reference(current) if isinstance(current, str) else reference
        )
        # Keep the signal until same-role repair has propagated all source row
        # identities. _clean_message always removes it from provider payloads.
        cleaned["_resume_payload_clipped"] = True

    tool_calls = cleaned.get("tool_calls")
    if tool_calls:
        try:
            encoded_calls = json.dumps(tool_calls, ensure_ascii=False, default=str)
        except Exception:
            encoded_calls = str(tool_calls)
        if len(encoded_calls) > max_chars:
            bounded_calls = []
            for call in tool_calls if isinstance(tool_calls, list) else []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                bounded = {
                    "id": str(call.get("id") or "")[:512],
                    "type": str(call.get("type") or "function")[:64],
                    "function": {
                        "name": str(
                            function.get("name") if isinstance(function, dict) else ""
                        )[:512],
                        "arguments": json.dumps(
                            {"archive_message_rows": source_rows}, separators=(",", ":")
                        ),
                    },
                }
                prospective = [*bounded_calls, bounded]
                if len(json.dumps(prospective, ensure_ascii=False)) > max_chars:
                    break
                bounded_calls = prospective
            cleaned["tool_calls"] = bounded_calls
            current = cleaned.get("content")
            cleaned["content"] = (
                append_reference(current) if isinstance(current, str) else reference
            )

    for field in (
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
    ):
        value = cleaned.get(field)
        if value is None:
            continue
        try:
            size = len(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            size = len(str(value))
        if size > max_chars:
            cleaned.pop(field, None)
    return cleaned


def _ensure_summary_envelope(summary: str, max_chars: int = 32_000) -> str:
    body = str(summary or "").strip()
    if not body.startswith(SUMMARY_PREFIX):
        body = f"{SUMMARY_PREFIX}\n{body}"
    if body.rstrip().endswith(_SUMMARY_END_MARKER):
        body = body.rstrip()[: -len(_SUMMARY_END_MARKER)].rstrip()
    reserve = len(_SUMMARY_END_MARKER) + 64
    if len(body) > max_chars - reserve:
        available = max_chars - reserve
        head_chars = available * 2 // 3
        tail_chars = available - head_chars
        body = (
            body[:head_chars].rstrip()
            + "\n...[summary middle truncated]...\n"
            + body[-tail_chars:].lstrip()
        )
    body = f"{body}\n\n{_SUMMARY_END_MARKER}"
    return body


def _summary_body(summary: str) -> str:
    body = str(summary or "").strip()
    if body.startswith(SUMMARY_PREFIX):
        body = body[len(SUMMARY_PREFIX) :].lstrip()
    if body.rstrip().endswith(_SUMMARY_END_MARKER):
        body = body.rstrip()[: -len(_SUMMARY_END_MARKER)].rstrip()
    return body


class _RollingReducer:
    def __init__(self, max_summary_chars: int) -> None:
        self._max_summary_chars = max_summary_chars
        self._fallback = ContextCompressor(
            model="resume-hard-summary",
            config_context_length=128_000,
            quiet_mode=True,
            # The map phase must be strictly local. Lean-mode augmentation can
            # ask an auxiliary model for chunk digests; the single optional
            # provider call belongs exclusively to refine() below.
            tail_mode="legacy",
        )
        self._levels: list[str | None] = []
        self.summary: str | None = None
        self.used_generated_summary = False
        self.has_human_turn = False

    def add(self, messages: Sequence[Mapping[str, Any]]) -> None:
        if not messages:
            return
        batch = [_clean_message(message) for message in messages]
        self.has_human_turn = self.has_human_turn or any(
            message.get("role") == "user" and is_human_intent(message)
            for message in batch
        )
        self._fallback._previous_summary = None
        leaf = _ensure_summary_envelope(
            self._fallback._build_static_fallback_summary(
                batch, reason="cold resume required a bounded projection"
            ),
            self._max_summary_chars,
        )
        level = 0
        while True:
            if level == len(self._levels):
                self._levels.append(leaf)
                break
            previous = self._levels[level]
            if previous is None:
                self._levels[level] = leaf
                break
            self._levels[level] = None
            leaf = self._merge(previous, leaf)
            level += 1

        self.summary = None

    def _merge(self, older: str, newer: str) -> str:
        return _ensure_summary_envelope(
            "\n\n".join(
                (
                    "## Earlier Reduced Segment\n" + _summary_body(older),
                    "## Later Reduced Segment\n" + _summary_body(newer),
                )
            ),
            self._max_summary_chars,
        )

    def _fold_levels(self) -> str | None:
        # High levels hold older/larger ranges; low levels hold the newest
        # remainder. Fold in chronological order. Binary carry keeps any one
        # source range from being repeatedly truncated once per input page.
        ordered = [item for item in reversed(self._levels) if item]
        if not ordered:
            return None
        merged = ordered[0]
        for item in ordered[1:]:
            merged = self._merge(merged, item)
        return merged

    def materialize(self) -> str | None:
        self.summary = self._fold_levels()
        return self.summary

    def refine(self, summarize: SummaryCallback | None) -> None:
        """Optionally improve the bounded local reduction with one LLM call."""
        if summarize is None or not self.summary:
            return
        source = {
            # This is a derived digest, never a fresh user request. Preserve
            # the human-source fact as metadata for the summarizer contract
            # without mis-grounding the whole digest as the latest ask.
            "role": "assistant",
            "content": self.summary,
            COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: self.has_human_turn,
        }
        try:
            generated = summarize([source], None)
        except Exception:
            generated = None
        if generated and str(generated).strip():
            self.summary = _ensure_summary_envelope(
                str(generated), self._max_summary_chars
            )
            self.used_generated_summary = True


def _compose_candidate(
    summary: str,
    tail: Sequence[Mapping[str, Any]],
    *,
    has_human_turn: bool,
    max_summary_chars: int,
    max_tail_message_chars: int,
) -> list[dict[str, Any]]:
    exact_tail = []
    for message in tail:
        bounded = _bounded_tail_message(message, max_tail_message_chars)
        row_id = int(message.get("id") or message.get("_row_id") or 0)
        bounded["_resume_archive_rows"] = [row_id] if row_id > 0 else []
        exact_tail.append(bounded)

    # The repair helper can merge same-role rows. Attach the whole run's raw
    # identities to every possible survivor so the post-repair cap can still
    # point at all exact source payloads instead of inventing row 0.
    run_start = 0
    while run_start < len(exact_tail):
        run_end = run_start + 1
        role = exact_tail[run_start].get("role")
        while (
            run_end < len(exact_tail)
            and role in {"user", "assistant"}
            and exact_tail[run_end].get("role") == role
        ):
            run_end += 1
        if run_end - run_start > 1:
            run_rows = [
                row_id
                for item in exact_tail[run_start:run_end]
                for row_id in item.get("_resume_archive_rows", [])
            ]
            run_was_clipped = any(
                bool(item.get("_resume_payload_clipped"))
                for item in exact_tail[run_start:run_end]
            )
            for item in exact_tail[run_start:run_end]:
                item["_resume_archive_rows"] = run_rows
                if run_was_clipped:
                    item["_resume_payload_clipped"] = True
        run_start = run_end

    from agent.agent_runtime_helpers import repair_message_sequence

    repair_message_sequence(None, exact_tail)
    exact_tail = [
        _bounded_tail_message(message, max_tail_message_chars)
        for message in exact_tail
    ]
    if exact_tail and exact_tail[0].get("role") == "user" and is_human_intent(
        exact_tail[0]
    ):
        # The summary and the first retained user content share one durable
        # carrier, matching the normal compressor's strict-template shape.
        carrier = dict(exact_tail[0])
        carrier.pop("api_content", None)
        carrier["content"] = (
            f"{_ensure_summary_envelope(summary, max_summary_chars)}\n\n"
            f"{carrier.get('content') or ''}"
        )
        carrier[COMPRESSED_SUMMARY_METADATA_KEY] = True
        carrier[COMPRESSED_SUMMARY_HAS_USER_TURN_KEY] = bool(has_human_turn)
        candidate = [carrier, *exact_tail[1:]]
    elif exact_tail and exact_tail[0].get("role") == "user":
        # Pre-provenance sessions deliberately classify bare user rows as
        # legacy_unknown. Do not hide or relabel that exact row by merging the
        # summary into it. A hidden synthetic user + assistant summary pair
        # gives strict templates a valid leading exchange, then leaves the
        # legacy user turn untouched and visible.
        candidate = [
            {
                "role": "user",
                "content": "[Historical context summary follows.]",
                "display_kind": "hidden",
                "origin_kind": "internal_system",
                "turn_kind": "runtime_scaffolding",
                "trust_kind": "no_control",
            },
            {
                "role": "assistant",
                "content": _ensure_summary_envelope(summary, max_summary_chars),
                "display_kind": "hidden",
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: bool(has_human_turn),
            },
            *exact_tail,
        ]
    else:
        candidate = [
            {
                "role": "user",
                "content": _ensure_summary_envelope(summary, max_summary_chars),
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: bool(has_human_turn),
            },
            *exact_tail,
        ]

    validation = validate_projection(candidate)
    if not validation.valid:
        raise ResumeHardCompactionUnsafe(
            f"hard-summary candidate is not replay-safe: {validation.reason}"
        )
    return candidate


def _first_complete_tail_group_end(
    tail: Sequence[Mapping[str, Any]],
) -> int:
    """Return a safe boundary after the oldest retained conversational group."""
    if not tail:
        return 0
    role = tail[0].get("role")
    if role == "user":
        index = 1
        if index < len(tail) and tail[index].get("role") == "assistant":
            index += 1
            while index < len(tail) and tail[index].get("role") == "tool":
                index += 1
        return index
    if role == "assistant":
        index = 1
        while index < len(tail) and tail[index].get("role") == "tool":
            index += 1
        return index
    return 1


def compact_oversized_resume(
    db: Any,
    session_id: str,
    *,
    policy: ResumeHardCompactionPolicy | None = None,
    summarize: SummaryCallback | None = None,
    progress: ProgressCallback | None = None,
    force: bool = False,
) -> ResumeHardCompactionResult:
    """Publish a bounded model-facing projection without deleting raw history."""
    policy = policy or ResumeHardCompactionPolicy()
    policy.validate()
    watermark = int(db.get_active_message_watermark(session_id) or 0)
    if watermark <= 0:
        return ResumeHardCompactionResult("no_progress", 0, 0, watermark)

    strategy_fingerprint = hashlib.sha256(
        (
            f"resume-hard-v2:{policy.target_rows}:{policy.page_rows}:"
            f"{policy.summary_chunk_rows}:{policy.max_summary_chars}:"
            f"{policy.max_tail_message_chars}:{policy.max_projection_tokens}:"
            f"{policy.max_page_materialized_chars}:{policy.strategy_identity}"
        ).encode()
    ).hexdigest()
    owner = secrets.token_hex(16)
    holder = (
        f"resume-hard:pid={os.getpid()}:tid={threading.get_ident()}:"
        f"nonce={secrets.token_hex(8)}"
    )
    if not db.try_acquire_compression_lock(
        session_id, holder, ttl_seconds=policy.lease_seconds
    ):
        raise ResumeHardCompactionBusy("session compression lease is busy")

    refresher = _LeaseRefresher(
        db, session_id, holder, policy.lease_seconds
    )
    refresher.start()
    reducer = _RollingReducer(policy.max_summary_chars)
    # Reserve two rows for the strict-template wrapper needed by legacy user
    # rows whose provenance cannot safely be upgraded during recovery.
    max_tail_rows = policy.target_rows - 2
    tail: deque[dict[str, Any]] = deque()
    tail_token_costs: deque[int] = deque()
    tail_tokens = 0
    summary_chunk: list[dict[str, Any]] = []
    source_rows = 0
    after_id = 0
    committed = False
    admitted = False
    source_hasher = hashlib.sha256(
        f"resume-hard-v2:{session_id}:{watermark}".encode()
    )
    # The source reader bounds ten payload-bearing columns independently.
    # Adapt row count so even their worst-case combined page stays within the
    # configured working-memory envelope.
    effective_page_rows = max(
        1,
        min(
            policy.page_rows,
            policy.max_page_materialized_chars
            // (policy.max_tail_message_chars * 10),
        ),
    )
    try:
        while True:
            page = db.get_compaction_source_page(
                session_id,
                after_id=after_id,
                through_id=watermark,
                limit=effective_page_rows,
                max_field_chars=policy.max_tail_message_chars,
            )
            if not page:
                break
            for row in page:
                row_id = int(row.get("id") or 0)
                after_id = max(after_id, row_id)
                source_rows += 1
                fingerprint_payload = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
                source_hasher.update(len(fingerprint_payload).to_bytes(8, "big"))
                source_hasher.update(fingerprint_payload)
                bounded_row = _bounded_tail_message(
                    row, policy.max_tail_message_chars
                )
                # Preserve only the immutable source identity while the row is
                # resident. _clean_message removes it before summary/provider
                # materialization; compose uses it for archive references.
                bounded_row["id"] = row_id
                row_tokens = estimate_provider_wire_tokens(
                    {"messages": [_clean_message(bounded_row)]}
                )
                while tail and (
                    len(tail) >= max_tail_rows
                    or tail_tokens + row_tokens > policy.max_projection_tokens
                ):
                    summary_chunk.append(tail.popleft())
                    tail_tokens -= tail_token_costs.popleft()
                tail.append(bounded_row)
                tail_token_costs.append(row_tokens)
                tail_tokens += row_tokens
                if len(summary_chunk) >= policy.summary_chunk_rows:
                    reducer.add(summary_chunk)
                    summary_chunk = []
                    _report_progress(progress, "summarizing", source_rows)
            if len(page) < effective_page_rows:
                break

        admission = db.claim_context_compaction(
            session_id,
            source_hasher.hexdigest(),
            strategy_fingerprint,
            owner=owner,
            force=force,
        )
        if admission != "admitted":
            raise ResumeHardCompactionBusy(
                f"hard-summary compaction admission is {admission}"
            )
        admitted = True

        if summary_chunk:
            reducer.add(summary_chunk)
        reducer.materialize()
        if reducer.summary is None:
            db.finish_context_compaction(
                session_id, owner=owner, outcome="no_progress"
            )
            return ResumeHardCompactionResult(
                "no_progress", source_rows, source_rows, watermark
            )
        if refresher.lost:
            raise ResumeHardCompactionBusy("session compression lease was lost")

        retained_tail = list(tail)
        first_user = next(
            (
                index
                for index, message in enumerate(retained_tail)
                if message.get("role") == "user"
            ),
            None,
        )
        if first_user:
            # Align the exact tail to a human turn without silently dropping
            # the tool/assistant prefix displaced by that alignment.
            reducer.add(retained_tail[:first_user])
            retained_tail = retained_tail[first_user:]

        # Reserve the full possible summary envelope, then roll complete oldest
        # groups out of the exact tail until the complete model-facing message
        # payload fits its aggregate estimated budget. A row cap alone cannot
        # protect a route from hundreds of individually large messages.
        reducer.materialize()
        budget_summary = _ensure_summary_envelope(
            "S" * policy.max_summary_chars,
            policy.max_summary_chars,
        )
        while retained_tail:
            budget_candidate = _compose_candidate(
                budget_summary,
                retained_tail,
                has_human_turn=reducer.has_human_turn,
                max_summary_chars=policy.max_summary_chars,
                max_tail_message_chars=policy.max_tail_message_chars,
            )
            if (
                estimate_provider_wire_tokens({"messages": budget_candidate})
                <= policy.max_projection_tokens
            ):
                break
            group_end = _first_complete_tail_group_end(retained_tail)
            # The newest conversational group is the irreducible live intent.
            # If it cannot fit even after every older group is reduced, fail
            # explicitly and leave the original active projection untouched.
            if group_end <= 0 or group_end >= len(retained_tail):
                break
            reducer.add(retained_tail[:group_end])
            retained_tail = retained_tail[group_end:]
            reducer.materialize()

        # Exactly one compressor refinement operation per cold recovery.
        # Paging and hierarchy reduction above are local, bounded work, so a
        # 20k-row legacy tip cannot turn into dozens of per-page round-trips.
        reducer.refine(summarize)

        candidate = _compose_candidate(
            reducer.summary,
            retained_tail,
            has_human_turn=reducer.has_human_turn,
            max_summary_chars=policy.max_summary_chars,
            max_tail_message_chars=policy.max_tail_message_chars,
        )
        if len(candidate) > policy.target_rows:
            raise ResumeHardCompactionUnsafe(
                "hard-summary candidate exceeds its row budget"
            )
        estimated_projection_tokens = estimate_provider_wire_tokens(
            {"messages": candidate}
        )
        if estimated_projection_tokens > policy.max_projection_tokens:
            raise ResumeHardCompactionUnsafe(
                "hard-summary candidate exceeds its estimated token budget "
                f"({estimated_projection_tokens} > {policy.max_projection_tokens})"
            )
        active_rows = db.archive_and_compact(
            session_id,
            candidate,
            watermark=watermark,
            lock_holder=holder,
        )
        committed = True
        try:
            db.finish_context_compaction(
                session_id, owner=owner, outcome="committed"
            )
        except Exception:
            # Publication is authoritative. A journal-write failure must not
            # turn a committed transcript into a reported failure; its running
            # receipt remains conservative until lease expiry/recovery.
            pass
        _report_progress(progress, "committed", active_rows)
        return ResumeHardCompactionResult(
            "committed",
            source_rows,
            active_rows,
            watermark,
            reducer.used_generated_summary,
        )
    except Exception:
        if admitted and not committed:
            try:
                db.finish_context_compaction(
                    session_id, owner=owner, outcome="aborted"
                )
            except Exception:
                pass
        raise
    finally:
        refresher.stop()
        db.release_compression_lock(session_id, holder)
