"""Continuable-session compression primitives.

This module is deliberately independent from the provider adapters.  It owns the
invariants that must hold regardless of which ContextEngine supplies narrative
summaries: durable rows are never deleted, the active projection is bounded, and
an unsafe projection is refused before a provider call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import re
import secrets
import threading
import weakref
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.message_provenance import is_human_intent


_RECOVERY_PREFIX = "[COMPACTION RECOVERY] session="
_PROVISIONAL_RECOVERY_PREFIX = "[COMPACTION RECOVERY PENDING] session="
_MAX_PREVIEW = 240
TOOL_PRESSURE_MIN_RECLAIM_TOKENS = 8192
TOOL_PRESSURE_SOFT_RATIO = 0.85
_PROVIDER_WIRE_BYTES_PER_TOKEN = 3


def _tokens(value: Any) -> int:
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4)
    if isinstance(value, Mapping):
        return sum(_tokens(k) + _tokens(v) for k, v in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return sum(_tokens(item) for item in value)
    return 1


def estimate_projection_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(_tokens(dict(message)) for message in messages)


def _provider_wire_token_bound(request: Mapping[str, Any]) -> int:
    """Estimate complete provider-wire usage in the token domain.

    Provider-specific tokenizers are not available at this layer.  Keep the
    existing structural token estimate, then conservatively account for JSON
    framing, escaping, route fields, and nested schemas at three serialized
    UTF-8 bytes per token.  Raw bytes are not tokens: treating them as equal
    falsely rejects large multilingual and schema-heavy requests that fit the
    provider context window.
    """
    public = {
        key: value for key, value in request.items()
        if not str(key).startswith("_") and not str(key).startswith("__")
    }
    serialized = json.dumps(public, ensure_ascii=True, sort_keys=True, default=str)
    serialized_bytes = len(serialized.encode("utf-8"))
    serialized_tokens = max(
        1,
        (serialized_bytes + _PROVIDER_WIRE_BYTES_PER_TOKEN - 1)
        // _PROVIDER_WIRE_BYTES_PER_TOKEN,
    )
    return max(_tokens(public), serialized_tokens)


def _strip_provider_private(value: Any) -> Any:
    """Copy provider data while removing durable/compression sidecars.

    The live transcript is never passed through this function in place.  Only
    the request copy is sanitized, including nested content blocks and tool
    envelopes.  Transport control keys (for example ``__bedrock_region__``)
    are intentionally preserved; they are consumed by dispatch before the
    SDK call and are not transcript sidecars.
    """
    if isinstance(value, Mapping):
        cleaned = {}
        for key, child in value.items():
            name = str(key)
            if name in {"_row_id", "_db_persisted"} or name.startswith(
                ("_compression", "_micro_compact")
            ):
                continue
            cleaned[key] = _strip_provider_private(child)
        return cleaned
    if isinstance(value, list):
        return [_strip_provider_private(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_strip_provider_private(child) for child in value)
    return value


def _wire_request_fits(agent: Any, request: Mapping[str, Any], *, output_reserve: int, safety_margin: int) -> bool:
    context_window = getattr(agent, "_config_context_length", None)
    if not isinstance(context_window, int) or context_window <= 0:
        compressor = getattr(agent, "context_compressor", None)
        context_window = getattr(compressor, "context_length", None)
    if not isinstance(context_window, int) or context_window <= 0:
        return True
    return _provider_wire_token_bound(request) + output_reserve + safety_margin <= context_window


def _has_incomplete_tool_group(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether the projection ends with an unanswered tool call."""
    calls = _tool_call_ids(messages)
    answered = {
        str(message.get("tool_call_id", ""))
        for message in messages
        if message.get("role") == "tool"
    }
    return bool(calls - answered)


def prune_tool_pressure_projection(
    agent: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    current_tokens: int | None = None,
    min_reclaim_tokens: int = TOOL_PRESSURE_MIN_RECLAIM_TOKENS,
) -> tuple[list, int]:
    """Bound completed tool history without changing durable transcript rows.

    This is intentionally a projection-only operation.  The executor has already
    persisted the completed group when this function is called; the returned
    list is therefore the only object that may be replaced.  An unanswered
    group makes the operation a conservative no-op rather than risking a call /
    result split.
    """
    source = [dict(message) for message in messages]
    unchanged = messages if isinstance(messages, list) else source
    if _has_incomplete_tool_group(source) or not validate_projection(source).valid:
        return unchanged, 0
    context_window = getattr(agent, "_config_context_length", None)
    if not isinstance(context_window, int) or context_window <= 0:
        compressor = getattr(agent, "context_compressor", None)
        context_window = getattr(compressor, "context_length", None)
    if not isinstance(context_window, int) or context_window <= 0:
        return unchanged, 0
    before = estimate_projection_tokens(source)
    observed = current_tokens if isinstance(current_tokens, int) and current_tokens > 0 else before
    soft_budget = int(context_window * TOOL_PRESSURE_SOFT_RATIO)
    if observed <= soft_budget:
        return unchanged, 0
    # ``observed`` is the full request estimate (messages plus system/tool
    # payload and wire overhead), while ``before`` is only the message
    # projection.  Keep the external portion as a bounded floor rather than
    # requiring the message projection to exceed the whole-request threshold.
    non_message_floor = max(0, observed - before)
    if non_message_floor >= soft_budget:
        return unchanged, 0
    coordinator = ensure_compression_coordinator(
        agent, trigger="tool_pressure", urgency=1
    )
    source_fingerprint = hashlib.sha256(
        json.dumps(source, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    if getattr(coordinator, "_tool_pressure_fingerprint", None) == source_fingerprint:
        return unchanged, 0
    output_reserve = getattr(agent, "max_tokens", 0)
    if not isinstance(output_reserve, int) or output_reserve < 0:
        output_reserve = 0
    safety_margin = getattr(agent, "_compression_safety_margin", 1024)
    if not isinstance(safety_margin, int) or safety_margin < 0:
        safety_margin = 1024
    # emergency_context_cut is also the canonical pairing/retention policy.
    # Adjust its context window so its safe input budget is exactly the
    # remaining message target after reserving the non-message floor.
    message_target = soft_budget - non_message_floor
    budget = CompressionBudget(
        message_target + output_reserve + safety_margin,
        output_reserve,
        safety_margin,
    )
    result = emergency_context_cut(
        source,
        budget,
        session_id=str(getattr(agent, "session_id", "") or ""),
        generation=int(getattr(agent, "_compression_generation", 0) or 0),
        watermark=int(getattr(agent, "_session_watermark", 0) or 0),
        min_reclaim_tokens=min_reclaim_tokens,
    )
    if not result.provider_call_allowed:
        return unchanged, 0
    after = estimate_projection_tokens(result.messages)
    reclaimed = max(0, before - after)
    if reclaimed < min_reclaim_tokens or after + non_message_floor > soft_budget:
        return unchanged, 0
    if result.recovery_identity and not _bind_recovery_identity(
        agent, source, result.messages, result.recovery_identity,
        int(getattr(agent, "_compression_generation", 0) or 0),
        int(getattr(agent, "_session_watermark", 0) or 0),
    ):
        return unchanged, 0
    coordinator.active_projection = [dict(message) for message in result.messages]
    coordinator._tool_pressure_fingerprint = hashlib.sha256(
        json.dumps(result.messages, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    setattr(
        agent,
        "_compression_projection_watermark",
        int(getattr(agent, "_compression_projection_watermark", 0) or 0) + 1,
    )
    return result.messages, reclaimed


@dataclass(frozen=True)
class CompressionBudget:
    context_window: int
    output_reserve: int
    safety_margin: int
    system_tokens: int = 0
    tool_schema_tokens: int = 0
    wire_overhead_tokens: int = 0

    @property
    def safe_input_budget(self) -> int:
        return max(0, self.context_window - self.output_reserve - self.safety_margin)

    @property
    def history_budget(self) -> int:
        return max(0, self.safe_input_budget - self.system_tokens - self.tool_schema_tokens - self.wire_overhead_tokens)

    def fits(self, messages: Sequence[Mapping[str, Any]]) -> bool:
        return self.system_tokens + self.tool_schema_tokens + self.wire_overhead_tokens + estimate_projection_tokens(messages) <= self.safe_input_budget


@dataclass(frozen=True)
class PolicyCapsule:
    latest_human_intent: str
    constraints: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    session_id: str = ""
    generation: int = 0
    watermark: int = 0
    provenance: str = "Cekasha"

    def message(self) -> dict[str, Any]:
        body = {
            "latest_human_intent": self.latest_human_intent,
            "constraints": list(self.constraints),
            "identifiers": list(self.identifiers),
            "session_id": self.session_id,
            "generation": self.generation,
            "watermark": self.watermark,
            "provenance": self.provenance,
        }
        return {"role": "system", "content": "[POLICY CAPSULE]\n" + json.dumps(body, ensure_ascii=False, sort_keys=True), "_compression_capsule": True}


def build_policy_capsule(messages: Iterable[Mapping[str, Any]], *, session_id: str, watermark: int, generation: int) -> PolicyCapsule:
    human = ""
    constraints: list[str] = []
    identifiers: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "user" and is_human_intent(message):
            human = str(message.get("content", ""))
        text = str(message.get("content", ""))
        if re.search(r"\b(?:MUST|NEVER|REQUIRED|approval|approv(?:e|al))\b", text, re.I):
            constraints.append(text[:_MAX_PREVIEW])
        identifiers.update(re.findall(r"(?:#[0-9]+|[0-9a-f]{7,40}|(?:/|\\)[\w./\\-]+)", text))
    return PolicyCapsule(human, tuple(dict.fromkeys(constraints[-32:])), tuple(sorted(identifiers)), session_id, generation, watermark)


def _bind_recovery_identity(agent: Any, source: Sequence[Mapping[str, Any]], retained: Sequence[Mapping[str, Any]], identity: str, generation: int, watermark: int) -> str | None:
    """Publish a marker only after canonical rows are proven durable."""
    db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    register = getattr(db, "register_compression_recovery", None)
    if not db or not session_id or not callable(register):
        return None
    def durable_id(message: Mapping[str, Any]) -> int | None:
        value = message.get("_row_id")
        return value if isinstance(value, int) and value > 0 else None

    try:
        from agent.context_compressor import ContextCompressor
    except Exception:
        ContextCompressor = None

    def is_synthetic(message: Mapping[str, Any]) -> bool:
        return bool(
            ContextCompressor is not None
            and ContextCompressor._is_synthetic_compression_user_turn(message)
        )

    try:
        # row marker.  The marker is never sent to a provider and is absent on
        # synthetic projection-only rows; refusing missing demotions is safer
        # than aliasing an older duplicate found by content.
        retained_ids = {row_id for message in retained if (row_id := durable_id(message)) is not None}
        ids = []
        for message in source:
            if message.get("role") == "system":
                continue
            row_id = durable_id(message)
            if row_id is None and is_synthetic(message):
                continue
            if row_id is None:
                return None
            if row_id not in retained_ids:
                ids.append(row_id)
        if not ids:
            return None
        fingerprint = hashlib.sha256(
            json.dumps([durable_id(message) for message in source], ensure_ascii=False).encode()
        ).hexdigest()
        canonical = register(
            session_id, ids, generation=generation, watermark=watermark,
            projection_fingerprint=fingerprint, recovery_identity=identity,
        )
        if not isinstance(canonical, str) or not canonical:
            return None
        old = f"{_PROVISIONAL_RECOVERY_PREFIX}{session_id} anchor={identity}"
        new = f"{_RECOVERY_PREFIX}{session_id} anchor={canonical}"
        for message in retained:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                message["content"] = message["content"].replace(old, new)
        return canonical
    except Exception:
        return None


def _tool_call_ids(messages: Sequence[Mapping[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or ():
                if isinstance(call, Mapping) and call.get("id"):
                    ids.add(str(call["id"]))
    return ids


@dataclass(frozen=True)
class ProjectionValidation:
    valid: bool
    reason: str = ""


def validate_projection(messages: Sequence[Mapping[str, Any]]) -> ProjectionValidation:
    calls = _tool_call_ids(messages)
    results: set[str] = set()
    previous_visible: str | None = None
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            if call_id not in calls:
                return ProjectionValidation(False, "orphan_tool_result")
            if call_id in results:
                return ProjectionValidation(False, "duplicate_tool_result")
            results.add(call_id)
            continue
        if role == "assistant" and message.get("tool_calls"):
            continue
        if role not in {"system", "user", "assistant"}:
            return ProjectionValidation(False, "unknown_role")
        if previous_visible == role and role != "system":
            return ProjectionValidation(False, "role_alternation")
        previous_visible = role
    missing = calls - results
    if missing:
        return ProjectionValidation(False, "missing_tool_result")
    return ProjectionValidation(True)


@dataclass(frozen=True)
class CutResult:
    messages: list[dict[str, Any]]
    outcome: str
    provider_call_allowed: bool
    recovery_identity: str = ""
    reason: str = ""


class ContextProjectionUnfit(RuntimeError):
    """Raised before transport when the irreducible request floor cannot fit."""

    outcome = "context_projection_unfit"

    def __init__(self, result: CutResult) -> None:
        self.result = result
        super().__init__(result.reason or self.outcome)


def _round_groups(messages: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Return complete tool envelopes with their source span.

    Narrative rows are deliberately not groups: they must not consume the
    semantic-round retention quota.  Results may arrive in any order, but an
    envelope is complete only when its contiguous tool-result tail contains
    exactly each declared call id.
    """
    groups: list[tuple[int, int, list[dict[str, Any]]]] = []
    index = 0
    while index < len(messages):
        item = dict(messages[index])
        calls = item.get("tool_calls") if item.get("role") == "assistant" else None
        expected = {
            str(call.get("id"))
            for call in calls or ()
            if isinstance(call, Mapping) and call.get("id")
        }
        if not expected:
            index += 1
            continue
        start = index
        group = [item]
        seen: set[str] = set()
        index += 1
        while index < len(messages) and seen != expected:
            result = dict(messages[index])
            if result.get("role") != "tool":
                break
            call_id = str(result.get("tool_call_id", ""))
            group.append(result)
            seen.add(call_id)
            index += 1
        if seen == expected and len(group) == len(expected) + 1:
            groups.append((start, index, group))
    return groups


def emergency_context_cut(
    messages: Sequence[Mapping[str, Any]],
    budget: CompressionBudget,
    *,
    session_id: str,
    generation: int,
    watermark: int = 0,
    min_reclaim_tokens: int = 0,
    wire_fit: Callable[[Sequence[Mapping[str, Any]]], bool] | None = None,
) -> CutResult:
    original = [dict(message) for message in messages]
    min_reclaim_tokens = max(0, int(min_reclaim_tokens))
    original_projection_tokens = estimate_projection_tokens(original)
    if min_reclaim_tokens == 0 and budget.fits(original) and (wire_fit is None or wire_fit(original)):
        return CutResult(original, "emergency_context_cut", True)
    capsule = build_policy_capsule(original, session_id=session_id, watermark=watermark, generation=generation)
    system = [dict(m) for m in original if m.get("role") == "system" and not m.get("_compression_capsule")][:1]
    human = [dict(m) for m in original if m.get("role") == "user" and is_human_intent(m)][-1:]
    non_system = [(index, dict(message)) for index, message in enumerate(original) if message.get("role") != "system"]
    groups = _round_groups([message for _, message in non_system])
    retained = groups[-6:]
    selected: list[tuple[int, dict[str, Any]]] = []
    for start, end, group in retained:
        for offset, item in enumerate(group):
            selected.append((non_system[start + offset][0], item))
    # Keep the latest human at its original position relative to retained
    # envelopes.  It is inserted once, rather than once in both prefix and
    # compacted history, so validation cannot reject an otherwise safe cut.
    human_index = next(
        (index for index in range(len(original) - 1, -1, -1)
         if original[index].get("role") == "user" and is_human_intent(original[index])),
        None,
    )
    if human:
        selected.append((human_index if human_index is not None else len(original), human[0]))
    selected.sort(key=lambda pair: pair[0])
    identity = secrets.token_urlsafe(24)
    reference = f"{_PROVISIONAL_RECOVERY_PREFIX}{session_id} anchor={identity} watermark={watermark}"
    compacted: list[dict[str, Any]] = []
    for _, item in selected:
        if item.get("role") == "tool" and _tokens(item.get("content", "")) > max(16, budget.history_budget // 3):
            item["content"] = reference + " preview=" + str(item.get("content", ""))[:_MAX_PREVIEW]
        compacted.append(item)
    recovery = {"role": "system", "content": reference, "_compression_recovery": True}
    prefix = system + [capsule.message(), recovery]
    candidate = prefix + compacted
    def meets_target(candidate_messages: Sequence[Mapping[str, Any]]) -> bool:
        reclaimed = original_projection_tokens - estimate_projection_tokens(candidate_messages)
        return (
            validate_projection(candidate_messages).valid
            and budget.fits(candidate_messages)
            and (wire_fit is None or wire_fit(candidate_messages))
            and reclaimed >= min_reclaim_tokens
        )

    if meets_target(candidate):
        return CutResult(candidate, "emergency_context_cut", True, identity)
    # A retained round is semantically protected, but its bulk body is not.
    # Compact oldest retained bodies progressively when the six-round envelope
    # still exceeds the safe input budget.
    for item in compacted:
        if item.get("role") != "tool":
            continue
        content = str(item.get("content", ""))
        if content.startswith(_RECOVERY_PREFIX):
            continue
        item["content"] = reference + " preview=" + content[:_MAX_PREVIEW]
        candidate = prefix + compacted
        if meets_target(candidate):
            break
    validation = validate_projection(candidate)
    if meets_target(candidate):
        return CutResult(candidate, "emergency_context_cut", True, identity)
    if validation.valid and budget.fits(candidate) and (wire_fit is None or wire_fit(candidate)):
        return CutResult(
            original,
            "context_projection_min_reclaim_unmet",
            False,
            identity,
            "minimum reclaim target cannot be met by eligible bodies",
        )
    if not budget.fits(system + [capsule.message()] + human) or (
        wire_fit is not None and not wire_fit(system + [capsule.message()] + human)
    ):
        return CutResult(original, "context_projection_unfit", False, identity, "irreducible system/tool/policy/user floor exceeds safe input budget")
    # A malformed retained tail is never sent; return the safe floor and let the
    # caller rebuild the next tool group rather than inventing a pairing.
    floor = system + [capsule.message()] + human
    if validate_projection(floor).valid and budget.fits(floor):
        if meets_target(floor):
            return CutResult(floor, "emergency_context_cut", True, identity, validation.reason)
        return CutResult(
            original,
            "context_projection_min_reclaim_unmet",
            False,
            identity,
            "minimum reclaim target cannot be met by eligible bodies",
        )
    return CutResult(original, "context_projection_unfit", False, identity, validation.reason)


def prepare_api_request(agent: Any, api_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the v3 fit gate to the actual provider-bound request."""
    # Keep sidecars available to compression/recovery, and sanitize only the
    # copy used for fit accounting and provider return. Removing row IDs before
    # the cut would make durable recovery binding fail closed.
    request = dict(api_kwargs)
    wire_request = _strip_provider_private(request)
    messages = request.get("messages")
    compressor = getattr(agent, "context_compressor", None)
    context_window = getattr(agent, "_config_context_length", None)
    if not isinstance(context_window, int) or context_window <= 0:
        context_window = getattr(compressor, "context_length", None)
    output_reserve = request.get("max_tokens", request.get("max_completion_tokens", 0))
    if "max_output_tokens" in request:
        output_reserve = request["max_output_tokens"]
    inference_config = request.get("inferenceConfig")
    if isinstance(inference_config, Mapping) and isinstance(
        inference_config.get("maxTokens"), int
    ):
        # Bedrock Converse carries output capacity in its nested control wire.
        output_reserve = inference_config["maxTokens"]
    if not isinstance(output_reserve, int) or output_reserve < 0:
        output_reserve = 0
    safety_margin = getattr(agent, "_compression_safety_margin", 1024)
    if not isinstance(safety_margin, int) or safety_margin < 0:
        safety_margin = 1024
    # Native Responses-shaped payloads have no canonical ``messages`` list.
    # They still must pass the final provider-wire guard before dispatch.
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        if not _wire_request_fits(
            agent, wire_request, output_reserve=output_reserve, safety_margin=safety_margin
        ):
            raise ContextProjectionUnfit(CutResult(
                [], "context_projection_unfit", False,
                reason="final provider wire payload exceeds safe context budget",
            ))
        return wire_request
    if not isinstance(context_window, int) or context_window <= 0:
        return wire_request
    tools = request.get("tools") or ()
    tool_tokens = estimate_projection_tokens(tools) if isinstance(tools, Sequence) else 0
    budget = CompressionBudget(context_window, output_reserve, safety_margin, tool_schema_tokens=tool_tokens)
    if budget.fits(messages) and _wire_request_fits(
        agent, wire_request, output_reserve=output_reserve, safety_margin=safety_margin
    ):
        return wire_request
    coordinator = ensure_compression_coordinator(agent, trigger="pre_send_fit_gate", urgency=3)
    def _candidate_wire_fits(candidate_messages: Sequence[Mapping[str, Any]]) -> bool:
        candidate_request = dict(request)
        candidate_request["messages"] = list(candidate_messages)
        return _wire_request_fits(
            agent,
            _strip_provider_private(candidate_request),
            output_reserve=output_reserve,
            safety_margin=safety_margin,
        )
    result = emergency_context_cut(
        messages,
        budget,
        session_id=str(getattr(agent, "session_id", "") or ""),
        generation=int(getattr(agent, "_compression_generation", 0) or 0),
        watermark=int(getattr(agent, "_session_watermark", 0) or 0),
        min_reclaim_tokens=0,
        wire_fit=_candidate_wire_fits,
    )
    request["messages"] = result.messages
    wire_request = _strip_provider_private(request)
    if not result.provider_call_allowed or not budget.fits(result.messages) or not _wire_request_fits(
        agent, wire_request, output_reserve=output_reserve, safety_margin=safety_margin
    ):
        if result.provider_call_allowed:
            result = replace(result, outcome="context_projection_unfit", provider_call_allowed=False, reason="final provider wire payload exceeds safe context budget")
        raise ContextProjectionUnfit(result)
    if result.recovery_identity and not _bind_recovery_identity(
        agent, messages, result.messages, result.recovery_identity,
        int(getattr(agent, "_compression_generation", 0) or 0),
        int(getattr(agent, "_session_watermark", 0) or 0),
    ):
        raise ContextProjectionUnfit(replace(result, outcome="context_projection_unfit", provider_call_allowed=False, reason="durable recovery registration failed"))
    coordinator.active_projection = [dict(message) for message in result.messages]
    # Binding may replace the provisional recovery marker in retained rows;
    # serialize only after that owner-process mutation is complete.
    return _strip_provider_private(request)


@dataclass(frozen=True)
class CompressionRequest:
    session_id: str
    generation: int
    trigger: str
    urgency: int = 1
    source_fingerprint: str = ""
    row_watermark: int = 0
    estimated_pressure: int = 0
    estimated_reclaim: int = 0
    deadline: float | None = None
    force: bool = False


@dataclass(frozen=True)
class CompressionAdmission:
    outcome: str
    attempt: "CompressionAttempt | None" = None


@dataclass(frozen=True)
class CompressionCandidate:
    session_id: str
    generation: int
    watermark: int
    prefix_hash: str
    schema_hash: str
    messages: list[dict[str, Any]]


@dataclass(frozen=True)
class CompressionSnapshot:
    """Immutable fence describing the stable prefix a worker may read."""

    session_id: str
    generation: int
    watermark: int
    prefix_hash: str
    schema_hash: str


def compression_route_is_eligible(route: Mapping[str, Any] | None) -> bool:
    """Require an explicit separately certified fast non-reasoning route."""
    return bool(
        isinstance(route, Mapping)
        and route.get("provider")
        and route.get("model")
        and route.get("certified_fast") is True
        and route.get("reasoning") is False
    )


@dataclass(frozen=True)
class CompressionAttempt:
    attempt_id: int
    generation: int
    urgency: int
    trigger: str


class CompressionCoordinator:
    """Small per-session owner that coalesces pressure requests and fences adopts."""

    def __init__(self, *, session_id: str, prefix_hash: str = "", schema_hash: str = "") -> None:
        self.session_id = session_id
        self.prefix_hash = prefix_hash
        self.schema_hash = schema_hash
        self._next_attempt = 0
        self._attempt: CompressionAttempt | None = None
        self.outcome: str | None = None
        self.active_projection: list[dict[str, Any]] | None = None
        self._tool_pressure_fingerprint: str | None = None
        self._admission_lock = threading.RLock()
        self._active_admission: tuple[int, str] | None = None
        self._terminal_generations: set[tuple[int, str]] = set()
        self._pending_request: CompressionRequest | None = None

    def request(self, request: CompressionRequest) -> CompressionAttempt:
        if request.session_id != self.session_id:
            raise ValueError("compression request belongs to another session")
        if self._attempt is not None and self._attempt.generation == request.generation:
            self._attempt = replace(self._attempt, urgency=max(self._attempt.urgency, request.urgency), trigger=request.trigger if request.urgency >= self._attempt.urgency else self._attempt.trigger)
            return self._attempt
        self._next_attempt += 1
        self._attempt = CompressionAttempt(self._next_attempt, request.generation, request.urgency, request.trigger)
        self.outcome = None
        return self._attempt

    def admit_execution(self, request: CompressionRequest) -> CompressionAdmission:
        """Admit one executor for a logical generation and source snapshot."""
        if request.session_id != self.session_id:
            raise ValueError("compression request belongs to another session")
        key = (request.generation, request.source_fingerprint)
        with self._admission_lock:
            attempt = self.request(request)
            if key in self._terminal_generations and not request.force:
                return CompressionAdmission("no_progress_suppressed", attempt)
            if self._active_admission == key:
                return CompressionAdmission("joined", attempt)
            if self._active_admission is not None:
                return CompressionAdmission("deferred_lock", attempt)
            self._active_admission = key
            return CompressionAdmission("admitted", attempt)

    def finish_execution(self, request: CompressionRequest, outcome: str) -> None:
        """Release admission and retain no-progress terminals for this snapshot."""
        key = (request.generation, request.source_fingerprint)
        with self._admission_lock:
            if self._active_admission == key:
                self._active_admission = None
            self.outcome = outcome
            if outcome in {"no_progress", "timed_out", "aborted"}:
                self._terminal_generations.add(key)

    @property
    def attempt(self) -> CompressionAttempt | None:
        return self._attempt

    def adopt(self, candidate: CompressionCandidate, *, current_watermark: int | None = None) -> bool:
        attempt = self._attempt
        if attempt is None or candidate.session_id != self.session_id or candidate.generation != attempt.generation:
            return False
        if candidate.prefix_hash != self.prefix_hash or candidate.schema_hash != self.schema_hash:
            return False
        if current_watermark is not None and candidate.watermark > current_watermark:
            return False
        if not validate_projection(candidate.messages).valid:
            return False
        self.active_projection = [dict(message) for message in candidate.messages]
        self.outcome = "candidate_adopted"
        return True

    def snapshot(self, *, watermark: int, generation: int | None = None) -> CompressionSnapshot:
        """Capture worker input without copying mutable live agent state."""
        active_generation = self.attempt.generation if self.attempt else 0
        return CompressionSnapshot(
            self.session_id,
            active_generation if generation is None else generation,
            watermark,
            self.prefix_hash,
            self.schema_hash,
        )

    def adopt_with_tail(
        self,
        candidate: CompressionCandidate,
        concurrent_tail: Iterable[Mapping[str, Any]],
        *,
        current_watermark: int,
    ) -> bool:
        """Validate and publish a candidate plus rows appended after its fence."""
        if candidate.watermark > current_watermark:
            return False
        tail = [dict(message) for message in concurrent_tail]
        combined = [dict(message) for message in candidate.messages] + tail
        if not validate_projection(combined).valid:
            return False
        if not self.adopt(candidate, current_watermark=current_watermark):
            return False
        self.active_projection = combined
        self.outcome = "candidate_adopted_with_tail"
        return True

    def mark_no_progress(self) -> bool:
        if self._attempt is None or self.outcome == "no_progress":
            return False
        self.outcome = "no_progress"
        return True


def ensure_compression_coordinator(agent: Any, *, trigger: str, urgency: int = 1) -> CompressionCoordinator:
    """Return the coordinator bound to an agent's current session identity."""
    physical_id = str(getattr(agent, "session_id", "") or "")
    root_getter = getattr(agent, "_conversation_root_id", None)
    try:
        logical_id = str(root_getter() or physical_id) if callable(root_getter) else physical_id
    except Exception:
        logical_id = physical_id
    session_db = getattr(agent, "_session_db", None)
    registry = globals().setdefault("_COMPRESSION_COORDINATORS", {})
    registry_lock = globals().setdefault("_COMPRESSION_COORDINATORS_LOCK", threading.RLock())
    with registry_lock:
        if session_db is not None:
            weak_registry = globals().setdefault(
                "_COMPRESSION_COORDINATORS_BY_DB", weakref.WeakKeyDictionary()
            )
            try:
                db_registry = weak_registry.setdefault(session_db, {})
                coordinator = db_registry.get(logical_id)
                if not isinstance(coordinator, CompressionCoordinator):
                    coordinator = CompressionCoordinator(session_id=logical_id)
                    db_registry[logical_id] = coordinator
            except TypeError:
                # A third-party SessionDB shim may not support weak references;
                # retain compatibility without using recyclable object ids.
                registry_key = (logical_id, id(session_db))
                coordinator = registry.get(registry_key)
                if not isinstance(coordinator, CompressionCoordinator):
                    coordinator = CompressionCoordinator(session_id=logical_id)
                    registry[registry_key] = coordinator
        else:
            registry_key = (logical_id, id(agent))
            coordinator = registry.get(registry_key)
            if not isinstance(coordinator, CompressionCoordinator):
                coordinator = CompressionCoordinator(session_id=logical_id)
                registry[registry_key] = coordinator
    setattr(agent, "_compression_coordinator", coordinator)
    generation = int(getattr(agent, "_compression_generation", 0) or 0)
    coordinator.request(CompressionRequest(logical_id, generation, trigger, urgency))
    return coordinator
