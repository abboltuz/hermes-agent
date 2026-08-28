"""Durable semantic provenance for Hermes conversation messages.

Provider ``role`` is a wire-protocol concern.  It deliberately remains
independent from who authored a message, why the turn exists, whether it may
authorize control behaviour, and how a client should render it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping, Optional


class _ValueEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class OriginKind(_ValueEnum):
    HUMAN_USER = "human_user"
    EXTERNAL_ACTOR = "external_actor"
    ASSISTANT = "assistant"
    TOOL = "tool"
    INTERNAL_SYSTEM = "internal_system"
    AUTOMATION = "automation"
    AGENT = "agent"
    IMPORTED = "imported"
    LEGACY_UNKNOWN = "legacy_unknown"


class TurnKind(_ValueEnum):
    PROMPT = "prompt"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    NOTIFICATION = "notification"
    CONTINUATION = "continuation"
    TASK_INSTRUCTION = "task_instruction"
    RUNTIME_SCAFFOLDING = "runtime_scaffolding"
    DELIVERY_MIRROR = "delivery_mirror"
    UI_ACTION = "ui_action"
    LEGACY_UNKNOWN = "legacy_unknown"


class TrustKind(_ValueEnum):
    USER_AUTHORIZED = "user_authorized"
    TRUSTED_INTERNAL = "trusted_internal"
    UNTRUSTED_EXTERNAL = "untrusted_external"
    NO_CONTROL = "no_control"
    LEGACY_UNKNOWN = "legacy_unknown"


ORIGIN_KIND_FIELD = "origin_kind"
TURN_KIND_FIELD = "turn_kind"
TRUST_KIND_FIELD = "trust_kind"
PROVENANCE_METADATA_FIELD = "provenance_metadata"

SEMANTIC_FIELDS = frozenset(
    {
        ORIGIN_KIND_FIELD,
        TURN_KIND_FIELD,
        TRUST_KIND_FIELD,
        PROVENANCE_METADATA_FIELD,
    }
)

_PROVENANCE_METADATA_KEYS = frozenset(
    {
        "producer",
        "source",
        "event_id",
        "event_kind",
        "task_id",
        "job_id",
        "process_id",
        "delegation_id",
        "run_id",
        "generation_id",
        "platform",
        "profile",
        "session_id",
        "thread_id",
        "chat_id",
        "message_id",
        "actor_id",
        "authorized_via",
        "import_source",
        "recovered",
    }
)
_SHORT_METADATA_KEYS = frozenset(
    {"producer", "source", "event_kind", "platform", "profile", "import_source"}
)
_SHORT_METADATA_LIMIT = 128
_ID_METADATA_LIMIT = 512
_MAX_METADATA_FIELDS = 18
_MAX_METADATA_BYTES = 4096


@dataclass(frozen=True)
class MessageProvenance:
    origin_kind: OriginKind
    turn_kind: TurnKind
    trust_kind: TrustKind
    metadata: dict[str, Any]

    def as_message_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            ORIGIN_KIND_FIELD: self.origin_kind.value,
            TURN_KIND_FIELD: self.turn_kind.value,
            TRUST_KIND_FIELD: self.trust_kind.value,
        }
        if self.metadata:
            fields[PROVENANCE_METADATA_FIELD] = dict(self.metadata)
        return fields


def _enum_value(enum_type, value: Any):
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{enum_type.__name__} must be a string enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"invalid {enum_type.__name__}: {value!r}") from exc


def normalize_provenance_metadata(value: Any) -> dict[str, Any]:
    """Return a bounded, flat allow-listed metadata object.

    The sidecar is intentionally unsuitable for arbitrary payloads.  It keeps
    only routing/audit identifiers and rejects nested or unknown fields so
    credentials and unbounded event bodies cannot accidentally become durable.
    """
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("provenance_metadata must be a JSON object") from exc
    if not isinstance(value, Mapping):
        raise ValueError("provenance_metadata must be an object")
    if len(value) > _MAX_METADATA_FIELDS:
        raise ValueError("provenance_metadata contains too many fields")

    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        if key not in _PROVENANCE_METADATA_KEYS:
            raise ValueError(f"unsupported provenance metadata field: {key!r}")
        if raw is None or raw == "":
            continue
        if key == "recovered":
            if not isinstance(raw, bool):
                raise ValueError("provenance metadata recovered must be boolean")
            normalized[key] = raw
            continue
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise ValueError(f"provenance metadata {key} must be text or integer")
        if isinstance(raw, int):
            if raw < -(2**63) or raw > 2**63 - 1:
                raise ValueError(f"provenance metadata {key} integer is out of range")
            normalized[key] = raw
            continue
        limit = _SHORT_METADATA_LIMIT if key in _SHORT_METADATA_KEYS else _ID_METADATA_LIMIT
        bounded = raw.strip()[:limit]
        if bounded:
            normalized[key] = bounded
    if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("provenance_metadata exceeds the encoded size limit")
    return normalized


def build_provenance(
    origin_kind: OriginKind | str,
    turn_kind: TurnKind | str,
    trust_kind: TrustKind | str,
    metadata: Any = None,
) -> MessageProvenance:
    """Validate and construct a trusted in-process provenance value."""
    origin = _enum_value(OriginKind, origin_kind)
    turn = _enum_value(TurnKind, turn_kind)
    trust = _enum_value(TrustKind, trust_kind)
    bounded = normalize_provenance_metadata(metadata)

    if origin == OriginKind.HUMAN_USER and trust != TrustKind.USER_AUTHORIZED:
        raise ValueError("human_user provenance must be user_authorized")
    if trust == TrustKind.USER_AUTHORIZED and origin not in {
        OriginKind.HUMAN_USER,
        OriginKind.EXTERNAL_ACTOR,
    }:
        raise ValueError("only human or external actors may be user_authorized")
    if trust == TrustKind.TRUSTED_INTERNAL and origin in {
        OriginKind.HUMAN_USER,
        OriginKind.EXTERNAL_ACTOR,
        OriginKind.IMPORTED,
    }:
        raise ValueError("actor/import provenance cannot claim trusted_internal")
    if turn == TurnKind.TOOL_RESULT and origin not in {
        OriginKind.TOOL,
        OriginKind.IMPORTED,
    }:
        raise ValueError("tool_result provenance requires tool or imported origin")
    return MessageProvenance(origin, turn, trust, bounded)


def decode_message_provenance(message: Mapping[str, Any]) -> Optional[MessageProvenance]:
    """Decode complete provenance fields, returning ``None`` when absent/invalid."""
    if not all(field in message for field in (ORIGIN_KIND_FIELD, TURN_KIND_FIELD, TRUST_KIND_FIELD)):
        return None
    try:
        return build_provenance(
            message.get(ORIGIN_KIND_FIELD),
            message.get(TURN_KIND_FIELD),
            message.get(TRUST_KIND_FIELD),
            message.get(PROVENANCE_METADATA_FIELD),
        )
    except ValueError:
        return None


def stamp_provenance(
    message: MutableMapping[str, Any],
    origin_kind: OriginKind | str,
    turn_kind: TurnKind | str,
    trust_kind: TrustKind | str,
    metadata: Any = None,
) -> MutableMapping[str, Any]:
    """Stamp provenance at a trusted producer boundary."""
    message.update(
        build_provenance(origin_kind, turn_kind, trust_kind, metadata).as_message_fields()
    )
    return message


def provenance_for_runtime_turn(
    *,
    platform: Any,
    display_kind: Any = None,
    metadata: Any = None,
) -> MessageProvenance:
    """Classify a newly-created turn from trusted runtime/session context."""
    metadata_map = metadata if isinstance(metadata, Mapping) else {}
    source_value = str(metadata_map.get("source") or "").strip().lower()
    event_value = str(metadata_map.get("event_kind") or "").strip().lower()
    if source_value in {
        "external_actor",
        "external_webhook",
        "msgraph_webhook",
        "raft_bridge",
        "webhook",
    }:
        return build_provenance(
            OriginKind.EXTERNAL_ACTOR,
            TurnKind.NOTIFICATION,
            TrustKind.UNTRUSTED_EXTERNAL,
            metadata,
        )
    if source_value in {"kanban", "delegation", "subagent"} or event_value in {
        "kanban_wake",
        "async_delegation_complete",
    }:
        return build_provenance(
            OriginKind.AGENT,
            TurnKind.CONTINUATION,
            TrustKind.TRUSTED_INTERNAL,
            metadata,
        )
    if source_value in {"heartbeat", "loop", "goal", "cron"} or event_value in {
        "heartbeat_tick",
        "loop_tick",
        "goal_continuation",
    }:
        return build_provenance(
            OriginKind.AUTOMATION,
            TurnKind.CONTINUATION,
            TrustKind.TRUSTED_INTERNAL,
            metadata,
        )
    if source_value in {"plugin", "plugin_injection"}:
        return build_provenance(
            OriginKind.AUTOMATION,
            TurnKind.TASK_INSTRUCTION,
            TrustKind.UNTRUSTED_EXTERNAL,
            metadata,
        )
    if isinstance(display_kind, str) and display_kind in _DISPLAY_PROVENANCE:
        return build_provenance(*_DISPLAY_PROVENANCE[display_kind], metadata)

    surface = str(platform or "").strip().lower()
    if surface in {"cron", "heartbeat", "loop", "goal", "automation", "batch"}:
        return build_provenance(
            OriginKind.AUTOMATION,
            TurnKind.TASK_INSTRUCTION,
            TrustKind.TRUSTED_INTERNAL,
            metadata,
        )
    if surface in {
        "subagent",
        "delegate",
        "kanban",
        "curator",
        "memory_review",
        "skill_review",
    }:
        return build_provenance(
            OriginKind.AGENT,
            TurnKind.TASK_INSTRUCTION,
            TrustKind.TRUSTED_INTERNAL,
            metadata,
        )
    if surface == "webhook":
        return build_provenance(
            OriginKind.EXTERNAL_ACTOR,
            TurnKind.NOTIFICATION,
            TrustKind.UNTRUSTED_EXTERNAL,
            metadata,
        )
    if surface == "acp":
        return build_provenance(
            OriginKind.EXTERNAL_ACTOR,
            TurnKind.PROMPT,
            TrustKind.USER_AUTHORIZED,
            metadata,
        )
    if surface in {"", "cli", "tui", "desktop", "voice"}:
        return build_provenance(
            OriginKind.HUMAN_USER,
            TurnKind.PROMPT,
            TrustKind.USER_AUTHORIZED,
            metadata,
        )
    # Unknown/custom session sources have no authenticated human boundary in
    # this helper. Gateway/API/ACP ingress supplies an explicit envelope, so a
    # bare integration source must fail closed rather than acquire controls.
    return build_provenance(
        OriginKind.EXTERNAL_ACTOR,
        TurnKind.NOTIFICATION,
        TrustKind.UNTRUSTED_EXTERNAL,
        metadata,
    )


def provenance_for_gateway_ingress(
    *,
    platform: Any,
    internal: bool = False,
    is_bot: bool = False,
    authorized: bool = True,
    internal_source: Any = None,
    event_kind: Any = None,
    metadata: Any = None,
) -> MessageProvenance:
    """Classify one messaging-gateway event from trusted adapter facts."""
    surface = str(getattr(platform, "value", platform) or "").lower()
    if internal:
        source_value = str(internal_source or "").strip().lower()
        event_value = str(event_kind or "").strip().lower()
        if source_value in {
            "external_actor",
            "external_webhook",
            "msgraph_webhook",
            "raft_bridge",
            "webhook",
        }:
            return build_provenance(
                OriginKind.EXTERNAL_ACTOR,
                TurnKind.NOTIFICATION,
                TrustKind.UNTRUSTED_EXTERNAL,
                metadata,
            )
        if source_value in {"plugin", "plugin_injection"}:
            return build_provenance(
                OriginKind.AUTOMATION,
                TurnKind.TASK_INSTRUCTION,
                TrustKind.UNTRUSTED_EXTERNAL,
                metadata,
            )
        if source_value in {"kanban", "delegation", "subagent"} or event_value in {
            "kanban_wake",
            "async_delegation_complete",
        }:
            return build_provenance(
                OriginKind.AGENT,
                TurnKind.CONTINUATION,
                TrustKind.TRUSTED_INTERNAL,
                metadata,
            )
        if source_value in {"heartbeat", "loop", "goal", "cron"} or event_value in {
            "heartbeat_tick",
            "loop_tick",
            "goal_continuation",
        }:
            return build_provenance(
                OriginKind.AUTOMATION,
                TurnKind.CONTINUATION,
                TrustKind.TRUSTED_INTERNAL,
                metadata,
            )
        if event_value in {
            "crash_recovery",
            "recall_interrupt",
            "resume",
            "session_handoff",
        }:
            return build_provenance(
                OriginKind.INTERNAL_SYSTEM,
                TurnKind.CONTINUATION,
                TrustKind.TRUSTED_INTERNAL,
                metadata,
            )
        return build_provenance(
            OriginKind.INTERNAL_SYSTEM,
            TurnKind.NOTIFICATION,
            TrustKind.TRUSTED_INTERNAL,
            metadata,
        )
    if is_bot or surface == "webhook" or not authorized:
        return build_provenance(
            OriginKind.EXTERNAL_ACTOR,
            TurnKind.NOTIFICATION,
            TrustKind.UNTRUSTED_EXTERNAL,
            metadata,
        )
    return build_provenance(
        OriginKind.EXTERNAL_ACTOR,
        TurnKind.PROMPT,
        TrustKind.USER_AUTHORIZED,
        metadata,
    )


def strip_untrusted_provenance(message: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an ingress message while discarding all caller-supplied claims."""
    reserved = SEMANTIC_FIELDS | {"display_kind", "display_metadata"}
    return {key: value for key, value in message.items() if key not in reserved}


_DISPLAY_PROVENANCE = {
    "hidden": (
        OriginKind.INTERNAL_SYSTEM,
        TurnKind.RUNTIME_SCAFFOLDING,
        TrustKind.NO_CONTROL,
    ),
    "internal_notification": (
        OriginKind.INTERNAL_SYSTEM,
        TurnKind.NOTIFICATION,
        TrustKind.TRUSTED_INTERNAL,
    ),
    "async_delegation_complete": (
        OriginKind.AGENT,
        TurnKind.CONTINUATION,
        TrustKind.TRUSTED_INTERNAL,
    ),
    "auto_continue": (
        OriginKind.INTERNAL_SYSTEM,
        TurnKind.CONTINUATION,
        TrustKind.TRUSTED_INTERNAL,
    ),
    "model_switch": (
        OriginKind.INTERNAL_SYSTEM,
        TurnKind.NOTIFICATION,
        TrustKind.NO_CONTROL,
    ),
    "personality_switch": (
        OriginKind.INTERNAL_SYSTEM,
        TurnKind.NOTIFICATION,
        TrustKind.NO_CONTROL,
    ),
}


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, Mapping)
        )
    return ""


def classify_legacy_message(message: Mapping[str, Any]) -> MessageProvenance:
    """Safely classify history that predates the durable contract.

    Only durable structural evidence and narrowly bounded stable markers are
    recognized.  An unresolved user-role row is deliberately *not* guessed to
    be human intent.
    """
    existing = decode_message_provenance(message)
    if existing is not None:
        return existing

    display_kind = message.get("display_kind")
    display_metadata = message.get("display_metadata")
    if display_kind == "internal_notification" and isinstance(
        display_metadata, Mapping
    ):
        source_value = str(display_metadata.get("source") or "").strip().lower()
        if source_value in {"kanban", "delegation", "subagent"}:
            return build_provenance(
                OriginKind.AGENT,
                TurnKind.CONTINUATION,
                TrustKind.TRUSTED_INTERNAL,
            )
        if source_value in {"heartbeat", "loop", "goal", "cron"}:
            return build_provenance(
                OriginKind.AUTOMATION,
                TurnKind.CONTINUATION,
                TrustKind.TRUSTED_INTERNAL,
            )
        if source_value in {"plugin", "plugin_injection"}:
            return build_provenance(
                OriginKind.AUTOMATION,
                TurnKind.TASK_INSTRUCTION,
                TrustKind.UNTRUSTED_EXTERNAL,
            )
    if isinstance(display_kind, str) and display_kind in _DISPLAY_PROVENANCE:
        return build_provenance(*_DISPLAY_PROVENANCE[display_kind])
    if message.get("_compressed_summary"):
        return build_provenance(
            OriginKind.INTERNAL_SYSTEM,
            TurnKind.RUNTIME_SCAFFOLDING,
            TrustKind.NO_CONTROL,
        )
    if any(
        message.get(flag)
        for flag in (
            "_length_continuation_nudge",
            "_runtime_continuation_synthetic",
            "_todo_snapshot_synthetic",
            "_empty_recovery_synthetic",
            "_verification_stop_synthetic",
            "_pre_verify_synthetic",
            "_dropped_toolcall_nudge",
            "_kanban_stop_synthetic",
        )
    ):
        return build_provenance(
            OriginKind.INTERNAL_SYSTEM,
            TurnKind.RUNTIME_SCAFFOLDING,
            TrustKind.NO_CONTROL,
        )

    text = _message_text(message).strip()
    if text:
        from agent.context_compressor import (
            COMPRESSION_CONTINUATION_USER_CONTENT,
            MAX_ITERATIONS_SUMMARY_REQUEST,
            _LEGACY_COMPRESSION_CONTINUATION_USER_CONTENT,
        )
        from tools.todo_tool import TODO_INJECTION_HEADER

        exact_scaffolds = {
            COMPRESSION_CONTINUATION_USER_CONTENT,
            _LEGACY_COMPRESSION_CONTINUATION_USER_CONTENT,
            MAX_ITERATIONS_SUMMARY_REQUEST,
        }
        stable_scaffold_prefixes = (
            TODO_INJECTION_HEADER,
            "[System: Your previous response was truncated",
            "[System: The previous response was cut off",
            "[System: Your previous tool call",
        )
        if text in exact_scaffolds or text.startswith(stable_scaffold_prefixes):
            return build_provenance(
                OriginKind.INTERNAL_SYSTEM,
                TurnKind.RUNTIME_SCAFFOLDING,
                TrustKind.NO_CONTROL,
            )
        if text.startswith("[IMPORTANT: Background process "):
            return build_provenance(
                OriginKind.INTERNAL_SYSTEM,
                TurnKind.NOTIFICATION,
                TrustKind.TRUSTED_INTERNAL,
            )
        if text.startswith("[IMPORTANT: MCP servers have been reloaded."):
            return build_provenance(
                OriginKind.INTERNAL_SYSTEM,
                TurnKind.NOTIFICATION,
                TrustKind.NO_CONTROL,
            )
        if text.startswith("Cronjob Response: "):
            return build_provenance(
                OriginKind.AUTOMATION,
                TurnKind.DELIVERY_MIRROR,
                TrustKind.NO_CONTROL,
            )
        if text == "(imported conversation begins with an assistant reply)":
            return build_provenance(
                OriginKind.IMPORTED,
                TurnKind.RUNTIME_SCAFFOLDING,
                TrustKind.NO_CONTROL,
            )

    role = message.get("role")
    if role == "tool":
        return build_provenance(OriginKind.TOOL, TurnKind.TOOL_RESULT, TrustKind.NO_CONTROL)
    if role == "assistant":
        turn = TurnKind.TOOL_CALL if message.get("tool_calls") else TurnKind.RESPONSE
        return build_provenance(OriginKind.ASSISTANT, turn, TrustKind.NO_CONTROL)
    if role == "system":
        return build_provenance(
            OriginKind.INTERNAL_SYSTEM,
            TurnKind.RUNTIME_SCAFFOLDING,
            TrustKind.TRUSTED_INTERNAL,
        )

    return build_provenance(
        OriginKind.LEGACY_UNKNOWN,
        TurnKind.LEGACY_UNKNOWN,
        TrustKind.LEGACY_UNKNOWN,
    )


def normalize_message_for_durable_write(message: Mapping[str, Any]) -> dict[str, Any]:
    """Return a durable message with complete, validated provenance.

    Missing/invalid claims never default to human intent.  Structurally
    unambiguous assistant/tool rows and known legacy scaffolds are classified;
    everything else becomes ``legacy_unknown``.
    """
    normalized = dict(message)
    provenance = classify_legacy_message(normalized)
    normalized.update(provenance.as_message_fields())
    if provenance.origin_kind == OriginKind.LEGACY_UNKNOWN and not normalized.get("display_kind"):
        normalized["display_kind"] = "legacy_unknown"
    return normalized


def is_human_intent(message: Mapping[str, Any]) -> bool:
    provenance = decode_message_provenance(message) or classify_legacy_message(message)
    return provenance.origin_kind in {
        OriginKind.HUMAN_USER,
        OriginKind.EXTERNAL_ACTOR,
    } and provenance.turn_kind in {
        TurnKind.PROMPT,
        TurnKind.TASK_INSTRUCTION,
        TurnKind.UI_ACTION,
    }


def may_authorize_control(message: Mapping[str, Any]) -> bool:
    provenance = decode_message_provenance(message) or classify_legacy_message(message)
    return (
        provenance.trust_kind == TrustKind.USER_AUTHORIZED
        and provenance.origin_kind in {OriginKind.HUMAN_USER, OriginKind.EXTERNAL_ACTOR}
        and provenance.turn_kind in {
            TurnKind.PROMPT,
            TurnKind.TASK_INSTRUCTION,
            TurnKind.UI_ACTION,
        }
    )


def is_actionable_continuation(message: Mapping[str, Any]) -> bool:
    provenance = decode_message_provenance(message) or classify_legacy_message(message)
    return provenance.turn_kind in {
        TurnKind.CONTINUATION,
        TurnKind.TASK_INSTRUCTION,
    } and provenance.trust_kind in {
        TrustKind.TRUSTED_INTERNAL,
        TrustKind.USER_AUTHORIZED,
    }


def is_display_visible(message: Mapping[str, Any]) -> bool:
    if message.get("display_kind") == "hidden":
        return False
    provenance = decode_message_provenance(message) or classify_legacy_message(message)
    return provenance.turn_kind != TurnKind.RUNTIME_SCAFFOLDING


def display_actor(message: Mapping[str, Any]) -> str:
    """Return a stable neutral actor label for UI projection."""
    provenance = decode_message_provenance(message) or classify_legacy_message(message)
    labels = {
        OriginKind.HUMAN_USER: "user",
        OriginKind.EXTERNAL_ACTOR: "external",
        OriginKind.ASSISTANT: "assistant",
        OriginKind.TOOL: "tool",
        OriginKind.INTERNAL_SYSTEM: "system",
        OriginKind.AUTOMATION: "automation",
        OriginKind.AGENT: "agent",
        OriginKind.IMPORTED: "imported",
        OriginKind.LEGACY_UNKNOWN: "unknown",
    }
    return labels[provenance.origin_kind]


def merge_same_role_carrier_provenance(
    target: MutableMapping[str, Any],
    source: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Merge semantics when alternation forces two same-role turns together.

    A carrier containing genuine human intent retains that identity. Two
    incompatible non-human sources fail neutral rather than arbitrarily
    laundering one producer through the other.
    """
    target_provenance = decode_message_provenance(target) or classify_legacy_message(
        target
    )
    source_provenance = decode_message_provenance(source) or classify_legacy_message(
        source
    )
    if is_human_intent(target):
        selected = target_provenance
    elif is_human_intent(source):
        selected = source_provenance
    elif target_provenance == source_provenance:
        selected = target_provenance
    else:
        selected = build_provenance(
            OriginKind.LEGACY_UNKNOWN,
            TurnKind.LEGACY_UNKNOWN,
            TrustKind.LEGACY_UNKNOWN,
        )

    for field in SEMANTIC_FIELDS:
        target.pop(field, None)
    target.update(selected.as_message_fields())
    if is_human_intent(target):
        target.pop("display_kind", None)
        target.pop("display_metadata", None)
    return target


def strip_provenance_for_provider(message: Mapping[str, Any]) -> dict[str, Any]:
    """Project one message to provider-safe protocol fields.

    Existing transport adapters remain free to apply provider-specific
    filtering afterwards.  This boundary guarantees semantic, audit, and
    presentation sidecars cannot leak to any provider.
    """
    stripped_fields = SEMANTIC_FIELDS | {"display_kind", "display_metadata"}
    if not any(key in message for key in stripped_fields):
        return message if isinstance(message, dict) else dict(message)
    projected = {
        key: value
        for key, value in message.items()
        if key not in stripped_fields
    }
    return projected


def strip_provenance_messages_for_provider(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy-on-write provider projection for a complete message list."""
    projected = [strip_provenance_for_provider(message) for message in messages]
    return messages if all(a is b for a, b in zip(projected, messages)) else projected
