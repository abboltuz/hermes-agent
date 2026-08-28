"""Bounded provenance envelope for trusted internal gateway turns."""

from __future__ import annotations

from typing import Any

from agent.message_provenance import build_provenance

INTERNAL_TURN_FIELD = "_hermes_internal_turn"

_ALLOWED_DISPLAY_KINDS = frozenset(
    {
        "internal_notification",
        "async_delegation_complete",
    }
)
_TEXT_LIMITS = {
    "source": 128,
    "kind": 128,
    "platform": 128,
}
_ID_KEYS = (
    "event_id",
    "delegation_id",
    "session_id",
    "task_id",
    "job_id",
    "process_id",
    "run_id",
    "generation_id",
    "event_kind",
)
_ID_LIMIT = 512

_ALLOWED_PROXY_PROVENANCE = frozenset(
    {
        ("external_actor", "prompt", "user_authorized"),
        ("external_actor", "notification", "untrusted_external"),
        ("internal_system", "notification", "trusted_internal"),
        ("internal_system", "notification", "no_control"),
        ("internal_system", "continuation", "trusted_internal"),
        ("internal_system", "runtime_scaffolding", "no_control"),
        ("automation", "continuation", "trusted_internal"),
        ("automation", "task_instruction", "trusted_internal"),
        ("automation", "task_instruction", "untrusted_external"),
        ("agent", "continuation", "trusted_internal"),
        ("agent", "task_instruction", "trusted_internal"),
    }
)


def _bounded_text(value: Any, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"internal turn metadata field {field} must be text")
    return value.strip()[:limit]


def build_internal_turn_envelope(
    display_kind: Any = None,
    display_metadata: Any = None,
    provenance: Any = None,
) -> dict[str, Any]:
    """Return an authenticated, bounded turn envelope.

    The private proxy field is accepted only when the API server has a
    configured key.  Presentation and semantic provenance remain independent:
    an untrusted external turn often has no special display treatment but must
    still cross the proxy without becoming an authorized API user.
    """
    envelope: dict[str, Any] = {}
    has_display = display_kind is not None or display_metadata is not None
    if has_display:
        if not isinstance(display_kind, str):
            raise ValueError("invalid internal turn display kind")
        kind = display_kind.strip()[:128]
        if kind not in _ALLOWED_DISPLAY_KINDS:
            raise ValueError("invalid internal turn display kind")
        if (
            not isinstance(display_metadata, dict)
            or display_metadata.get("internal") is not True
        ):
            raise ValueError("internal turn metadata must set internal=true")

        source = _bounded_text(
            display_metadata.get("source"),
            field="source",
            limit=_TEXT_LIMITS["source"],
        )
        event_kind = _bounded_text(
            display_metadata.get("kind"),
            field="kind",
            limit=_TEXT_LIMITS["kind"],
        )
        if not source or not event_kind:
            raise ValueError("internal turn metadata requires source and kind")

        metadata: dict[str, Any] = {
            "source": source,
            "internal": True,
            "kind": event_kind,
        }
        for key in ("platform",):
            raw_value = display_metadata.get(key)
            if raw_value is None:
                continue
            value = _bounded_text(raw_value, field=key, limit=_TEXT_LIMITS[key])
            if value:
                metadata[key] = value
        for key in _ID_KEYS:
            value = display_metadata.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                raise ValueError(
                    f"internal turn metadata field {key} must be text or integer"
                )
            if isinstance(value, int) and not isinstance(value, bool):
                metadata[key] = value
                continue
            bounded = _bounded_text(value, field=key, limit=_ID_LIMIT)
            if bounded:
                metadata[key] = bounded
        if "recovered" in display_metadata:
            recovered = display_metadata["recovered"]
            if not isinstance(recovered, bool):
                raise ValueError(
                    "internal turn metadata field recovered must be boolean"
                )
            metadata["recovered"] = recovered
        envelope.update(
            {"display_kind": kind, "display_metadata": metadata}
        )

    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("internal turn provenance must be an object")
        try:
            normalized = build_provenance(
                provenance.get("origin_kind"),
                provenance.get("turn_kind"),
                provenance.get("trust_kind"),
                provenance.get("provenance_metadata"),
            )
        except ValueError as exc:
            raise ValueError(f"invalid internal turn provenance: {exc}") from exc
        semantic_key = (
            normalized.origin_kind.value,
            normalized.turn_kind.value,
            normalized.trust_kind.value,
        )
        if semantic_key not in _ALLOWED_PROXY_PROVENANCE:
            raise ValueError(
                "internal turn provenance is not allowed for proxy ingress"
            )
        envelope["provenance"] = normalized.as_message_fields()

    if not envelope:
        raise ValueError("internal turn envelope is empty")
    return envelope


def parse_internal_turn_envelope(
    value: Any,
) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Validate an inbound envelope and return its persistence sidecar."""
    if not isinstance(value, dict):
        raise ValueError("internal turn envelope must be an object")
    envelope = build_internal_turn_envelope(
        value.get("display_kind"),
        value.get("display_metadata"),
        value.get("provenance"),
    )
    return (
        envelope.get("display_kind"),
        envelope.get("display_metadata"),
        envelope.get("provenance"),
    )
