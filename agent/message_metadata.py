"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset(
    {
        "timestamp",
        "display_kind",
        "display_metadata",
        "origin_kind",
        "turn_kind",
        "trust_kind",
        "provenance_metadata",
    }
)

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time. All other callers use
    the local wall clock at the point the message enters the live transcript.
    Returning the same mapping keeps the helper convenient at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp, semantically classify, and append one live transcript message."""
    from agent.message_provenance import normalize_message_for_durable_write

    stamp_message_timestamp(message, timestamp=timestamp)
    message.update(normalize_message_for_durable_write(message))
    messages.append(message)
    return message
