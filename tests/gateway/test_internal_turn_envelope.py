from __future__ import annotations

import pytest

from gateway.internal_turn import parse_internal_turn_envelope
from gateway.platforms.base import MessageEvent
from gateway.run import _internal_event_display_metadata
from gateway.session import SessionSource
from gateway.config import Platform


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"source": {"nested": "not text"}},
        {"event_id": True},
        {"recovered": "yes"},
    ],
)
def test_internal_turn_envelope_rejects_malformed_typed_fields(invalid_field):
    metadata = {
        "source": "process",
        "internal": True,
        "kind": "process_notification",
        **invalid_field,
    }

    with pytest.raises(ValueError, match="internal turn metadata"):
        parse_internal_turn_envelope(
            {
                "display_kind": "internal_notification",
                "display_metadata": metadata,
            }
        )


def test_internal_turn_envelope_rejects_local_human_authority_claim():
    with pytest.raises(ValueError, match="not allowed for proxy ingress"):
        parse_internal_turn_envelope(
            {
                "provenance": {
                    "origin_kind": "human_user",
                    "turn_kind": "prompt",
                    "trust_kind": "user_authorized",
                }
            }
        )


def test_internal_event_metadata_preserves_all_bounded_audit_identifiers():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1")
    event = MessageEvent(
        text="wake",
        source=source,
        internal=True,
        metadata={
            "source": "kanban",
            "kind": "kanban_wake",
            "event_id": 1,
            "task_id": "task-1",
            "job_id": "job-1",
            "run_id": 2,
            "process_id": "proc-1",
            "delegation_id": "delegation-1",
            "generation_id": 3,
            "session_id": "session-1",
            "event_kind": "completed",
            "recovered": True,
            "ignored": "never durable",
        },
    )

    assert _internal_event_display_metadata(event, source) == {
        "source": "kanban",
        "internal": True,
        "kind": "kanban_wake",
        "platform": "telegram",
        "event_id": 1,
        "task_id": "task-1",
        "job_id": "job-1",
        "run_id": 2,
        "process_id": "proc-1",
        "delegation_id": "delegation-1",
        "generation_id": 3,
        "session_id": "session-1",
        "event_kind": "completed",
        "recovered": True,
    }
