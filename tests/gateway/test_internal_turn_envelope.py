from __future__ import annotations

import pytest

from gateway.internal_turn import parse_internal_turn_envelope


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
