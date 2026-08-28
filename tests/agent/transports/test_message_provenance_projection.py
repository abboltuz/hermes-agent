"""Every provider transport must remove Hermes provenance sidecars."""

from __future__ import annotations

from typing import Any

import pytest

from agent.message_provenance import SEMANTIC_FIELDS
from agent.transports import get_transport


FORBIDDEN = set(SEMANTIC_FIELDS) | {"display_kind", "display_metadata"}


def _assert_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        assert FORBIDDEN.isdisjoint(value)
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def _message():
    return {
        "role": "user",
        "content": "hello",
        "display_kind": "legacy_unknown",
        "display_metadata": {"ui": True},
        "origin_kind": "human_user",
        "turn_kind": "prompt",
        "trust_kind": "user_authorized",
        "provenance_metadata": {"producer": "test"},
    }


@pytest.mark.parametrize(
    ("module", "mode", "model", "extra"),
    [
        ("agent.transports.chat_completions", "chat_completions", "gpt-4o", {}),
        ("agent.transports.anthropic", "anthropic_messages", "claude-sonnet-4-5", {}),
        (
            "agent.transports.bedrock",
            "bedrock_converse",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            {},
        ),
        (
            "agent.transports.codex",
            "codex_responses",
            "gpt-5.6-sol",
            {"is_codex_backend": True},
        ),
    ],
)
def test_provider_kwargs_never_contain_provenance(module, mode, model, extra):
    __import__(module)
    transport = get_transport(mode)
    source = _message()

    kwargs = transport.build_kwargs(model=model, messages=[source], tools=[], **extra)

    _assert_no_forbidden_keys(kwargs)
    assert source["origin_kind"] == "human_user"
    assert source["provenance_metadata"] == {"producer": "test"}
