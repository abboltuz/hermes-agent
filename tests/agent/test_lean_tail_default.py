"""Behavior contracts for bounded lean-tail retention."""

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    LEAN_TAIL_CAP_TOKENS,
    LEAN_TAIL_FLOOR_TOKENS,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _compressor(context_length=1_000_000, **kwargs):
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=context_length,
    ):
        compressor = ContextCompressor(
            "test/model",
            threshold_percent=0.85,
            quiet_mode=True,
            **kwargs,
        )
        _ = compressor.context_length
    return compressor


def test_default_config_and_compressor_select_lean_tail():
    compressor = _compressor()

    assert DEFAULT_CONFIG["compression"]["tail_mode"] == "lean"
    assert compressor.tail_mode == "lean"
    assert compressor.tail_token_budget == LEAN_TAIL_CAP_TOKENS
    assert LEAN_TAIL_FLOOR_TOKENS <= compressor.tail_token_budget


def test_model_switch_recomputes_tail_through_active_mode():
    compressor = _compressor()

    compressor.update_model("test/smaller", context_length=400_000)

    expected = max(
        LEAN_TAIL_FLOOR_TOKENS,
        min(LEAN_TAIL_CAP_TOKENS, int(400_000 * 0.025)),
    )
    assert compressor.tail_mode == "lean"
    assert compressor.tail_token_budget == expected


def test_explicit_legacy_mode_keeps_threshold_proportional_tail():
    compressor = _compressor(tail_mode="legacy")

    assert compressor.tail_mode == "legacy"
    assert compressor.tail_token_budget == int(
        compressor.threshold_tokens * compressor.summary_target_ratio
    )
