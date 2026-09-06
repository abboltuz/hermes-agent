"""Explicit OpenRouter budgets survive request shaping (#41035 / #99725).

Adapted from upstream's regression contract by liuhao1024 and Teknium.
"""

import pytest

from agent.auxiliary_client import _build_call_kwargs


@pytest.mark.parametrize("provider,base_url", [
    ("openrouter", None),
    (" OpenRouter ", None),
    ("custom", "https://openrouter.ai/api/v1"),
    ("openai", "https://api.openrouter.ai/api/v1"),
])
@pytest.mark.parametrize("model,expected_key", [
    ("anthropic/claude-sonnet-4-6", "max_tokens"),
    ("openai/gpt-5.4", "max_completion_tokens"),
])
@pytest.mark.parametrize("cap", [4096, None])
def test_explicit_cap_preserved_in_provider_wire_format(provider, base_url, model, expected_key, cap):
    kwargs = _build_call_kwargs(
        provider=provider, model=model,
        messages=[{"role": "user", "content": "Judge this result"}],
        base_url=base_url, max_tokens=cap,
    )
    token_fields = {key: kwargs[key] for key in ("max_tokens", "max_completion_tokens") if key in kwargs}
    assert token_fields == ({expected_key: cap} if cap is not None else {})


@pytest.mark.parametrize("base_url", [
    "https://openrouter.ai.unrelated.example/v1",
    "https://unrelated.example/openrouter.ai/v1",
    "https://api.openai.com/v1",
])
def test_unrelated_endpoint_retains_existing_omission_policy(base_url):
    kwargs = _build_call_kwargs(
        provider="custom", model="test-model", messages=[],
        base_url=base_url, max_tokens=4096,
    )
    assert "max_tokens" not in kwargs
    assert "max_completion_tokens" not in kwargs
