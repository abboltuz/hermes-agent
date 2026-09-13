"""Contract tests for the bundled Mistral AI provider profile."""

from __future__ import annotations

import json
import urllib.request
from typing import cast
from unittest.mock import patch

from hermes_cli.auth import is_runtime_provider_routable
from hermes_cli.provider_catalog import provider_catalog_by_slug
from providers import get_provider_profile
from providers.base import ProviderProfile


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_profile_is_first_class_across_runtime_and_desktop_catalog():
    profile = get_provider_profile("mistral-ai")

    assert profile is not None
    assert profile.__class__.__module__ == "plugins.model_providers.mistral"
    assert profile.name == "mistral"
    assert profile.display_name == "Mistral AI"
    assert profile.auth_type == "api_key"
    assert profile.api_mode == "chat_completions"
    assert profile.env_vars == ("MISTRAL_API_KEY", "MISTRAL_BASE_URL")
    assert profile.base_url == "https://api.mistral.ai/v1"
    assert profile.models_url == "https://api.mistral.ai/v1/models"
    assert profile.supports_vision is True
    assert profile.default_aux_model in profile.fallback_models
    assert is_runtime_provider_routable("mistral")

    descriptor = provider_catalog_by_slug()["mistral"]
    assert descriptor.tab == "keys"
    assert descriptor.api_key_env_vars == ("MISTRAL_API_KEY",)
    assert descriptor.base_url_env_var == "MISTRAL_BASE_URL"
    assert descriptor.signup_url == "https://console.mistral.ai/api-keys/"


def test_live_catalog_keeps_only_distinct_agentic_chat_models():
    profile = get_provider_profile("mistral")
    assert profile is not None
    payload = {
        "data": [
            {
                "id": "mistral-medium-latest",
                "archived": False,
                "capabilities": {
                    "completion_chat": True,
                    "function_calling": True,
                },
            },
            {
                "id": "mistral-embed",
                "capabilities": {
                    "completion_chat": False,
                    "function_calling": False,
                },
            },
            {
                "id": "chat-without-tools",
                "capabilities": {
                    "completion_chat": True,
                    "function_calling": False,
                },
            },
            {
                "id": "retired-model",
                "archived": True,
                "capabilities": {
                    "completion_chat": True,
                    "function_calling": True,
                },
            },
            {
                "id": "mistral-medium-latest",
                "capabilities": {
                    "completion_chat": True,
                    "function_calling": True,
                },
            },
        ]
    }
    captured: dict[str, object] = {}

    def opener(request: object, *, timeout: float) -> _Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(payload)

    with patch("hermes_cli.urllib_security.open_credentialed_url", opener):
        models = profile.fetch_models(api_key="test-secret", timeout=2.5)

    assert models == ["mistral-medium-latest"]
    request = cast(urllib.request.Request, captured["request"])
    assert request.full_url == "https://api.mistral.ai/v1/models"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert captured["timeout"] == 2.5


def test_catalog_failure_uses_static_fallbacks():
    profile = get_provider_profile("mistral")
    assert profile is not None

    with patch(
        "hermes_cli.urllib_security.open_credentialed_url",
        side_effect=OSError("offline"),
    ):
        assert profile.fetch_models(api_key="test-secret") is None

    assert profile.fallback_models
    assert all(model.endswith("-latest") for model in profile.fallback_models)


def test_custom_base_url_preserves_generic_openai_catalog_contract():
    profile = get_provider_profile("mistral")
    assert profile is not None

    with patch.object(
        ProviderProfile,
        "fetch_models",
        return_value=["proxy-model"],
    ) as generic_fetch:
        models = profile.fetch_models(
            api_key="proxy-secret",
            base_url="https://gateway.example.test/v1",
            timeout=1.25,
        )

    assert models == ["proxy-model"]
    generic_fetch.assert_called_once_with(
        api_key="proxy-secret",
        base_url="https://gateway.example.test/v1",
        timeout=1.25,
    )
