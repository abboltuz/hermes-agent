"""Mistral AI provider profile.

Mistral exposes an OpenAI-compatible Chat Completions endpoint, but its model
catalog also includes non-agentic products such as embeddings, OCR, moderation,
and audio models.  Keep those out of Hermes' model picker by requiring the
catalog capabilities needed by the agent runtime.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, cast

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_MODELS_URL = f"{MISTRAL_BASE_URL}/models"


def _agentic_model_ids(payload: Any) -> list[str]:
    """Return distinct, non-archived chat models with function calling."""

    items = (
        payload
        if isinstance(payload, list)
        else payload.get("data", [])
        if isinstance(payload, dict)
        else []
    )
    if not isinstance(items, list):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("archived") is True:
            continue
        capabilities = item.get("capabilities")
        if not isinstance(capabilities, dict):
            continue
        if capabilities.get("completion_chat") is not True:
            continue
        if capabilities.get("function_calling") is not True:
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        if model_id and model_id not in seen:
            seen.add(model_id)
            result.append(model_id)
    return result


class MistralProfile(ProviderProfile):
    """Mistral API profile with capability-aware live model discovery."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        caller_base = (base_url or "").strip().rstrip("/")
        if caller_base and caller_base != self.base_url.rstrip("/"):
            # A MISTRAL_BASE_URL override can point at a generic OpenAI-style
            # gateway whose catalog does not expose Mistral capabilities.
            return cast(
                list[str] | None,
                super().fetch_models(
                    api_key=api_key,
                    base_url=caller_base,
                    timeout=timeout,
                ),
            )

        req = urllib.request.Request(self.models_url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())

        from hermes_cli.urllib_security import open_credentialed_url

        try:
            with open_credentialed_url(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _agentic_model_ids(payload)
        except Exception as exc:
            logger.debug("fetch_models(mistral): %s", exc)
            return None


mistral = MistralProfile(
    name="mistral",
    aliases=("mistral-ai", "la-plateforme"),
    display_name="Mistral AI",
    description="Mistral AI — official API using workspace-scoped API keys",
    signup_url="https://console.mistral.ai/api-keys/",
    env_vars=("MISTRAL_API_KEY", "MISTRAL_BASE_URL"),
    base_url=MISTRAL_BASE_URL,
    models_url=MISTRAL_MODELS_URL,
    auth_type="api_key",
    api_mode="chat_completions",
    supports_vision=True,
    default_aux_model="mistral-small-latest",
    fallback_models=(
        "mistral-medium-latest",
        "mistral-large-latest",
        "mistral-small-latest",
        "devstral-latest",
        "devstral-small-latest",
        "codestral-latest",
    ),
)

register_provider(mistral)
