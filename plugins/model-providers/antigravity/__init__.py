"""Google Antigravity subscription provider profile."""
from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile
from agent.antigravity_bridge_client import (
    BRIDGE_MARKER_BASE_URL, CURATED_FALLBACK_MODELS, filter_antigravity_models,
)


class AntigravityProfile(ProviderProfile):
    def fetch_models(self, *, api_key: str | None = None,
                     base_url: str | None = None, timeout: float = 8.0) -> list[str] | None:
        del api_key, base_url, timeout
        try:
            from agent.antigravity_bridge_client import AntigravityBridgeClient
            client = AntigravityBridgeClient()
            try:
                return filter_antigravity_models(client.list_models())
            finally:
                client.close()
        except Exception:
            return None


antigravity = AntigravityProfile(
    name="antigravity",
    aliases=("google-antigravity",),
    display_name="Google Antigravity",
    description="Google Antigravity subscription via a managed local bridge",
    api_mode="chat_completions",
    env_vars=(),
    base_url=BRIDGE_MARKER_BASE_URL,
    auth_type="external_process",
    supports_health_check=False,
    fallback_models=CURATED_FALLBACK_MODELS,
)
register_provider(antigravity)
