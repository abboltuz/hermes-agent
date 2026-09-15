"""OpenCode Free provider profile.

OpenCode's free model tier on the Zen relay (https://opencode.ai/zen/v1).
KEYLESS: the relay serves free-tier models anonymously and rejects any
Authorization bearer it doesn't recognize with 401 — so this provider
never sends a credential at all (the runtime resolver pins the keyless
placeholder and an empty Authorization header; see
hermes_cli.models.opencode_zen_free_runtime). No OpenCode account needed.
Select via ``hermes model`` or ``/model free``.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# Keyless fingerprint — literals, NOT imported from hermes_cli.models.
# Import order forbids it: hermes_cli.models triggers provider discovery
# while still partially initialized (models.py -> agent.model_metadata ->
# providers -> this plugin), so a top-level `from hermes_cli.models import`
# fails and discovery silently drops the whole provider. Keep these in sync
# with OPENCODE_CLI_USER_AGENT / OPENCODE_CLI_CLIENT_ID in
# hermes_cli.models; tests/agent/test_opencode_free_provider.py pins the
# equality (it imports both modules fully initialized, so no cycle there).
#
# Profile.default_headers is static by nature (a plain dict on the
# dataclass), so it carries the static half of the CLI fingerprint only —
# User-Agent + x-opencode-client. The dynamic request IDs
# (x-opencode-session/project/request) merge per client build via
# opencode_zen_free_headers(), which every free-tier client construction
# flows through (agent_init, agent_runtime_helpers, auxiliary_client).
_CLI_USER_AGENT = "opencode/latest"
_CLI_CLIENT_ID = "cli"
_KEYLESS_HEADERS = {
    "Authorization": "",
    "User-Agent": _CLI_USER_AGENT,
    "x-opencode-client": _CLI_CLIENT_ID,
}


class OpenCodeFreeProfile(ProviderProfile):
    """OpenCode Free — keyless, with Ox Alpha reasoning controls.

    Ox Alpha (x-preview-f-free) is reachable through this provider as well
    as opencode-zen; both share the same wire contract (reasoning_effort
    accepts exactly low/high/max — anything else 400s). The translation
    lives in the zen plugin; resolve it through the registered zen profile's
    module so the two providers can never drift.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            import sys

            from providers import get_provider_profile

            zen_profile = get_provider_profile("opencode-zen")
            zen_module = sys.modules[type(zen_profile).__module__]
            return zen_module._build_ox_alpha_reasoning_extras(reasoning_config, model)
        except Exception:
            return {}, {}


opencode_free = OpenCodeFreeProfile(
    name="opencode-free",
    aliases=("free", "opencode_free"),
    env_vars=(),  # keyless — nothing to configure
    base_url="https://opencode.ai/zen/v1",
    display_name="OpenCode Free",
    description="OpenCode free models — keyless, no account needed",
    default_headers=dict(_KEYLESS_HEADERS),
    # Free-tier wire traffic impersonates the official CLI (see
    # hermes_cli.models.OPENCODE_CLI_USER_AGENT); laguna stays the default
    # aux until big-pickle is re-verified live under the new fingerprint.
    default_aux_model="laguna-s-2.1-free",
)

register_provider(opencode_free)
