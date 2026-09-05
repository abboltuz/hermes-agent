"""Behavior tests for hermes_cli.inventory.

Locks the invariants the three migrated consumers (web_server.py
/api/model/options, tui_gateway model.options, tui_gateway model.save_key)
depend on:

- load_picker_context() reproduces the inline 17-LOC config-slice exactly.
- with_overrides() is truthy-only (empty agent attrs must not clobber).
- build_models_payload() returns a stable {providers, model, provider}
  shape and delegates curation to list_authenticated_providers (does not
  call provider_model_ids per row).
- canonical_order keys on slug membership, not is_user_defined — section
  3 of list_authenticated_providers sets is_user_defined=True for
  canonical slugs in the providers: dict, and that flag must NOT demote
  them to the tail.
- picker_hints adds authenticated/auth_type/key_env/warning per row,
  matching the TUI ModelPickerDialog shape.
"""

from __future__ import annotations

from unittest.mock import patch


from hermes_cli.inventory import (
    ConfigContext,
    build_models_payload,
    load_picker_context,
)


# ─── load_picker_context ───────────────────────────────────────────────


def _cfg(model=None, providers=None, custom_providers=None) -> dict:
    return {
        "model": model if model is not None else {},
        "providers": providers if providers is not None else {},
        "custom_providers": custom_providers if custom_providers is not None else [],
    }






# ─── with_overrides ────────────────────────────────────────────────────


def _empty_ctx(provider="orig", model="orig-model", base_url="orig-url"):
    return ConfigContext(
        current_provider=provider,
        current_model=model,
        current_base_url=base_url,
        user_providers={},
        custom_providers=[],
    )






# ─── build_models_payload ──────────────────────────────────────────────


def _list_auth_returning(rows: list[dict]):
    """Patch list_authenticated_providers to return a fixed row list."""
    return patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=rows,
    )


def _nous_row(model: str = "openai/gpt-5.5") -> dict:
    return {
        "slug": "nous",
        "name": "Nous",
        "models": [model],
        "total_models": 1,
        "is_current": True,
        "is_user_defined": False,
        "source": "built-in",
    }




def test_cli_model_picker_forwards_force_refresh_to_probe_flags():
    """CLI /model picker must pass force_refresh to probe flags (#65652, #65650).

    Normal open (/model bare) skips non-current probes; /model --refresh probes
    all custom providers to freshen their model lists.
    """
    ctx = _empty_ctx()

    # Normal open — skip non-current probes
    force_refresh = False
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=[],
    ) as mock_list:
        build_models_payload(
            ctx,
            probe_custom_providers=force_refresh,
            probe_current_custom_provider=not force_refresh,
        )
    assert mock_list.call_args.kwargs["probe_custom_providers"] is False
    assert mock_list.call_args.kwargs["probe_current_custom_provider"] is True

    # Refresh open — probe everything
    force_refresh = True
    with patch(
        "hermes_cli.model_switch.list_authenticated_providers",
        return_value=[],
    ) as mock_list:
        build_models_payload(
            ctx,
            probe_custom_providers=force_refresh,
            probe_current_custom_provider=not force_refresh,
        )
    assert mock_list.call_args.kwargs["probe_custom_providers"] is True
    assert mock_list.call_args.kwargs["probe_current_custom_provider"] is False


def test_list_authenticated_providers_force_fresh_is_keyword_only():
    """``force_fresh_nous_tier`` must be keyword-only on the public listing API.

    It was inserted between ``custom_providers`` and ``max_models``; making it
    keyword-only ensures no positional caller passing ``max_models`` as the 5th
    arg silently mis-binds it to the tier-refresh flag. Pin the contract so a
    future signature edit that drops the ``*`` separator is caught.
    """
    import inspect

    from hermes_cli.model_switch import list_authenticated_providers

    sig = inspect.signature(list_authenticated_providers)
    param = sig.parameters["force_fresh_nous_tier"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False




def test_include_unconfigured_appends_canonical_skeletons():
    """include_unconfigured=True adds CANONICAL_PROVIDERS rows that
    list_authenticated_providers didn't emit. Skeleton rows have empty
    models and source='canonical'."""
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx(provider="openrouter")
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, include_unconfigured=True)
    # All canonical providers other than openrouter should appear as
    # skeleton rows.
    from hermes_cli.models import CANONICAL_PROVIDERS

    seen_slugs = {r["slug"] for r in payload["providers"]}
    for entry in CANONICAL_PROVIDERS:
        assert entry.slug in seen_slugs, f"missing {entry.slug}"
    # Skeletons have empty models and source='canonical'.
    skeletons = [r for r in payload["providers"]
                 if r.get("source") == "canonical"]
    assert all(r["models"] == [] for r in skeletons)
    assert all(r["total_models"] == 0 for r in skeletons)


def test_explicit_only_filters_ambient_credentials_but_keeps_current_and_custom_rows():
    rows = [
        {"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.4"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "gemini", "name": "Gemini", "models": ["gemini-2.5-pro"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "built-in"},
        {"slug": "copilot", "name": "Copilot", "models": ["gpt-5.4"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "nous", "name": "Nous", "models": ["anthropic/claude-sonnet-5"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "custom:lab", "name": "Lab", "models": ["lab-1"],
         "total_models": 1, "is_current": False, "is_user_defined": True,
         "source": "user-config"},
        {"slug": "moa", "name": "MoA", "models": ["default"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "virtual"},
    ]
    ctx = _empty_ctx(provider="openai-codex", model="gpt-5.4")
    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.read_raw_config", return_value={}),
        patch(
            "hermes_cli.auth.is_provider_explicitly_configured",
            side_effect=lambda slug: slug == "gemini",
        ),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    assert [row["slug"] for row in payload["providers"]] == [
        "openai-codex",
        "gemini",
        "custom:lab",
    ]


def test_explicit_only_keeps_anthropic_row_with_oauth_credentials():
    """Anthropic OAuth logins are deliberate sign-ins, not ambient credentials.

    Claude Code (~/.claude/.credentials.json) and Hermes' own device flow
    leave no trace in active_provider / model.provider / API-key env vars,
    so is_provider_explicitly_configured() returns False even though
    list_authenticated_providers just accepted those same credentials when
    building the row. The desktop explicit-only filter must keep it.
    """
    rows = [
        {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-5"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
        {"slug": "copilot", "name": "Copilot", "models": ["gpt-5.4"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
    ]
    ctx = _empty_ctx(provider="opencode-go", model="glm-5.3")
    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.read_raw_config", return_value={}),
        patch(
            "hermes_cli.auth.is_provider_explicitly_configured",
            return_value=False,
        ),
        patch(
            "hermes_cli.inventory._anthropic_oauth_credentials_present",
            return_value=True,
        ),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    slugs = [row["slug"] for row in payload["providers"]]
    assert "anthropic" in slugs, (
        "Anthropic OAuth login must survive the explicit-only filter"
    )
    assert "copilot" not in slugs, (
        "ambient credential discovery must stay filtered"
    )


def test_explicit_only_drops_anthropic_row_without_oauth_credentials():
    """No OAuth token and no explicit config -> Anthropic stays hidden."""
    rows = [
        {"slug": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-5"],
         "total_models": 1, "is_current": False, "is_user_defined": False,
         "source": "hermes"},
    ]
    ctx = _empty_ctx(provider="opencode-go", model="glm-5.3")
    with (
        _list_auth_returning(rows),
        patch("hermes_cli.config.read_raw_config", return_value={}),
        patch(
            "hermes_cli.auth.is_provider_explicitly_configured",
            return_value=False,
        ),
        patch(
            "hermes_cli.inventory._anthropic_oauth_credentials_present",
            return_value=False,
        ),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    assert "anthropic" not in [row["slug"] for row in payload["providers"]]


def test_anthropic_oauth_presence_accepts_pool_only_oauth_entry():
    """A pool-only OAuth entry (auth.json credential_pool.anthropic) counts.

    Wired/device-flow tokens land in the credential pool, not in
    .anthropic_oauth.json or ~/.claude/.credentials.json. The presence
    check must accept them or the row is built and then silently dropped.
    """
    from hermes_cli.inventory import _anthropic_oauth_credentials_present

    with (
        patch(
            "agent.anthropic_adapter.read_hermes_oauth_credentials",
            return_value=None,
        ),
        patch(
            "agent.anthropic_adapter.read_claude_code_credentials",
            return_value=None,
        ),
        patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                {"auth_type": "oauth", "access_token": "sk-ant-oat01-pool"}
            ],
        ),
    ):
        assert _anthropic_oauth_credentials_present() is True

    # api_key pool entries are NOT OAuth logins — presence must stay False
    # (they are handled by the explicit-config gate / env var paths).
    with (
        patch(
            "agent.anthropic_adapter.read_hermes_oauth_credentials",
            return_value=None,
        ),
        patch(
            "agent.anthropic_adapter.read_claude_code_credentials",
            return_value=None,
        ),
        patch(
            "hermes_cli.auth.read_credential_pool",
            return_value=[
                {"auth_type": "api_key", "access_token": "sk-ant-api03-key"}
            ],
        ),
    ):
        assert _anthropic_oauth_credentials_present() is False



# ─── picker_hints ──────────────────────────────────────────────────────


def test_picker_hints_marks_authed_rows_authenticated():
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, picker_hints=True)
    assert payload["providers"][0]["authenticated"] is True


def test_picker_hints_api_key_warning_format():
    """For api_key providers with a defined env var, the warning must
    point to that env var."""
    rows = []
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(
            ctx, include_unconfigured=True, picker_hints=True,
        )
    # anthropic uses api_key + ANTHROPIC_API_KEY.
    anthropic = next(
        r for r in payload["providers"] if r["slug"] == "anthropic"
    )
    assert "ANTHROPIC_API_KEY" in anthropic["warning"]
    assert anthropic["warning"].startswith("paste ")


# ─── canonical_order ───────────────────────────────────────────────────


def test_canonical_order_uses_slug_not_is_user_defined_flag():
    """Section 3 of list_authenticated_providers sets is_user_defined=True
    for canonical slugs that appear in the providers: config dict.
    canonical_order MUST key on slug membership, not the flag — otherwise
    canonical providers configured via the keyed schema get demoted to
    the tail.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    canonical_slug = CANONICAL_PROVIDERS[2].slug  # any canonical
    rows = [
        # A truly-custom row (correct: is_user_defined=True)
        {"slug": "custom:Ollama", "name": "Ollama", "models": [],
         "total_models": 0, "is_current": False, "is_user_defined": True,
         "source": "user-config"},
        # A canonical row that the substrate flagged as user-defined
        # because the user configured it via providers: dict.
        {"slug": canonical_slug, "name": "x", "models": ["m1"],
         "total_models": 1, "is_current": False, "is_user_defined": True,
         "source": "built-in"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, canonical_order=True)
    slugs = [r["slug"] for r in payload["providers"]]
    # Canonical-slug row must come BEFORE truly-custom rows, regardless
    # of is_user_defined.
    canonical_idx = slugs.index(canonical_slug)
    custom_idx = slugs.index("custom:Ollama")
    assert canonical_idx < custom_idx, (
        f"canonical {canonical_slug} demoted to tail "
        f"(canonical_idx={canonical_idx} > custom_idx={custom_idx})"
    )




# ─── Integration: end-to-end through real load_picker_context ──────────


def test_end_to_end_with_real_context_no_credentials_leak(monkeypatch):
    """Full pipeline: real load_picker_context + real
    list_authenticated_providers. Verify no credential string ever
    appears in the returned payload, even with picker_hints=True."""
    canary = "sk-canary-XYZ-must-not-appear"
    monkeypatch.setenv("OPENROUTER_API_KEY", canary)
    monkeypatch.setenv("ANTHROPIC_API_KEY", canary)
    cfg = _cfg(model={"provider": "openrouter"})
    with patch("hermes_cli.config.load_config", return_value=cfg):
        ctx = load_picker_context()
    payload = build_models_payload(
        ctx, include_unconfigured=True, picker_hints=True,
    )
    import json as _json

    assert canary not in _json.dumps(payload)


def test_payload_shape_compatible_with_modelpickerdialog_frontend():
    """Frontend (web/src/components/ModelPickerDialog.tsx) reads:
    name, slug, models, total_models, is_current, warning, authenticated.
    Verify every authenticated/skeleton row exposes those keys.
    """
    rows = [
        {"slug": "openrouter", "name": "OpenRouter", "models": ["m1"],
         "total_models": 1, "is_current": True, "is_user_defined": False,
         "source": "built-in"},
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(
            ctx, include_unconfigured=True, picker_hints=True,
        )
    required_keys = {"name", "slug", "models", "total_models", "is_current",
                     "authenticated"}
    for row in payload["providers"]:
        missing = required_keys - row.keys()
        assert not missing, f"row {row['slug']} missing keys: {missing}"


# ─── Aggregator dedup (issue #45954) ───────────────────────────────────


def _user_provider_row(slug: str, models: list[str]) -> dict:
    return {
        "slug": slug,
        "name": slug.title(),
        "models": models,
        "total_models": len(models),
        "is_current": False,
        "is_user_defined": True,
        "source": "user-config",
    }


def _aggregator_row(slug: str, models: list[str]) -> dict:
    return {
        "slug": slug,
        "name": slug.title(),
        "models": models,
        "total_models": len(models),
        "is_current": False,
        "is_user_defined": False,
        "source": "built-in",
    }


def test_user_defined_rows_carry_alias_set_for_gui_current_match():
    """Custom provider rows must expose `aliases` so the desktop picker can
    match a session's canonical `custom:<key>` identity against the row's
    bare-key slug (#87035). Built-in rows carry no aliases.
    """
    rows = [
        {
            "slug": "myep",
            "name": "My Endpoint",
            "models": ["my-model"],
            "total_models": 1,
            "is_current": True,
            "is_user_defined": True,
            "source": "user-config",
            "api_url": "http://localhost:8000/v1",
        },
        _nous_row() | {"is_current": False},
    ]
    ctx = _empty_ctx(provider="custom:myep", model="my-model")

    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    by_slug = {r["slug"]: r for r in payload["providers"]}
    aliases = by_slug["myep"]["aliases"]
    # The canonical session identity must be matchable via the alias set.
    assert "custom:myep" in aliases
    assert "myep" in aliases
    assert "custom:my-endpoint" in aliases
    assert "aliases" not in by_slug["nous"]


def test_aggregator_dedup_removes_overlapping_models():
    """Models served by a user-defined provider are removed from
    aggregator rows so the picker doesn't show them under the wrong
    provider.  (#45954)"""
    rows = [
        _user_provider_row("litellm-proxy", [
            "nvidia/nim/minimax-m3",
            "nvidia/nim/kimi-k2.6",
        ]),
        _aggregator_row("openrouter", [
            "minimax/minimax-m3",
            "nvidia/nim/minimax-m3",  # overlaps with litellm-proxy
            "anthropic/claude-sonnet-4.6",
        ]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")
    proxy_row = next(r for r in payload["providers"] if r["slug"] == "litellm-proxy")

    # User-defined provider keeps all its models
    assert proxy_row["models"] == ["nvidia/nim/minimax-m3", "nvidia/nim/kimi-k2.6"]

    # Aggregator lost the overlapping model but kept the rest
    assert "nvidia/nim/minimax-m3" not in or_row["models"]
    assert "minimax/minimax-m3" in or_row["models"]
    assert "anthropic/claude-sonnet-4.6" in or_row["models"]
    assert or_row["total_models"] == 2




def test_flat_namespace_reseller_keeps_first_party_models_overlapping_user_proxy():
    """opencode-go / opencode-zen are flagged ``is_aggregator=True`` (their
    flat ``/v1/models`` returns bare IDs the model-switch resolver searches),
    but they are NOT routing aggregators — every model they list is a
    first-party model under the user's subscription. When a user also runs a
    custom proxy that happens to serve a same-named model, the picker dedup
    must NOT strip the reseller's own catalog. Regression for #47077, where
    opencode-go showed only 13 of 19 models because minimax-m3/m2.7/m2.5,
    glm-5/5.1, and deepseek-v4-flash were deduped against an overlapping
    custom provider.
    """
    rows = [
        _user_provider_row("custom:my-proxy", [
            "minimax-m3", "minimax-m2.7", "glm-5", "deepseek-v4-flash",
        ]),
        _aggregator_row("opencode-go", [
            "kimi-k2.6", "minimax-m3", "minimax-m2.7", "glm-5",
            "deepseek-v4-flash", "qwen3.7-max",
        ]),
        _aggregator_row("openrouter", ["minimax-m3", "anthropic/claude-sonnet-4.6"]),
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        payload = build_models_payload(ctx)

    go_row = next(r for r in payload["providers"] if r["slug"] == "opencode-go")
    or_row = next(r for r in payload["providers"] if r["slug"] == "openrouter")

    # The reseller keeps ALL of its first-party models — nothing stripped.
    assert go_row["models"] == [
        "kimi-k2.6", "minimax-m3", "minimax-m2.7", "glm-5",
        "deepseek-v4-flash", "qwen3.7-max",
    ]
    assert go_row["total_models"] == 6

    # A TRUE routing aggregator is still deduped against the user's models.
    assert "minimax-m3" not in or_row["models"]
    assert "anthropic/claude-sonnet-4.6" in or_row["models"]




def test_build_models_payload_no_max_models_returns_full_list():
    """When max_models is not passed (None), build_models_payload must
    return the full model list — not truncate to the old default of 50.
    Regression for #48279: Kilo Gateway picker was capped at 50 of 336
    models, making most models undiscoverable via search."""
    full_models = [f"model-{i}" for i in range(100)]
    rows = [
        {
            "slug": "kilocode",
            "name": "Kilo Code",
            "models": full_models,
            "total_models": len(full_models),
            "is_current": False,
            "is_user_defined": False,
            "source": "built-in",
        },
    ]
    ctx = _empty_ctx()
    with _list_auth_returning(rows):
        # No max_models argument — should return all 100 models
        payload = build_models_payload(ctx)

    kilo_row = next(r for r in payload["providers"] if r["slug"] == "kilocode")
    assert kilo_row["models"] == full_models
    assert kilo_row["total_models"] == 100
    assert len(kilo_row["models"]) == 100


# ─── refresh flag (cache-bust) ─────────────────────────────────────────


def test_build_models_payload_forwards_refresh_flag():
    """build_models_payload must forward refresh= to list_authenticated_providers.

    The desktop picker's "Refresh Models" control passes refresh=True; the
    flag has to reach list_authenticated_providers so the per-provider
    model-id cache gets busted. Default opens pass refresh=False.
    """
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured["refresh"] = kwargs.get("refresh")
        return []

    with patch("hermes_cli.model_switch.list_authenticated_providers", side_effect=_capture):
        build_models_payload(_empty_ctx())
    assert captured["refresh"] is False

    with patch("hermes_cli.model_switch.list_authenticated_providers", side_effect=_capture):
        build_models_payload(_empty_ctx(), refresh=True)
    assert captured["refresh"] is True


def test_list_authenticated_providers_refresh_busts_cache():
    """refresh=True clears the provider-model disk cache exactly once;
    refresh=False leaves it untouched (so normal picker opens stay snappy)."""
    from hermes_cli import model_switch

    with patch("hermes_cli.models.clear_provider_models_cache") as clear:
        model_switch.list_authenticated_providers(refresh=False)
        assert clear.call_count == 0
        model_switch.list_authenticated_providers(refresh=True)
        assert clear.call_count == 1


# ─── Antigravity managed-provider inventory ─────────────────────────────


def test_antigravity_connected_managed_account_appears_without_config_or_current_provider():
    """A connected managed account is explicit picker authentication by itself."""
    class ManagedClient:
        instances = []

        def __init__(self):
            self.calls = []
            self.closed = False
            self.instances.append(self)

        def list_accounts(self):
            self.calls.append("accounts")
            return {"connected": True}

        def list_models(self):
            self.calls.append("models")
            return [
                {"id": "antigravity-gemini-3-pro"},
                {"id": "bridge-secret-token"},
            ]

        def close(self):
            self.closed = True

    ctx = _empty_ctx(provider="openrouter")
    with (
        _list_auth_returning([]),
        patch("agent.antigravity_bridge_client.AntigravityBridgeClient", ManagedClient),
        patch("hermes_cli.models.cached_provider_model_ids", side_effect=AssertionError("duplicate catalog probe")),
    ):
        payload = build_models_payload(ctx, explicit_only=True, picker_hints=True)

    row = next(row for row in payload["providers"] if row["slug"] == "antigravity")
    assert row["authenticated"] is True
    assert row["models"] == ["antigravity-gemini-3-pro"]
    assert ManagedClient.instances[0].calls == ["accounts", "models"]
    assert ManagedClient.instances[0].closed is True
    assert "bridge" not in str(row).lower()
    assert "secret" not in str(row).lower()


def test_antigravity_account_probe_failure_or_disconnected_snapshot_does_not_add_authenticated_row():
    """Unverified account state never turns the managed skeleton into an auth row."""
    class DisconnectedClient:
        def list_accounts(self):
            return {"connected": False}

        def close(self):
            pass

    ctx = _empty_ctx(provider="openrouter")
    with (
        _list_auth_returning([]),
        patch("agent.antigravity_bridge_client.AntigravityBridgeClient", DisconnectedClient),
    ):
        disconnected = build_models_payload(ctx, explicit_only=True, picker_hints=True)

    with (
        _list_auth_returning([]),
        patch("agent.antigravity_bridge_client.AntigravityBridgeClient", side_effect=RuntimeError("private account diagnostic")),
    ):
        failed = build_models_payload(ctx, explicit_only=True, picker_hints=True)

    assert "antigravity" not in {row["slug"] for row in disconnected["providers"]}
    assert "antigravity" not in {row["slug"] for row in failed["providers"]}
    assert "private" not in str(failed).lower()


def test_antigravity_truthy_non_boolean_connected_snapshot_does_not_authorize_managed_row():
    """Only a literal boolean connection result authorizes managed discovery."""
    class TruthyConnectedClient:
        instances = []

        def __init__(self):
            self.closed = False
            self.list_models_called = False
            self.instances.append(self)

        def list_accounts(self):
            return {"connected": "yes"}

        def list_models(self):
            self.list_models_called = True
            raise AssertionError("model discovery must require literal True")

        def close(self):
            self.closed = True

    ctx = _empty_ctx(provider="openrouter")
    with (
        _list_auth_returning([]),
        patch("agent.antigravity_bridge_client.AntigravityBridgeClient", TruthyConnectedClient),
    ):
        payload = build_models_payload(ctx, explicit_only=True, picker_hints=True)

    assert "antigravity" not in {row["slug"] for row in payload["providers"]}
    assert TruthyConnectedClient.instances[0].list_models_called is False
    assert TruthyConnectedClient.instances[0].closed is True


def test_antigravity_current_provider_uses_filtered_live_catalog_without_api_key():
    """A configured Antigravity subscription is a native picker row.

    Its managed bridge is not an API-key endpoint, so the row must not depend
    on a dummy credential. The inventory still filters a bridge result before
    returning it to web and TUI consumers.
    """
    from agent.antigravity_bridge_client import filter_antigravity_models

    live = [
        "antigravity-gemini-3-pro",
        "antigravity-claude-sonnet-4-6",
        "embedding-001",
        "bridge-secret-token",
    ]
    expected = filter_antigravity_models([{"id": model_id} for model_id in live]) or []
    ctx = _empty_ctx(provider="antigravity", model="antigravity-gemini-3-pro")

    with (
        _list_auth_returning([]),
        patch("hermes_cli.models.cached_provider_model_ids", return_value=live),
    ):
        payload = build_models_payload(ctx, explicit_only=True, picker_hints=True)

    row = next(row for row in payload["providers"] if row["slug"] == "antigravity")
    assert row["name"] == "Google Antigravity"
    assert row["models"] == expected
    assert row["total_models"] == len(expected)
    assert row["is_current"] is True
    assert row["authenticated"] is True
    assert "key_env" not in row
    assert "warning" not in row
    assert "bridge" not in str(row).lower()
    assert "secret" not in str(row).lower()


def test_antigravity_picker_keeps_current_namespaced_gpt_oss_catalog_row():
    """A public GPT-OSS bridge row survives the real picker safety pipeline."""
    current_gpt_oss = "antigravity-gpt-oss-120b"
    ctx = _empty_ctx(provider="antigravity", model=current_gpt_oss)

    with (
        _list_auth_returning([]),
        patch("hermes_cli.models.cached_provider_model_ids", return_value=[current_gpt_oss]),
    ):
        payload = build_models_payload(ctx, explicit_only=True, picker_hints=True)

    row = next(row for row in payload["providers"] if row["slug"] == "antigravity")
    assert row["models"] == [current_gpt_oss]
    assert row["total_models"] == 1


def test_antigravity_configured_provider_preserves_fallback_and_unconfigured_semantics():
    """Configured Antigravity gets its curated fallback; a setup skeleton does not."""
    from agent.antigravity_bridge_client import CURATED_FALLBACK_MODELS

    configured = ConfigContext(
        current_provider="",
        current_model="",
        current_base_url="",
        user_providers={"antigravity": {}},
        custom_providers=[],
    )
    with (
        _list_auth_returning([]),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=list(CURATED_FALLBACK_MODELS),
        ),
    ):
        configured_payload = build_models_payload(
            configured, explicit_only=True, picker_hints=True
        )
    configured_row = next(
        row for row in configured_payload["providers"] if row["slug"] == "antigravity"
    )
    assert configured_row["models"] == sorted(CURATED_FALLBACK_MODELS)
    assert configured_row["authenticated"] is True

    unconfigured = _empty_ctx(provider="openrouter")
    with _list_auth_returning([]):
        unconfigured_payload = build_models_payload(
            unconfigured, include_unconfigured=True, picker_hints=True
        )
    skeleton = next(
        row for row in unconfigured_payload["providers"] if row["slug"] == "antigravity"
    )
    assert skeleton["models"] == []
    assert skeleton["authenticated"] is False
    assert skeleton["auth_type"] == "external_process"
    assert skeleton["key_env"] == ""
    assert "bridge" not in str(skeleton).lower()


def test_antigravity_disabled_config_is_hidden_unless_it_is_current():
    """A disabled managed provider follows ordinary picker enable semantics."""
    disabled = ConfigContext(
        current_provider="openrouter",
        current_model="",
        current_base_url="",
        user_providers={"antigravity": {"enabled": False}},
        custom_providers=[],
    )
    current = ConfigContext(
        current_provider="antigravity",
        current_model="antigravity-gemini-3-pro",
        current_base_url="",
        user_providers={"antigravity": {"enabled": False}},
        custom_providers=[],
    )

    with _list_auth_returning([]):
        disabled_payload = build_models_payload(disabled, explicit_only=True)
        current_payload = build_models_payload(current, explicit_only=True)

    disabled_slugs = {row["slug"] for row in disabled_payload["providers"]}
    current_row = next(
        row for row in current_payload["providers"] if row["slug"] == "antigravity"
    )
    assert "antigravity" not in disabled_slugs
    assert current_row["is_current"] is True


def test_antigravity_configured_row_honors_excluded_providers():
    """Catalog exclusions apply equally to managed provider row injection."""
    ctx = ConfigContext(
        current_provider="nous",
        current_model="",
        current_base_url="",
        user_providers={"antigravity": {}},
        custom_providers=[],
        excluded_providers=[" ANTIGRAVITY "],
    )
    rows = [_nous_row()]

    with _list_auth_returning(rows):
        payload = build_models_payload(ctx, explicit_only=True)

    assert {row["slug"] for row in payload["providers"]} == {"nous"}


def test_antigravity_exclusion_suppresses_current_and_unconfigured_rows():
    """An excluded provider never reappears during fallback-row post-processing."""
    current = ConfigContext(
        current_provider="antigravity",
        current_model="antigravity-gemini-3-pro",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=[" ANTIGRAVITY "],
    )
    unconfigured = ConfigContext(
        current_provider="nous",
        current_model="",
        current_base_url="",
        user_providers={},
        custom_providers=[],
        excluded_providers=["antigravity"],
    )

    with _list_auth_returning([]):
        current_payload = build_models_payload(current, explicit_only=True)
        unconfigured_payload = build_models_payload(unconfigured, include_unconfigured=True)

    assert "antigravity" not in {row["slug"] for row in current_payload["providers"]}
    assert "antigravity" not in {row["slug"] for row in unconfigured_payload["providers"]}


def test_antigravity_picker_drops_malformed_bridge_model_ids():
    """Picker payloads retain only safe Gemini/Claude model identifiers."""
    live = [
        "antigravity-gemini-3-pro",
        "claude-sonnet-4-6",
        "claude-bridge-marker-INVENTORY_PROBE",
        "gemini-sdkbridge://antigravity?credential=INVENTORY_PROBE",
        "gemini-3-pro\ncredential=INVENTORY_PROBE",
        "gemini-api-key-sk-live-inventoryprobe",
        "gemini-password-inventoryprobe",
        "gemini-bearer-inventoryprobe",
        "claude-sessionid-inventoryprobe",
    ]
    ctx = _empty_ctx(provider="antigravity", model="antigravity-gemini-3-pro")

    with (
        _list_auth_returning([]),
        patch("hermes_cli.models.cached_provider_model_ids", return_value=live),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    row = next(row for row in payload["providers"] if row["slug"] == "antigravity")
    assert row["models"] == ["antigravity-gemini-3-pro", "claude-sonnet-4-6"]
    assert "bridge" not in str(row).lower()
    assert "credential" not in str(row).lower()


def test_antigravity_picker_rejects_oversized_or_numeric_suffix_model_ids():
    """The bridge cannot grow picker payloads with unbounded numeric model IDs."""
    live = [
        "antigravity-gemini-3-pro",
        "claude-sonnet-4-6",
        "gemini-3-pro-99",
        "gemini-999999999999999999999999999999999999999999999999-pro",
        "gemini-3-pro" + "-99" * 1_001,
    ]
    ctx = _empty_ctx(provider="antigravity", model="antigravity-gemini-3-pro")

    with (
        _list_auth_returning([]),
        patch("hermes_cli.models.cached_provider_model_ids", return_value=live),
    ):
        payload = build_models_payload(ctx, explicit_only=True)

    row = next(row for row in payload["providers"] if row["slug"] == "antigravity")
    assert row["models"] == ["antigravity-gemini-3-pro", "claude-sonnet-4-6"]


def test_antigravity_explicit_only_normalizes_configured_provider_key():
    """Explicit-only preserves a managed row for a normalized config key."""
    ctx = ConfigContext(
        current_provider="nous",
        current_model="",
        current_base_url="",
        user_providers={" ANTIGRAVITY ": {}},
        custom_providers=[],
    )

    with _list_auth_returning([]):
        payload = build_models_payload(ctx, explicit_only=True)

    row = next(row for row in payload["providers"] if row["slug"] == "antigravity")
    assert row["is_current"] is False
    assert row["source"] == "managed"


# ─── _apply_featured (one-flagship-per-lab shortlist) ──────────────────


class _FakeInfo:
    def __init__(self, release_date: str) -> None:
        self.release_date = release_date


def _apply_featured_with_dates(rows, dates: dict[str, str]):
    """Run _apply_featured with a deterministic models.dev stub."""
    from hermes_cli import inventory

    def _fake_get_model_info(provider, model):
        return _FakeInfo(dates[model]) if model in dates else None

    with patch("agent.models_dev.get_model_info", side_effect=_fake_get_model_info):
        inventory._apply_featured(rows)



