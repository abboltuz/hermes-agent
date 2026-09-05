from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def antigravity_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _profile(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("antigravity")
    assert profile is not None
    calls = []

    def fetch_models(**kwargs):
        calls.append(kwargs)
        return fetch_models.result

    fetch_models.result = None
    monkeypatch.setattr(profile, "fetch_models", fetch_models)
    return profile, calls, fetch_models


def _write_cache(home, models):
    (home / "provider_models_cache.json").write_text(
        json.dumps(
            {
                "antigravity": {
                    "fp": "stale",
                    "at": time.time(),
                    "models": models,
                }
            }
        ),
        encoding="utf-8",
    )


def test_live_antigravity_catalog_is_used_by_picker(antigravity_home, monkeypatch):
    from hermes_cli import models

    _profile_obj, calls, fetch = _profile(monkeypatch)
    fetch.result = ["antigravity-gemini-3-pro", "antigravity-claude-sonnet-4-6"]

    assert models.provider_model_ids("antigravity") == fetch.result
    assert calls == [{}]
    assert not (antigravity_home / "provider_models_cache.json").exists()


def test_cached_live_antigravity_catalog_is_uncached_and_not_persisted(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    _profile_obj, calls, fetch = _profile(monkeypatch)
    live_models = ["antigravity-gemini-3-pro", "antigravity-claude-sonnet-4-6"]
    fetch.result = live_models

    def fail_cache_update(*args, **kwargs):
        raise AssertionError("Antigravity must not enter the generic model cache")

    monkeypatch.setattr(models, "update_provider_cache_entry", fail_cache_update)

    assert models.cached_provider_model_ids("antigravity") == live_models
    assert calls == [{}]
    assert not (antigravity_home / "provider_models_cache.json").exists()


def test_live_empty_is_authoritative_for_direct_and_cached_picker(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    _profile_obj, calls, fetch = _profile(monkeypatch)
    _write_cache(antigravity_home, ["stale-antigravity-model"])
    fetch.result = []

    assert models.provider_model_ids("antigravity") == []
    assert models.cached_provider_model_ids("antigravity", force_refresh=True) == []
    assert len(calls) == 2
    saved = json.loads((antigravity_home / "provider_models_cache.json").read_text())
    assert saved["antigravity"]["models"] == ["stale-antigravity-model"]


def test_discovery_failure_uses_profile_fallback_without_cache_write(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    profile, calls, fetch = _profile(monkeypatch)
    assert fetch.result is None

    assert models.provider_model_ids("antigravity") == list(profile.fallback_models)
    assert models.cached_provider_model_ids("antigravity", force_refresh=True) == list(
        profile.fallback_models
    )
    assert len(calls) == 2
    assert not (antigravity_home / "provider_models_cache.json").exists()


def test_discovery_exception_uses_profile_fallback_without_leaking_error(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    profile = __import__("providers", fromlist=["get_provider_profile"]).get_provider_profile(
        "antigravity"
    )
    assert profile is not None
    calls = []

    def fetch_models(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("private bridge failure")

    monkeypatch.setattr(profile, "fetch_models", fetch_models)

    assert models.provider_model_ids("antigravity") == list(profile.fallback_models)
    assert models.cached_provider_model_ids("antigravity", force_refresh=True) == list(
        profile.fallback_models
    )
    assert len(calls) == 2
    assert not (antigravity_home / "provider_models_cache.json").exists()


def test_google_antigravity_alias_uses_antigravity_picker_without_stealing_google(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    _profile_obj, calls, fetch = _profile(monkeypatch)
    fetch.result = ["antigravity-gemini-live"]

    assert models.normalize_provider("google-antigravity") == "antigravity"
    assert models.provider_model_ids("google-antigravity") == fetch.result
    assert calls == [{}]
    assert models.normalize_provider("google") == "gemini"
    assert models.provider_model_ids("google") == models.provider_model_ids("gemini")


def test_cursor_and_api_key_provider_controls_keep_existing_paths(monkeypatch):
    from hermes_cli import models
    from providers import get_provider_profile

    cursor = get_provider_profile("cursor")
    assert cursor is not None
    monkeypatch.setattr(cursor, "fetch_models", lambda **kwargs: ["cursor-live"])
    assert models.provider_model_ids("cursor") == ["cursor-live"]

    api_key_profile = get_provider_profile("openrouter")
    if api_key_profile is not None:
        monkeypatch.setattr(
            api_key_profile,
            "fetch_models",
            lambda **kwargs: ["openrouter-live"],
        )
    # Static fallback remains available without credentials/network access.
    assert models.provider_model_ids("openrouter")


def test_safe_antigravity_picker_ids_allow_preview_customtools_only_for_gemini():
    from hermes_cli.inventory import _safe_antigravity_model_ids

    valid_gemini = "antigravity-gemini-3.1-pro-preview"
    valid_customtools = "antigravity-gemini-3.1-pro-preview-customtools"
    valid_claude = "antigravity-claude-sonnet-4-6"
    valid_gpt_oss = "antigravity-gpt-oss-120b"
    rejected = [
        "antigravity-gemini-3.1-pro-preview-customtool",
        "antigravity-gemini-3.1-pro-preview-customtools-extra",
        "antigravity-gemini-3.1-pro-preview-customtools://host",
        "antigravity-claude-sonnet-4-6-customtools",
        "gpt-oss-120b",
        "antigravity-gpt-oss-120b-medium",
        "antigravity-gpt-oss-120b-extra",
        "antigravity-gpt-oss-120b://host",
        "antigravity-gpt-oss-120b\x00",
        valid_gpt_oss + "x" * 97,
        valid_customtools + "x" * 97,
    ]

    assert _safe_antigravity_model_ids(
        [valid_gemini, valid_customtools, valid_claude, valid_gpt_oss, *rejected]
    ) == [valid_gemini, valid_customtools, valid_claude, valid_gpt_oss]


def test_primary_runtime_resolves_connected_antigravity_managed_account(monkeypatch):
    from agent import antigravity_bridge_client, antigravity_bridge_transport
    from hermes_cli.runtime_provider import resolve_runtime_provider

    calls = []

    class FakeClient:
        def __init__(self, *, bridge_command):
            calls.append(("init", bridge_command))

        def list_accounts(self):
            calls.append(("list_accounts",))
            return {"connected": True, "total": 1}

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(
        antigravity_bridge_transport,
        "resolve_antigravity_bridge_command",
        lambda: "/tmp/fake-antigravity-bridge",
    )
    monkeypatch.setattr(antigravity_bridge_client, "AntigravityBridgeClient", FakeClient)

    runtime = resolve_runtime_provider(
        requested="antigravity",
        target_model="antigravity-gemini-3-pro",
    )

    assert runtime == {
        "provider": "antigravity",
        "api_mode": "chat_completions",
        "base_url": "sdkbridge://antigravity",
        "api_key": "antigravity-managed",
        "command": "/tmp/fake-antigravity-bridge",
        "args": [],
        "source": "managed-bridge",
        "requested_provider": "antigravity",
    }
    assert calls == [
        ("init", "/tmp/fake-antigravity-bridge"),
        ("list_accounts",),
        ("close",),
    ]


@pytest.mark.parametrize(
    ("logged_in", "expected"),
    [(True, True), (False, False), (1, False), ("true", False)],
)
def test_connected_antigravity_account_is_an_inference_provider(
    antigravity_home, monkeypatch, logged_in, expected
):
    from hermes_cli import auth, main

    calls = []

    def fake_auth_status(provider):
        calls.append(provider)
        if provider == "antigravity":
            return {"provider": provider, "logged_in": logged_in, "configured": logged_in}
        return {"provider": provider, "logged_in": False}

    monkeypatch.setattr(auth, "get_auth_status", fake_auth_status)

    assert main._has_any_provider_configured() is expected
    assert "antigravity" in calls


def test_antigravity_auth_registry_uses_managed_external_process_contract():
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider

    config = PROVIDER_REGISTRY["antigravity"]
    assert resolve_provider("antigravity") == "antigravity"
    assert config.auth_type == "external_process"
    assert config.inference_base_url == "sdkbridge://antigravity"
    assert config.api_key_env_vars == ()
    assert config.base_url_env_var == ""


@pytest.mark.parametrize("connected", [False, None, 1, "true", [], {}])
def test_antigravity_external_credentials_require_literal_connected(
    monkeypatch, connected
):
    from agent import antigravity_bridge_client, antigravity_bridge_transport
    from hermes_cli.auth import AuthError, resolve_external_process_provider_credentials

    class FakeClient:
        def __init__(self, *, bridge_command):
            assert bridge_command == "/tmp/fake-antigravity-bridge"

        def list_accounts(self):
            return {"connected": connected, "total": 1}

        def close(self):
            return None

    monkeypatch.setattr(
        antigravity_bridge_transport,
        "resolve_antigravity_bridge_command",
        lambda: "/tmp/fake-antigravity-bridge",
    )
    monkeypatch.setattr(antigravity_bridge_client, "AntigravityBridgeClient", FakeClient)

    with pytest.raises(AuthError) as exc_info:
        resolve_external_process_provider_credentials("antigravity")
    assert exc_info.value.code == "missing_antigravity_account"


def test_antigravity_external_credentials_fail_closed_without_bridge(monkeypatch):
    from agent import antigravity_bridge_transport
    from hermes_cli.auth import AuthError, resolve_external_process_provider_credentials

    monkeypatch.setattr(
        antigravity_bridge_transport,
        "resolve_antigravity_bridge_command",
        lambda: None,
    )

    with pytest.raises(AuthError) as exc_info:
        resolve_external_process_provider_credentials("antigravity")
    assert exc_info.value.code == "missing_antigravity_bridge"


def test_antigravity_auth_status_projects_only_safe_account_state(monkeypatch):
    from agent import antigravity_bridge_client, antigravity_bridge_transport
    from hermes_cli.auth import get_auth_status

    private_values = ["account-private", "project-private", "token-private", "/private/store"]

    class FakeClient:
        def __init__(self, *, bridge_command):
            assert bridge_command == "/tmp/fake-antigravity-bridge"

        def list_accounts(self):
            return {
                "connected": True,
                "total": 2,
                "current": {"gemini": private_values[0]},
                "accounts": [{"id": private_values[0], "projectId": private_values[1]}],
                "accessToken": private_values[2],
                "storePath": private_values[3],
            }

        def close(self):
            return None

    monkeypatch.setattr(
        antigravity_bridge_transport,
        "resolve_antigravity_bridge_command",
        lambda: "/tmp/fake-antigravity-bridge",
    )
    monkeypatch.setattr(antigravity_bridge_client, "AntigravityBridgeClient", FakeClient)

    status = get_auth_status("antigravity")

    assert status == {
        "provider": "antigravity",
        "logged_in": True,
        "configured": True,
        "bridge_available": True,
        "base_url": "sdkbridge://antigravity",
        "source": "managed-bridge",
        "account_count": 2,
    }
    encoded = json.dumps(status, sort_keys=True)
    assert all(value not in encoded for value in private_values)


def test_antigravity_auth_status_redacts_bridge_errors(monkeypatch):
    from agent import antigravity_bridge_client, antigravity_bridge_transport
    from hermes_cli.auth import (
        AuthError,
        get_auth_status,
        resolve_external_process_provider_credentials,
    )

    private_error = "private account and token details"

    class FailingClient:
        def __init__(self, *, bridge_command):
            assert bridge_command == "/tmp/fake-antigravity-bridge"

        def list_accounts(self):
            raise RuntimeError(private_error)

        def close(self):
            return None

    monkeypatch.setattr(
        antigravity_bridge_transport,
        "resolve_antigravity_bridge_command",
        lambda: "/tmp/fake-antigravity-bridge",
    )
    monkeypatch.setattr(
        antigravity_bridge_client,
        "AntigravityBridgeClient",
        FailingClient,
    )

    status = get_auth_status("antigravity")
    assert status["logged_in"] is False
    assert status["configured"] is False
    assert status["error"] == "RuntimeError"
    assert private_error not in json.dumps(status, sort_keys=True)

    with pytest.raises(AuthError) as exc_info:
        resolve_external_process_provider_credentials("antigravity")
    assert exc_info.value.code == "antigravity_account_status_unavailable"
    assert private_error not in str(exc_info.value)
