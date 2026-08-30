from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from agent.cursor_bridge_client import AsyncCursorBridgeClient, CursorBridgeClient
from agent.cursor_sdk_auth import (
    CursorAuthError,
    read_sdk_credentials,
    resolve_cursor_api_key,
    save_sdk_credentials,
    sdk_auth_path,
)


@pytest.fixture(autouse=True)
def _reset_cursor_refresh_coordinator():
    import hermes_cli.models as models_mod

    with models_mod._cursor_refresh_lock:
        models_mod._cursor_refresh_inflight.clear()
        models_mod._cursor_refresh_failed_at.clear()
    yield
    with models_mod._cursor_refresh_lock:
        models_mod._cursor_refresh_inflight.clear()
        models_mod._cursor_refresh_failed_at.clear()


@pytest.fixture
def cursor_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_BACKEND_URL", raising=False)
    return tmp_path


def test_profile_credential_is_private_atomic_and_not_returned_by_status(cursor_home):
    path = save_sdk_credentials(
        backend_url="https://api2.cursor.sh",
        api_key="test-redacted",
        email="person@example.invalid",
    )
    assert path == sdk_auth_path()
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
    stored = read_sdk_credentials()
    assert stored is not None
    assert stored["source"] == "cursor_login"

    from hermes_cli.auth import get_auth_status

    status = get_auth_status("cursor")
    assert status["logged_in"] is True
    assert status["source"] == "cursor_login"
    assert status["source_label"] == str(path)
    assert "api_key" not in status
    assert "token_preview" not in status


def test_profile_dotenv_precedes_stale_process_env(cursor_home, monkeypatch):
    (cursor_home / ".env").write_text("CURSOR_API_KEY=dotenv-value\n", encoding="utf-8")
    monkeypatch.setenv("CURSOR_API_KEY", "stale-shell-value")
    assert resolve_cursor_api_key() == ("dotenv-value", "env")


def test_cursor_profile_fingerprint_is_scoped_and_unscoped_resolution_fails_closed(
    tmp_path, monkeypatch
):
    """Multiplexed catalog identity must follow the active profile context."""
    from agent.secret_scope import (
        UnscopedSecretError,
        is_multiplex_active,
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli import models

    was_multiplexed = is_multiplex_active()
    monkeypatch.setenv("CURSOR_API_KEY", "process-only-sentinel")
    set_multiplex_active(True)
    try:
        with pytest.raises(UnscopedSecretError):
            resolve_cursor_api_key()

        home_a = tmp_path / "profile-a"
        home_b = tmp_path / "profile-b"
        home_a_token = set_hermes_home_override(home_a)
        scope_a_token = set_secret_scope({"CURSOR_API_KEY": "scoped-a"})
        try:
            fingerprint_a = models._credential_fingerprint("cursor")
        finally:
            reset_secret_scope(scope_a_token)
            reset_hermes_home_override(home_a_token)

        home_b_token = set_hermes_home_override(home_b)
        scope_b_token = set_secret_scope({"CURSOR_API_KEY": "scoped-b"})
        try:
            fingerprint_b = models._credential_fingerprint("cursor")
        finally:
            reset_secret_scope(scope_b_token)
            reset_hermes_home_override(home_b_token)

        assert fingerprint_a != fingerprint_b
    finally:
        set_multiplex_active(was_multiplexed)


def test_credential_save_rejects_symlink_target(cursor_home):
    target = cursor_home / "must-not-change"
    target.write_text("safe", encoding="utf-8")
    path = sdk_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(CursorAuthError, match="symlink"):
        save_sdk_credentials(
            backend_url="https://api2.cursor.sh",
            api_key="test-redacted",
        )
    assert target.read_text(encoding="utf-8") == "safe"


def test_canonical_status_remove_and_logout_use_cursor_store(cursor_home, capsys):
    save_sdk_credentials(
        backend_url="https://api2.cursor.sh", api_key="test-redacted"
    )
    from hermes_cli.auth_commands import auth_remove_command, auth_status_command

    auth_status_command(SimpleNamespace(provider="cursor"))
    assert capsys.readouterr().out == "cursor: logged in\n"
    auth_remove_command(SimpleNamespace(provider="cursor", target=None, index=None))
    assert "Removed the profile-scoped Cursor credential" in capsys.readouterr().out
    assert not sdk_auth_path().exists()

    save_sdk_credentials(
        backend_url="https://api2.cursor.sh", api_key="test-redacted"
    )
    from hermes_cli.auth import clear_provider_auth

    assert clear_provider_auth("cursor") is True
    assert not sdk_auth_path().exists()


def test_canonical_login_installs_pinned_bridge_when_missing(cursor_home, monkeypatch, capsys):
    from agent import cursor_bridge_transport, cursor_sdk_auth
    from hermes_cli.auth_commands import auth_add_command

    installed = []
    monkeypatch.setattr(
        cursor_sdk_auth,
        "login",
        lambda **_kwargs: {"email": "person@example.invalid"},
    )
    monkeypatch.setattr(cursor_bridge_transport, "resolve_bridge_command", lambda: None)
    monkeypatch.setattr(
        cursor_bridge_transport,
        "download_bridge",
        lambda **_kwargs: installed.append(True) or "/tmp/fake-cursor-sdk-bridge",
    )

    auth_add_command(
        SimpleNamespace(
            provider="cursor",
            auth_type="oauth",
            no_browser=True,
            label="",
        )
    )
    output = capsys.readouterr().out
    assert installed == [True]
    assert "Installing the pinned Cursor SDK bridge" in output
    assert "person@example.invalid" in output


def test_picker_and_desktop_account_catalog_share_cursor_auth(cursor_home, monkeypatch):
    save_sdk_credentials(
        backend_url="https://api2.cursor.sh", api_key="test-redacted"
    )
    import hermes_cli.model_switch as model_switch
    import hermes_cli.models as models

    monkeypatch.setattr(
        models,
        "cached_provider_model_ids",
        lambda slug, *args, **kwargs: ["auto"] if slug == "cursor" else [],
    )
    rows = model_switch.list_authenticated_providers(
        probe_custom_providers=False,
        for_picker=True,
    )
    cursor_row = next(row for row in rows if row["slug"] == "cursor")
    assert cursor_row["models"] == ["auto"]

    from hermes_cli.web_server import (
        _build_oauth_catalog,
        _oauth_provider_disconnect_command,
    )

    card = next(row for row in _build_oauth_catalog() if row["id"] == "cursor")
    assert card["flow"] == "browser_poll"
    assert card["cli_command"] == "hermes auth add cursor"
    assert card.get("disconnect_command") is None
    assert _oauth_provider_disconnect_command(card) is None


def test_provider_model_catalog_uses_cursor_profile_live_fetch(cursor_home, monkeypatch):
    from hermes_cli import models
    from providers import get_provider_profile

    profile = get_provider_profile("cursor")
    assert profile is not None
    monkeypatch.setattr(profile, "fetch_models", lambda **_kwargs: ["auto", "live-model"])
    assert models.provider_model_ids("cursor") == ["auto", "live-model"]


def test_cursor_discovery_none_is_ordinary_auto_fallback_not_cached_live(cursor_home, monkeypatch):
    from hermes_cli import models
    from providers import get_provider_profile

    profile = get_provider_profile("cursor")
    assert profile is not None
    monkeypatch.setattr(profile, "fetch_models", lambda **_kwargs: None)

    assert models.provider_model_ids("cursor") == ["auto"]
    out = models.cached_provider_model_ids("cursor", force_refresh=True)
    assert out == ["auto"]
    cache_path = models._provider_models_cache_path()
    if cache_path.exists():
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        assert "cursor" not in saved


def test_cursor_sdk_auto_catalog_is_cached_as_live(cursor_home, monkeypatch):
    from hermes_cli import models
    from providers import get_provider_profile

    profile = get_provider_profile("cursor")
    assert profile is not None
    monkeypatch.setattr(profile, "fetch_models", lambda **_kwargs: ["auto"])

    assert models.provider_model_ids("cursor") == ["auto"]
    assert models.cached_provider_model_ids("cursor", force_refresh=True) == ["auto"]
    saved = json.loads(models._provider_models_cache_path().read_text(encoding="utf-8"))
    assert saved["cursor"]["models"] == ["auto"]


def test_runtime_provider_resolves_profile_cursor_account(cursor_home, monkeypatch):
    save_sdk_credentials(
        backend_url="https://api2.cursor.sh", api_key="test-redacted"
    )
    from agent import cursor_bridge_transport
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setattr(
        cursor_bridge_transport,
        "resolve_bridge_command",
        lambda *args, **kwargs: "/tmp/fake-cursor-sdk-bridge",
    )
    runtime = resolve_runtime_provider(requested="cursor", target_model="auto")
    assert runtime["provider"] == "cursor"
    assert runtime["api_key"] == "test-redacted"
    assert runtime["base_url"] == "sdkbridge://cursor"
    assert runtime["command"] == "/tmp/fake-cursor-sdk-bridge"


def test_model_flow_uses_live_cursor_models_and_persists_provider(cursor_home, monkeypatch):
    save_sdk_credentials(
        backend_url="https://api2.cursor.sh", api_key="test-redacted"
    )
    from agent import cursor_bridge_client, cursor_bridge_transport
    from hermes_cli import auth, model_setup_flows

    recorded = {}

    class FakeClient:
        def __init__(self, **kwargs):
            recorded["client_kwargs"] = kwargs

        def list_models(self):
            return [{"id": "live-cursor-model"}]

        def close(self):
            return None

    monkeypatch.setattr(
        cursor_bridge_transport,
        "resolve_bridge_command",
        lambda *args, **kwargs: "/tmp/fake-cursor-sdk-bridge",
    )
    monkeypatch.setattr(cursor_bridge_client, "CursorBridgeClient", FakeClient)
    monkeypatch.setattr(
        model_setup_flows, "_prompt_auth_credentials_choice", lambda _prompt: "use"
    )
    monkeypatch.setattr(
        auth,
        "_prompt_model_selection",
        lambda models, **_kwargs: models[0],
    )
    monkeypatch.setattr(auth, "_save_model_choice", lambda model: recorded.setdefault("model", model))
    monkeypatch.setattr(
        auth,
        "_update_config_for_provider",
        lambda provider, base_url: recorded.update(provider=provider, base_url=base_url),
    )

    model_setup_flows._model_flow_cursor({}, current_model="auto")
    assert recorded["client_kwargs"]["bridge_command"] == "/tmp/fake-cursor-sdk-bridge"
    assert recorded["model"] == "live-cursor-model"
    assert recorded["provider"] == "cursor"
    assert recorded["base_url"] == "sdkbridge://cursor"


def test_auxiliary_router_returns_sync_and_async_cursor_clients(cursor_home, monkeypatch):
    from agent import cursor_bridge_transport
    from agent import cursor_sdk_auth
    from agent.auxiliary_client import resolve_provider_client

    monkeypatch.setattr(
        cursor_sdk_auth,
        "resolve_cursor_api_key",
        lambda: ("test-redacted", "cursor_login"),
    )
    monkeypatch.setattr(
        cursor_bridge_transport,
        "resolve_bridge_command",
        lambda *args, **kwargs: "/tmp/fake-cursor-sdk-bridge",
    )

    sync_client, sync_model = resolve_provider_client("cursor", model="auto")
    async_client, async_model = resolve_provider_client(
        "cursor", model="auto", async_mode=True
    )
    assert sync_client is not None
    assert async_client is not None
    try:
        assert isinstance(sync_client, CursorBridgeClient)
        assert sync_client._bridge_command == "/tmp/fake-cursor-sdk-bridge"
        assert sync_model == "auto"
        assert isinstance(async_client, AsyncCursorBridgeClient)
        assert async_client._sync._bridge_command == "/tmp/fake-cursor-sdk-bridge"
        assert async_model == "auto"
    finally:
        sync_client.close()
        import asyncio
        asyncio.run(async_client.close())


def test_browser_poll_timeout_is_terminal_and_scrubs_session(cursor_home, monkeypatch):
    from hermes_cli import web_server as ws

    class Handshake:
        login_url = "https://cursor.com/loginDeepControl?challenge=fixture"
        uuid = "fixture-uuid"
        verifier = "fixture-verifier"

    with ws._oauth_sessions_lock:
        ws._oauth_sessions.clear()
    sid, session = ws._new_oauth_session("cursor", "browser_poll", profile="selected")
    session.update(handshake=Handshake(), uuid=Handshake.uuid, verifier=Handshake.verifier,
                   expires_at=9999999999)
    monkeypatch.setattr("agent.cursor_sdk_auth.poll_for_login_tokens", lambda **_: None)
    monkeypatch.setattr("agent.cursor_bridge_transport.resolve_bridge_command", lambda: "/tmp/bridge")
    ws._cursor_login_worker(sid)
    assert session["status"] == "error"
    assert "uuid" not in session and "verifier" not in session
    with ws._oauth_sessions_lock:
        ws._oauth_sessions.clear()


def test_cursor_worker_bridge_failure_is_redacted_and_stops_before_auth(
    cursor_home, monkeypatch
):
    from agent import cursor_bridge_transport
    from agent.cursor_sdk_auth import create_login_handshake
    from hermes_cli import web_server as ws

    with ws._oauth_sessions_lock:
        ws._oauth_sessions.clear()
    sid, session = ws._new_oauth_session("cursor", "browser_poll")
    session.update(handshake=create_login_handshake(), expires_at=9999999999)
    calls = []
    monkeypatch.setattr(
        cursor_bridge_transport, "resolve_bridge_command", lambda: None
    )

    def fail_download(**_kwargs):
        raise RuntimeError("bridge-secret-fixture")

    monkeypatch.setattr(cursor_bridge_transport, "download_bridge", fail_download)
    monkeypatch.setattr(
        "agent.cursor_sdk_auth.poll_for_login_tokens",
        lambda **_: calls.append("poll"),
    )
    monkeypatch.setattr(
        "agent.cursor_sdk_auth.mint_user_api_key",
        lambda **_: calls.append("mint"),
    )
    monkeypatch.setattr(
        "agent.cursor_sdk_auth.save_sdk_credentials",
        lambda **_: calls.append("save"),
    )

    try:
        ws._cursor_login_worker(sid)
        assert calls == []
        assert session["status"] == "error"
        assert session["error_message"] == (
            "Cursor sign-in could not be completed. Please try again."
        )
        assert "bridge-secret-fixture" not in session["error_message"]
        assert "handshake" not in session
    finally:
        with ws._oauth_sessions_lock:
            ws._oauth_sessions.clear()


def test_gc_cancels_and_scrubs_cursor_worker(cursor_home):
    from hermes_cli import web_server as ws

    with ws._oauth_sessions_lock:
        ws._oauth_sessions.clear()
    sid, session = ws._new_oauth_session("cursor", "browser_poll")
    session.update(created_at=0, uuid="fixture-uuid", verifier="fixture-verifier")
    ws._gc_oauth_sessions()
    assert session["cancelled"] is True
    assert session["cancel_event"].is_set()
    assert "uuid" not in session and "verifier" not in session
    with ws._oauth_sessions_lock:
        assert sid not in ws._oauth_sessions
        ws._oauth_sessions.clear()


def test_cursor_worker_cancelled_during_email_never_mints_or_saves(cursor_home, monkeypatch):
    from hermes_cli import web_server as ws
    from agent import cursor_bridge_transport
    from agent.cursor_sdk_auth import create_login_handshake

    with ws._oauth_sessions_lock:
        ws._oauth_sessions.clear()
    sid, session = ws._new_oauth_session("cursor", "browser_poll", profile="selected")
    session.update(handshake=create_login_handshake(), expires_at=9999999999)
    cancel_event = session["cancel_event"]
    calls = []
    monkeypatch.setattr(cursor_bridge_transport, "resolve_bridge_command", lambda: "/tmp/bridge")
    monkeypatch.setattr("agent.cursor_sdk_auth.poll_for_login_tokens", lambda **_: {"accessToken": "token-fixture"})
    monkeypatch.setattr("agent.cursor_sdk_auth.resolve_backend_url", lambda: "https://cursor.example")
    def email(_url, _token):
        cancel_event.set()
        return "person@example.invalid"
    monkeypatch.setattr("agent.cursor_sdk_auth.get_login_email", email)
    monkeypatch.setattr("agent.cursor_sdk_auth.mint_user_api_key", lambda **_: calls.append("mint"))
    monkeypatch.setattr("agent.cursor_sdk_auth.save_sdk_credentials", lambda **_: calls.append("save"))
    try:
        ws._cursor_login_worker(sid)
        assert calls == []
        assert session["status"] == "pending"
    finally:
        with ws._oauth_sessions_lock:
            ws._oauth_sessions.clear()


def test_cursor_worker_success_order_and_scrubs_secrets(cursor_home, monkeypatch):
    from hermes_cli import web_server as ws
    from agent import cursor_bridge_transport
    from agent.cursor_sdk_auth import create_login_handshake

    with ws._oauth_sessions_lock:
        ws._oauth_sessions.clear()
    sid, session = ws._new_oauth_session("cursor", "browser_poll", profile="selected")
    (cursor_home / "profiles" / "selected").mkdir(parents=True)
    session.update(handshake=create_login_handshake(), expires_at=9999999999)
    events = []
    monkeypatch.setattr(cursor_bridge_transport, "resolve_bridge_command", lambda: events.append("bridge") or "/tmp/bridge")
    monkeypatch.setattr("agent.cursor_sdk_auth.poll_for_login_tokens", lambda **_: events.append("poll") or {"accessToken": "token-fixture"})
    monkeypatch.setattr("agent.cursor_sdk_auth.resolve_backend_url", lambda: "https://cursor.example")
    monkeypatch.setattr("agent.cursor_sdk_auth.get_login_email", lambda *_: events.append("email") or "person@example.invalid")
    monkeypatch.setattr("agent.cursor_sdk_auth.mint_user_api_key", lambda **_: events.append("mint") or "api-fixture")
    monkeypatch.setattr("agent.cursor_sdk_auth.save_sdk_credentials", lambda **_: events.append("save"))
    try:
        ws._cursor_login_worker(sid)
        assert session["status"] == "approved"
        assert events[:5] == ["bridge", "bridge", "poll", "email", "mint"]
        assert events[-1] == "save"
        assert "handshake" not in session and "api_key" not in session
    finally:
        with ws._oauth_sessions_lock:
            ws._oauth_sessions.clear()
