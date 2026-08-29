from __future__ import annotations

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
    assert card["cli_command"] == "hermes auth add cursor"
    assert _oauth_provider_disconnect_command(card) == "hermes cursor logout"


def test_provider_model_catalog_uses_cursor_profile_live_fetch(cursor_home, monkeypatch):
    from hermes_cli import models
    from providers import get_provider_profile

    profile = get_provider_profile("cursor")
    assert profile is not None
    monkeypatch.setattr(profile, "fetch_models", lambda **_kwargs: ["auto", "live-model"])
    assert models.provider_model_ids("cursor") == ["auto", "live-model"]


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
