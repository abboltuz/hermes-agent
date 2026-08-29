"""Behavioral security contracts for the Cursor SDK bridge."""

import io
import json
import os
from pathlib import Path

import pytest

from agent.cursor_bridge_transport import (
    CursorBridgeError,
    _build_subprocess_env,
    endpoint_from_discovery,
)
from agent.cursor_bridge_wire import decode_call_custom_tool_request
from agent.cursor_sdk_auth import CursorAuthError, resolve_backend_url, resolve_website_url
from hermes_cli.auth import get_auth_status
from agent.cursor_bridge_transport import download_bridge
from agent import cursor_bridge_transport as transport


def test_discovery_rejects_non_loopback_and_url_tricks(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    base = {"schemaVersion": 1, "transport": "tcp", "protocol": "connect", "authToken": "x"}
    for url in ("https://127.0.0.1:1234", "http://127.0.0.2:1234", "http://127.0.0.1:1234?x=1", "http://u:p@127.0.0.1:1234"):
        with pytest.raises(CursorBridgeError):
            endpoint_from_discovery({**base, "url": url})


def test_discovery_token_file_must_be_private_regular_file(tmp_path):
    token_dir = tmp_path / "cursor-sdk-bridge-test"
    token_dir.mkdir(mode=0o700)
    token_dir.chmod(0o700)
    token = token_dir / "auth-token"
    token.write_text("secret")
    token.chmod(0o600)
    payload = {"schemaVersion": 1, "transport": "tcp", "protocol": "connect", "url": "http://127.0.0.1:1234", "authTokenFile": str(token)}
    assert endpoint_from_discovery(payload).url == "http://127.0.0.1:1234"
    token.chmod(0o644)
    with pytest.raises(CursorBridgeError):
        endpoint_from_discovery(payload)


def test_subprocess_environment_does_not_inherit_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-propagate")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-propagate")
    env = _build_subprocess_env("cursor-key")
    assert env["CURSOR_API_KEY"] == "cursor-key"
    assert "OPENAI_API_KEY" not in env
    assert "TELEGRAM_BOT_TOKEN" not in env


def test_bridge_process_uses_official_callback_token_env_not_argv(monkeypatch, tmp_path):
    captured = {}
    ready = {
        "schemaVersion": 1,
        "transport": "tcp",
        "protocol": "connect",
        "url": "http://127.0.0.1:43123",
        "authToken": "bridge-capability",
    }

    class FakePopen:
        def __init__(self, args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs["env"]
            self.stdout = io.StringIO()
            self.stderr = io.StringIO(
                transport.READY_LINE_PREFIX + json.dumps(ready) + "\n"
            )
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode or 0

        def kill(self):
            self.returncode = -9

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-propagate")
    monkeypatch.setattr(transport.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(transport.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_STARTUP_TIMEOUT_SECONDS", 1.0)
    process = transport.CursorBridgeProcess(
        command="/tmp/fake-cursor-bridge",
        api_key="cursor-key",
        workspace=str(tmp_path),
        tool_callback_url="http://127.0.0.1:43210/callback",
        tool_callback_auth_token="callback-capability",
    )
    try:
        process.start()
    finally:
        process.stop()

    assert "callback-capability" not in captured["args"]
    assert captured["env"]["CURSOR_SDK_TOOL_CALLBACK_AUTH_TOKEN"] == "callback-capability"
    assert "CURSOR_TOOL_CALLBACK_AUTH_TOKEN" not in captured["env"]
    assert "OPENAI_API_KEY" not in captured["env"]


def test_bridge_stderr_diagnostics_redact_child_secrets(tmp_path):
    process = transport.CursorBridgeProcess(
        command="/tmp/fake-cursor-bridge",
        api_key="opaque-cursor-secret",
        workspace=str(tmp_path),
        tool_callback_auth_token="opaque-callback-secret",
    )
    redacted = process._redact_diagnostic(
        "CURSOR_API_KEY=opaque-cursor-secret callback=opaque-callback-secret"
    )
    assert "opaque-cursor-secret" not in redacted
    assert "opaque-callback-secret" not in redacted


def test_wire_rejects_truncated_and_oversized_fields():
    with pytest.raises(ValueError, match="truncated"):
        decode_call_custom_tool_request(b"\x0a\x05x")
    with pytest.raises(ValueError, match="size limit"):
        # field 1, length 262,145 (> the 256 KiB container cap)
        decode_call_custom_tool_request(b"\x0a\x81\x80\x10")


def test_auth_urls_are_https_and_allowlisted():
    assert resolve_website_url("https://cursor.com/") == "https://cursor.com"
    assert resolve_backend_url("https://api2.cursor.sh/") == "https://api2.cursor.sh"
    for url in (
        "http://cursor.com",
        "https://cursor.com.attacker.test",
        "https://user:pass@cursor.com",
        "https://cursor.com/?next=https://attacker.test",
    ):
        with pytest.raises(CursorAuthError):
            resolve_website_url(url)
    for url in ("https://api2.cursor.sh.evil", "http://api2.cursor.sh"):
        with pytest.raises(CursorAuthError):
            resolve_backend_url(url)


def test_cursor_auth_status_uses_explicit_profile_credential(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CURSOR_API_KEY", "test-cursor-key")
    status = get_auth_status("cursor")
    assert status["provider"] == "cursor"
    assert status["logged_in"] is True
    assert status["credential_source"] == "env"


def test_bridge_download_rejects_unknown_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(CursorBridgeError, match="no pinned digest"):
        download_bridge("0.0.0", progress=False)
