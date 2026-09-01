"""Contract tests for the Antigravity account-control JSON-RPC adapter."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from agent.antigravity_bridge_transport import AntigravityBridgeError

ACCOUNT_ID = "acct_123e4567-e89b-12d3-a456-426614174000"
SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"
AUTH_URL = (
    "https://accounts.google.com/o/oauth2/v2/auth?client_id=client&response_type=code"
    "&redirect_uri=http%3A%2F%2Flocalhost%3A51121%2Foauth-callback&scope=openid"
    "&code_challenge=challenge&code_challenge_method=S256&state=state"
    "&access_type=offline&prompt=consent"
)


class FakeAccountService:
    def __init__(self, label: str, *, fail: Exception | None = None, close_fail: Exception | None = None):
        self.label = label
        self.calls: list[tuple[Any, ...]] = []
        self.fail = fail
        self.close_fail = close_fail
        self.close_calls = 0

    def _result(self, name: str, *args: Any) -> dict[str, Any]:
        self.calls.append((name, *args))
        if self.fail:
            raise self.fail
        if name == "start_oauth":
            return {
                "session_id": SESSION_ID,
                "auth_url": AUTH_URL,
                "flow": "browser_poll",
                "status": "pending",
                "expires_at": 123,
                "poll_interval_ms": 1000,
            }
        if name in {"poll_oauth", "cancel_oauth"}:
            return {"status": "cancelled"}
        return {"accounts": [{"id": self.label}]}

    def list_accounts(self):
        return self._result("list_accounts")

    def set_account_enabled(self, account_id, enabled):
        return self._result("set_account_enabled", account_id, enabled)

    def set_account_priority(self, account_id, priority):
        return self._result("set_account_priority", account_id, priority)

    def remove_account(self, account_id):
        return self._result("remove_account", account_id)

    def start_oauth(self, project_id=""):
        return self._result("start_oauth", project_id)

    def poll_oauth(self, session_id):
        return self._result("poll_oauth", session_id)

    def cancel_oauth(self, session_id):
        return self._result("cancel_oauth", session_id)

    def close(self):
        self.close_calls += 1
        if self.close_fail:
            raise self.close_fail


@pytest.fixture()
def server():
    mod = importlib.import_module("tui_gateway.server")
    original_methods = dict(mod._methods)
    try:
        yield mod
    finally:
        cleanup = getattr(mod, "_shutdown_antigravity_services", None)
        if cleanup is not None:
            cleanup()
        mod._methods.clear()
        mod._methods.update(original_methods)


def _rpc(server, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return server.handle_request({"jsonrpc": "2.0", "id": method, "method": method, "params": params})


def test_registers_all_antigravity_rpc_methods_for_real_json_rpc_dispatch(server):
    assert {
        "antigravity.accounts.list",
        "antigravity.accounts.enabled",
        "antigravity.accounts.priority",
        "antigravity.accounts.remove",
        "antigravity.oauth.start",
        "antigravity.oauth.poll",
        "antigravity.oauth.cancel",
    } <= set(server._methods)


def test_account_and_oauth_calls_delegate_under_the_requested_profile(server, monkeypatch, tmp_path):
    homes = {"alpha": tmp_path / "alpha"}
    homes["alpha"].mkdir()
    instances: list[FakeAccountService] = []
    monkeypatch.setattr(server, "_profile_home", lambda profile: homes.get(profile))
    monkeypatch.setattr(
        server,
        "_antigravity_service_factory",
        lambda: instances.append(FakeAccountService("alpha")) or instances[-1],
    )

    assert _rpc(server, "antigravity.accounts.list", {"profile": "alpha"})["result"] == {
        "accounts": {"accounts": [{"id": "alpha"}]}
    }
    assert _rpc(server, "antigravity.accounts.enabled", {"profile": "alpha", "account_id": ACCOUNT_ID, "enabled": True})["result"]["snapshot"] == {"accounts": [{"id": "alpha"}]}
    assert _rpc(server, "antigravity.accounts.priority", {"profile": "alpha", "account_id": ACCOUNT_ID, "priority": 2})["result"]["snapshot"] == {"accounts": [{"id": "alpha"}]}
    assert _rpc(server, "antigravity.accounts.remove", {"profile": "alpha", "account_id": ACCOUNT_ID})["result"]["snapshot"] == {"accounts": [{"id": "alpha"}]}
    assert _rpc(server, "antigravity.oauth.start", {"profile": "alpha", "project_id": "project"})["result"] == {
        "session_id": SESSION_ID,
        "auth_url": AUTH_URL,
        "flow": "browser_poll",
        "status": "pending",
        "expires_at": 123,
        "poll_interval_ms": 1000,
    }
    assert _rpc(server, "antigravity.oauth.poll", {"profile": "alpha", "session_id": SESSION_ID})["result"] == {"status": "cancelled"}
    assert _rpc(server, "antigravity.oauth.cancel", {"profile": "alpha", "session_id": SESSION_ID})["result"] == {"status": "cancelled"}
    assert len(instances) == 1
    assert instances[0].calls == [
        ("list_accounts",),
        ("set_account_enabled", ACCOUNT_ID, True),
        ("set_account_priority", ACCOUNT_ID, 2),
        ("remove_account", ACCOUNT_ID),
        ("start_oauth", "project"),
        ("poll_oauth", SESSION_ID),
        ("cancel_oauth", SESSION_ID),
    ]


def test_profile_registry_reuses_only_its_own_service_and_shutdown_closes_once(server, monkeypatch, tmp_path):
    homes = {name: tmp_path / name for name in ("alpha", "beta")}
    for home in homes.values():
        home.mkdir()
    instances: list[FakeAccountService] = []
    monkeypatch.setattr(server, "_profile_home", lambda profile: homes.get(profile))
    monkeypatch.setattr(
        server,
        "_antigravity_service_factory",
        lambda: instances.append(FakeAccountService(str(len(instances)))) or instances[-1],
    )

    assert _rpc(server, "antigravity.accounts.list", {"profile": "alpha"})["result"]["accounts"]["accounts"] == [{"id": "0"}]
    assert _rpc(server, "antigravity.accounts.list", {"profile": "beta"})["result"]["accounts"]["accounts"] == [{"id": "1"}]
    assert _rpc(server, "antigravity.accounts.list", {"profile": "alpha"})["result"]["accounts"]["accounts"] == [{"id": "0"}]
    assert [instance.calls for instance in instances] == [[("list_accounts",), ("list_accounts",)], [("list_accounts",)]]

    server._shutdown_antigravity_services()
    server._shutdown_antigravity_services()

    assert [instance.close_calls for instance in instances] == [1, 1]


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("antigravity.accounts.enabled", {"account_id": "bad", "enabled": True}),
        ("antigravity.accounts.priority", {"account_id": ACCOUNT_ID, "priority": 0}),
        ("antigravity.accounts.remove", {"account_id": "bad"}),
        ("antigravity.oauth.start", {"project_id": "bad\x00"}),
        ("antigravity.oauth.poll", {"session_id": "bad"}),
        ("antigravity.oauth.cancel", {"session_id": "bad"}),
    ],
)
def test_invalid_antigravity_params_return_standard_error_without_constructing_service(server, monkeypatch, method, params):
    factory_calls = []
    monkeypatch.setattr(server, "_antigravity_service_factory", lambda: factory_calls.append(True))

    response = _rpc(server, method, params)

    assert response["error"] == {"code": -32602, "message": "invalid Antigravity parameters"}
    assert factory_calls == []


@pytest.mark.parametrize("error", [AntigravityBridgeError("secret bridge marker"), RuntimeError("secret unexpected marker")])
def test_service_failures_are_mapped_to_fixed_safe_errors_without_leaking_details(server, monkeypatch, error):
    monkeypatch.setattr(server, "_antigravity_service_factory", lambda: FakeAccountService("x", fail=error))

    response = _rpc(server, "antigravity.accounts.list", {})

    assert response["error"] == {
        "code": 5201 if isinstance(error, AntigravityBridgeError) else 5202,
        "message": "Antigravity account service unavailable" if isinstance(error, AntigravityBridgeError) else "Antigravity account service failed",
    }
    assert "secret" not in str(response)


def test_shutdown_observes_cleanup_failure_without_leaking_or_stopping_other_services(server, monkeypatch, caplog, tmp_path):
    homes = {name: tmp_path / name for name in ("first", "second")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setattr(server, "_profile_home", lambda profile: homes.get(profile))
    services = [
        FakeAccountService("first", close_fail=RuntimeError("secret cleanup marker")),
        FakeAccountService("second"),
    ]
    factory_services = list(services)
    monkeypatch.setattr(server, "_antigravity_service_factory", lambda: factory_services.pop(0))

    _rpc(server, "antigravity.accounts.list", {"profile": "first"})
    _rpc(server, "antigravity.accounts.list", {"profile": "second"})
    server._shutdown_antigravity_services()

    assert [service.close_calls for service in services] == [1, 1]
    assert "secret cleanup marker" not in caplog.text
