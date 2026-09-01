"""Contract tests for the Antigravity account-control JSON-RPC adapter."""
from __future__ import annotations

import importlib
import threading
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
    with mod._antigravity_services_condition:
        mod._antigravity_services.clear()
        mod._antigravity_services_pending.clear()
        setattr(mod, "_antigravity_services_closures", 0)
        setattr(mod, "_antigravity_services_shutdown", False)
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
    assert {
        "antigravity.accounts.list",
        "antigravity.accounts.enabled",
        "antigravity.accounts.priority",
        "antigravity.accounts.remove",
        "antigravity.oauth.start",
        "antigravity.oauth.poll",
        "antigravity.oauth.cancel",
    } <= server._LONG_HANDLERS


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


def test_antigravity_rejects_path_shaped_profile_before_service_allocation(server, monkeypatch, tmp_path):
    factory_calls = []
    monkeypatch.setattr(server, "_antigravity_service_factory", lambda: factory_calls.append(True))

    for profile in (str(tmp_path), "../outside"):
        response = _rpc(server, "antigravity.accounts.list", {"profile": profile})
        assert response["error"] == {
            "code": -32602,
            "message": "invalid Antigravity parameters",
        }

    assert factory_calls == []


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


def test_shutdown_closes_every_service_and_prevents_post_shutdown_allocation(server, monkeypatch):
    old_service = FakeAccountService("old")
    new_service = FakeAccountService("new")
    services = iter((old_service, new_service))
    monkeypatch.setattr(server, "_antigravity_service_factory", lambda: next(services))
    monkeypatch.setattr(server, "_antigravity_profile_key", lambda: "profile")

    assert server._antigravity_service() is old_service
    server._shutdown_antigravity_services()
    with pytest.raises(RuntimeError):
        server._antigravity_service()
    server._shutdown_antigravity_services()

    assert [old_service.close_calls, new_service.close_calls] == [1, 0]
    assert server._antigravity_services == {}


def test_shutdown_waits_for_and_closes_an_inflight_service_allocation(server, monkeypatch):
    old_service = FakeAccountService("old")
    new_service = FakeAccountService("new")
    factory_started = threading.Event()
    release_factory = threading.Event()
    services = iter((old_service, new_service))
    errors = []

    def factory():
        service = next(services)
        if service is new_service:
            factory_started.set()
            assert release_factory.wait(timeout=5)
        return service

    monkeypatch.setattr(server, "_antigravity_service_factory", factory)
    monkeypatch.setattr(
        server,
        "_antigravity_profile_key",
        lambda: "new" if threading.current_thread().name == "allocator" else "old",
    )
    assert server._antigravity_service() is old_service

    def allocate():
        try:
            server._antigravity_service()
        except RuntimeError as exc:
            errors.append(exc)

    allocator = threading.Thread(target=allocate, name="allocator")
    allocator.start()
    assert factory_started.wait(timeout=5)
    shutdown = threading.Thread(target=server._shutdown_antigravity_services)
    shutdown.start()
    release_factory.set()
    allocator.join(timeout=5)
    shutdown.join(timeout=5)

    assert not allocator.is_alive()
    assert not shutdown.is_alive()
    assert len(errors) == 1
    assert [old_service.close_calls, new_service.close_calls] == [1, 1]
    assert server._antigravity_services == {}


def test_antigravity_dispatch_uses_pool_and_preserves_request_profile_scope(
    server, monkeypatch, tmp_path
):
    home = tmp_path / "alpha"
    home.mkdir()
    started = threading.Event()
    release = threading.Event()
    writes = []
    threads = []

    class BlockingService(FakeAccountService):
        def list_accounts(self):
            started.set()
            assert release.wait(timeout=5)
            return super().list_accounts()

    class Pool:
        def submit(self, callback):
            thread = threading.Thread(target=callback)
            threads.append(thread)
            thread.start()
            return thread

    class Transport:
        def write(self, response):
            writes.append(response)
            return True

    observed_homes = []
    monkeypatch.setattr(server, "_profile_home", lambda profile: home if profile == "alpha" else None)
    monkeypatch.setattr(
        server,
        "_antigravity_service_factory",
        lambda: observed_homes.append(server.get_hermes_home()) or BlockingService("alpha"),
    )
    monkeypatch.setattr(server, "_pool", Pool())
    server._methods["test.fast"] = lambda rid, params: {"id": rid, "result": "fast"}
    transport = Transport()

    assert server.dispatch(
        {"jsonrpc": "2.0", "id": "slow", "method": "antigravity.accounts.list", "params": {"profile": "alpha"}},
        transport,
    ) is None
    assert started.wait(timeout=5)
    assert server.dispatch({"jsonrpc": "2.0", "id": "fast", "method": "test.fast"}, transport) == {
        "id": "fast",
        "result": "fast",
    }

    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert writes == [
        {
            "jsonrpc": "2.0",
            "id": "slow",
            "result": {"accounts": {"accounts": [{"id": "alpha"}]}},
        }
    ]
    assert observed_homes == [home]
