"""Behavioral contract for the plugin-local Antigravity account services."""
from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from agent.antigravity_bridge_transport import AntigravityBridgeError


ACCOUNT_ID = "acct_123e4567-e89b-12d3-a456-426614174000"
SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"


def _service_module():
    # Provider discovery loads the hyphenated plugin directory under the
    # importable ``plugins.model_providers`` package namespace.
    import model_tools  # noqa: F401
    module = importlib.import_module("plugins.model_providers.antigravity.account_service")
    return module.AntigravityAccountService, module.AsyncAntigravityAccountService


class FakeSyncClient:
    def __init__(self):
        self.calls: list[tuple[Any, ...]] = []
        self.close_calls = 0
        self.snapshot: dict[str, Any] = {}
        self.oauth_start: dict[str, Any] = {}
        self.oauth_status: dict[str, Any] = {}

    def list_accounts(self):
        self.calls.append(("list_accounts",))
        return self.snapshot

    def set_account_enabled(self, account_id, enabled):
        self.calls.append(("set_account_enabled", account_id, enabled))
        return self.snapshot

    def set_account_priority(self, account_id, priority):
        self.calls.append(("set_account_priority", account_id, priority))
        return self.snapshot

    def remove_account(self, account_id):
        self.calls.append(("remove_account", account_id))
        return self.snapshot

    def start_oauth(self, project_id=""):
        self.calls.append(("start_oauth", project_id))
        return self.oauth_start

    def poll_oauth(self, session_id):
        self.calls.append(("poll_oauth", session_id))
        return self.oauth_status

    def cancel_oauth(self, session_id):
        self.calls.append(("cancel_oauth", session_id))
        return self.oauth_status

    def close(self):
        self.close_calls += 1


class FakeAsyncClient:
    def __init__(self):
        self.calls: list[tuple[Any, ...]] = []
        self.close_calls = 0
        self.abort_calls = 0
        self.snapshot: dict[str, Any] = {}
        self.oauth_start: dict[str, Any] = {}
        self.oauth_status: dict[str, Any] = {}

    async def list_accounts(self):
        self.calls.append(("list_accounts",))
        return self.snapshot

    async def set_account_enabled(self, account_id, enabled):
        self.calls.append(("set_account_enabled", account_id, enabled))
        return self.snapshot

    async def set_account_priority(self, account_id, priority):
        self.calls.append(("set_account_priority", account_id, priority))
        return self.snapshot

    async def remove_account(self, account_id):
        self.calls.append(("remove_account", account_id))
        return self.snapshot

    async def start_oauth(self, project_id=""):
        self.calls.append(("start_oauth", project_id))
        return self.oauth_start

    async def poll_oauth(self, session_id):
        self.calls.append(("poll_oauth", session_id))
        return self.oauth_status

    async def cancel_oauth(self, session_id):
        self.calls.append(("cancel_oauth", session_id))
        return self.oauth_status

    async def close(self):
        self.close_calls += 1

    def abort_inflight(self):
        self.abort_calls += 1


@pytest.fixture
def safe_values():
    return {
        "snapshot": {"accounts": []},
        "oauth_start": {"session_id": SESSION_ID, "status": "pending"},
        "oauth_status": {"status": "cancelled"},
    }


def _set_values(client, values):
    client.snapshot = values["snapshot"]
    client.oauth_start = values["oauth_start"]
    client.oauth_status = values["oauth_status"]
    return client


def test_sync_service_delegates_exactly_and_returns_client_snapshots_unchanged(safe_values):
    AntigravityAccountService, _ = _service_module()
    client = _set_values(FakeSyncClient(), safe_values)
    service = AntigravityAccountService(client=client)

    assert service.list_accounts() is safe_values["snapshot"]
    assert service.set_account_enabled(ACCOUNT_ID, True) is safe_values["snapshot"]
    assert service.set_account_priority(ACCOUNT_ID, 2) is safe_values["snapshot"]
    assert service.remove_account(ACCOUNT_ID) is safe_values["snapshot"]
    assert service.start_oauth("project") is safe_values["oauth_start"]
    assert service.poll_oauth(SESSION_ID) is safe_values["oauth_status"]
    assert service.cancel_oauth(SESSION_ID) is safe_values["oauth_status"]
    assert client.calls == [
        ("list_accounts",),
        ("set_account_enabled", ACCOUNT_ID, True),
        ("set_account_priority", ACCOUNT_ID, 2),
        ("remove_account", ACCOUNT_ID),
        ("start_oauth", "project"),
        ("poll_oauth", SESSION_ID),
        ("cancel_oauth", SESSION_ID),
    ]


def test_sync_service_closes_one_injected_client_once_on_repeated_close_and_context_exit(safe_values):
    AntigravityAccountService, _ = _service_module()
    client = _set_values(FakeSyncClient(), safe_values)
    service = AntigravityAccountService(client=client)

    with pytest.raises(RuntimeError, match="context failure"):
        with service:
            raise RuntimeError("context failure")
    service.close()
    service.close()

    assert client.close_calls == 1


def test_sync_service_factory_owns_one_client_and_does_not_create_another(safe_values):
    AntigravityAccountService, _ = _service_module()
    client = _set_values(FakeSyncClient(), safe_values)
    factory_calls = []

    def factory(*, bridge_command):
        factory_calls.append(bridge_command)
        return client

    service = AntigravityAccountService(client_factory=factory, bridge_command=["bridge"])
    assert service.list_accounts() is safe_values["snapshot"]
    service.close()

    assert factory_calls == [["bridge"]]
    assert client.close_calls == 1


def test_sync_constructor_rejects_ambiguous_ownership_without_running_factory(safe_values):
    AntigravityAccountService, _ = _service_module()
    client = _set_values(FakeSyncClient(), safe_values)
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return client

    with pytest.raises(TypeError, match="client cannot be combined"):
        AntigravityAccountService(client=client, client_factory=factory)

    assert factory_calls == []
    assert client.close_calls == 0


def test_sync_factory_error_passes_through_without_constructing_a_service():
    AntigravityAccountService, _ = _service_module()
    error = AntigravityBridgeError("safe bridge error")

    def factory():
        raise error

    with pytest.raises(AntigravityBridgeError) as raised:
        AntigravityAccountService(client_factory=factory)

    assert raised.value is error


def test_sync_service_preserves_bridge_error_and_invalid_client_input_without_extra_work(safe_values):
    AntigravityAccountService, _ = _service_module()
    client = _set_values(FakeSyncClient(), safe_values)
    error = AntigravityBridgeError("safe bridge error")

    def fail(account_id, enabled):
        client.calls.append(("set_account_enabled", account_id, enabled))
        raise error

    client.set_account_enabled = fail
    service = AntigravityAccountService(client=client)

    with pytest.raises(AntigravityBridgeError) as raised:
        service.set_account_enabled("not-an-account", True)

    assert raised.value is error
    assert client.calls == [("set_account_enabled", "not-an-account", True)]


@pytest.mark.asyncio
async def test_async_service_has_exact_awaitable_delegation_and_single_async_cleanup(safe_values):
    _, AsyncAntigravityAccountService = _service_module()
    client = _set_values(FakeAsyncClient(), safe_values)
    service = AsyncAntigravityAccountService(client=client)

    assert await service.list_accounts() is safe_values["snapshot"]
    assert await service.set_account_enabled(ACCOUNT_ID, False) is safe_values["snapshot"]
    assert await service.set_account_priority(ACCOUNT_ID, 3) is safe_values["snapshot"]
    assert await service.remove_account(ACCOUNT_ID) is safe_values["snapshot"]
    assert await service.start_oauth() is safe_values["oauth_start"]
    assert await service.poll_oauth(SESSION_ID) is safe_values["oauth_status"]
    assert await service.cancel_oauth(SESSION_ID) is safe_values["oauth_status"]
    async with service:
        pass
    await service.close()

    assert client.calls == [
        ("list_accounts",),
        ("set_account_enabled", ACCOUNT_ID, False),
        ("set_account_priority", ACCOUNT_ID, 3),
        ("remove_account", ACCOUNT_ID),
        ("start_oauth", ""),
        ("poll_oauth", SESSION_ID),
        ("cancel_oauth", SESSION_ID),
    ]
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_async_cancellation_is_delegated_once_to_client_abort_semantics(safe_values):
    _, AsyncAntigravityAccountService = _service_module()
    client = _set_values(FakeAsyncClient(), safe_values)
    entered = asyncio.Event()

    async def cancelled(session_id):
        client.calls.append(("cancel_oauth", session_id))
        entered.set()
        await asyncio.Event().wait()

    client.cancel_oauth = cancelled
    service = AsyncAntigravityAccountService(client=client)
    task = asyncio.create_task(service.cancel_oauth(SESSION_ID))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.calls == [("cancel_oauth", SESSION_ID)]
    assert client.abort_calls == 0


def test_service_import_keeps_antigravity_provider_registry_contract():
    AntigravityAccountService, _ = _service_module()
    from providers import get_provider_profile

    profile = get_provider_profile("antigravity")
    assert AntigravityAccountService.__module__.endswith("antigravity.account_service")
    assert profile is not None
    assert profile.name == "antigravity"
    assert profile.base_url == "sdkbridge://antigravity"
