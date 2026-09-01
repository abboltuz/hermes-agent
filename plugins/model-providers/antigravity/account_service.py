"""Plugin-local account control services for the Antigravity bridge.

``AntigravityAccountService`` and ``AsyncAntigravityAccountService`` are thin,
client-owning facades.  Their account and OAuth methods pass arguments through
unchanged and return the bridge client's detached, privacy-safe snapshots and
statuses unchanged.  They do not select accounts or retain credentials.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from agent.antigravity_bridge_client import (
    AntigravityBridgeClient,
    AsyncAntigravityBridgeClient,
)


class _SyncAccountClient(Protocol):
    def list_accounts(self) -> dict[str, Any]: ...
    def set_account_enabled(self, account_id: str, enabled: bool) -> dict[str, Any]: ...
    def set_account_priority(self, account_id: str, priority: int) -> dict[str, Any]: ...
    def remove_account(self, account_id: str) -> dict[str, Any]: ...
    def start_oauth(self, project_id: str = "") -> dict[str, Any]: ...
    def poll_oauth(self, session_id: str) -> dict[str, Any]: ...
    def cancel_oauth(self, session_id: str) -> dict[str, Any]: ...
    def close(self) -> None: ...


class _AsyncAccountClient(Protocol):
    async def list_accounts(self) -> dict[str, Any]: ...
    async def set_account_enabled(self, account_id: str, enabled: bool) -> dict[str, Any]: ...
    async def set_account_priority(self, account_id: str, priority: int) -> dict[str, Any]: ...
    async def remove_account(self, account_id: str) -> dict[str, Any]: ...
    async def start_oauth(self, project_id: str = "") -> dict[str, Any]: ...
    async def poll_oauth(self, session_id: str) -> dict[str, Any]: ...
    async def cancel_oauth(self, session_id: str) -> dict[str, Any]: ...
    async def close(self) -> None: ...


class AntigravityAccountService:
    """Synchronously delegate account management to one owned bridge client."""

    def __init__(
        self,
        *,
        client: _SyncAccountClient | None = None,
        client_factory: Callable[..., _SyncAccountClient] | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            if client_factory is not None or client_kwargs:
                raise TypeError("client cannot be combined with client_factory or client options")
            self._client = client
        else:
            factory = client_factory or AntigravityBridgeClient
            self._client = factory(**client_kwargs)
        self._closed = False

    def list_accounts(self) -> dict[str, Any]:
        return self._client.list_accounts()

    def set_account_enabled(self, account_id: str, enabled: bool) -> dict[str, Any]:
        return self._client.set_account_enabled(account_id, enabled)

    def set_account_priority(self, account_id: str, priority: int) -> dict[str, Any]:
        return self._client.set_account_priority(account_id, priority)

    def remove_account(self, account_id: str) -> dict[str, Any]:
        return self._client.remove_account(account_id)

    def start_oauth(self, project_id: str = "") -> dict[str, Any]:
        return self._client.start_oauth(project_id)

    def poll_oauth(self, session_id: str) -> dict[str, Any]:
        return self._client.poll_oauth(session_id)

    def cancel_oauth(self, session_id: str) -> dict[str, Any]:
        return self._client.cancel_oauth(session_id)

    def close(self) -> None:
        """Close the owned client once, even when called from multiple exits."""
        if self._closed:
            return
        self._closed = True
        self._client.close()

    def __enter__(self) -> "AntigravityAccountService":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self.close()
        return False


class AsyncAntigravityAccountService:
    """Asynchronously delegate account management to one owned bridge client."""

    def __init__(
        self,
        *,
        client: _AsyncAccountClient | None = None,
        client_factory: Callable[..., _AsyncAccountClient] | None = None,
        **client_kwargs: Any,
    ) -> None:
        if client is not None:
            if client_factory is not None or client_kwargs:
                raise TypeError("client cannot be combined with client_factory or client options")
            self._client = client
        else:
            factory = client_factory or AsyncAntigravityBridgeClient
            self._client = factory(**client_kwargs)
        self._closed = False

    async def list_accounts(self) -> dict[str, Any]:
        return await self._client.list_accounts()

    async def set_account_enabled(self, account_id: str, enabled: bool) -> dict[str, Any]:
        return await self._client.set_account_enabled(account_id, enabled)

    async def set_account_priority(self, account_id: str, priority: int) -> dict[str, Any]:
        return await self._client.set_account_priority(account_id, priority)

    async def remove_account(self, account_id: str) -> dict[str, Any]:
        return await self._client.remove_account(account_id)

    async def start_oauth(self, project_id: str = "") -> dict[str, Any]:
        return await self._client.start_oauth(project_id)

    async def poll_oauth(self, session_id: str) -> dict[str, Any]:
        return await self._client.poll_oauth(session_id)

    async def cancel_oauth(self, session_id: str) -> dict[str, Any]:
        return await self._client.cancel_oauth(session_id)

    async def close(self) -> None:
        """Close the owned client once, including asynchronous context exits."""
        if self._closed:
            return
        self._closed = True
        await self._client.close()

    async def __aenter__(self) -> "AsyncAntigravityAccountService":
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        await self.close()
        return False
