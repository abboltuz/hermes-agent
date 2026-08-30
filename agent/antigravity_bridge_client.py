"""OpenAI-compatible facade for the managed Antigravity bridge."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

from agent.antigravity_bridge_transport import (
    AntigravityBridgeError, AntigravityBridgeProcess, AntigravityHTTPTransport,
    resolve_antigravity_bridge_command,
)

BRIDGE_MARKER_BASE_URL = "sdkbridge://antigravity"
SUPPORTED_MODEL_FAMILIES = ("gemini", "claude")
CURATED_FALLBACK_MODELS = (
    "antigravity-gemini-3-pro",
    "antigravity-claude-sonnet-4-6",
)


def filter_antigravity_models(items: list[dict[str, Any]] | None) -> list[str] | None:
    if items is None:
        return None
    result: set[str] = set()
    for item in items:
        model_id = str(item.get("id") or "").strip() if isinstance(item, dict) else ""
        if model_id and model_id.lower().startswith((
            "gemini-", "claude-", "antigravity-gemini-", "antigravity-claude-",
        )):
            result.add(model_id)
    return sorted(result, key=lambda value: (value.lower(), value))


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


class _Completions:
    def __init__(self, client: "AntigravityBridgeClient"):
        self.client = client

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise AntigravityBridgeError("Antigravity streaming is not available in Phase A")
        payload = dict(kwargs)
        response = self.client._get_transport().json_request("POST", "/v1/chat/completions", payload)
        return _namespace(response)


class _Chat:
    def __init__(self, client: "AntigravityBridgeClient"):
        self.completions = _Completions(client)


class AntigravityBridgeClient:
    supports_abort_inflight = True

    def __init__(self, *, bridge_command: str | list[str] | None = None,
                 base_url: str | None = None, api_key: str | None = None, **_: Any):
        del base_url, api_key
        self._bridge_command = bridge_command
        self._process: AntigravityBridgeProcess | None = None
        self.__transport: AntigravityHTTPTransport | None = None
        self._lock = threading.Lock()
        self.chat = _Chat(self)
        self.is_closed = False

    def _ensure_transport(self) -> AntigravityHTTPTransport:
        if self.is_closed:
            raise AntigravityBridgeError("Antigravity bridge client is closed")
        if self.__transport is not None and self._process is not None:
            return self.__transport
        command = self._bridge_command or resolve_antigravity_bridge_command()
        if not command:
            raise AntigravityBridgeError("Antigravity bridge artifact is not installed")
        process = AntigravityBridgeProcess(command)
        endpoint = process.start()
        if self.is_closed:
            process.stop()
            raise AntigravityBridgeError("Antigravity bridge client is closed")
        self._process = process
        self.__transport = AntigravityHTTPTransport(endpoint)
        return self.__transport

    def _get_transport(self) -> AntigravityHTTPTransport:
        if self.is_closed:
            raise AntigravityBridgeError("Antigravity bridge client is closed")
        return self.__transport or self._ensure_transport()

    def list_models(self) -> list[dict[str, Any]]:
        data = self._get_transport().json_request("GET", "/v1/models")
        if isinstance(data, dict):
            data = data.get("data", [])
        return data if isinstance(data, list) else []

    def close(self) -> None:
        with self._lock:
            if self.is_closed:
                return
            self.is_closed = True
            process, self._process = self._process, None
            self.__transport = None
        if process is not None:
            process.stop()

    def abort_inflight(self) -> None:
        """Stop the owned child so a blocked request thread can unwind."""
        with self._lock:
            self.is_closed = True
            process, self._process = self._process, None
            self.__transport = None
        if process is not None:
            process.stop()


class AsyncAntigravityBridgeClient:
    supports_abort_inflight = True

    def __init__(self, **kwargs: Any):
        self._sync = AntigravityBridgeClient(**kwargs)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sync.chat.completions.create, **kwargs)

    async def list_models(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._sync.list_models)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    async def abort_inflight(self) -> None:
        await asyncio.to_thread(self._sync.abort_inflight)
