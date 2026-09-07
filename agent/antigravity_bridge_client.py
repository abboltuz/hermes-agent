"""OpenAI-compatible facade for the managed Antigravity bridge."""
from __future__ import annotations

import asyncio
import re
import threading
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qsl, quote, urlparse

from agent.antigravity_bridge_transport import (
    AntigravityBridgeError, AntigravityBridgeProcess, AntigravityHTTPTransport,
    resolve_antigravity_bridge_command,
)

BRIDGE_MARKER_BASE_URL = "sdkbridge://antigravity"
SUPPORTED_MODEL_FAMILIES = ("gemini", "claude", "gpt-oss")
CURATED_FALLBACK_MODELS = (
    "antigravity-gemini-3.8-flash",
    "antigravity-gemini-3.7-flash",
    "antigravity-gemini-3.6-flash",
    "antigravity-gemini-3.1-pro",
    "antigravity-claude-sonnet-4-6",
    "antigravity-claude-opus-4-6-thinking",
    "antigravity-gpt-oss-120b",
)


_MAX_SAFE_INTEGER = (1 << 53) - 1
_ACCOUNT_STATUSES = frozenset({"disabled", "verification_required", "cooling_down", "limited", "available"})
_OAUTH_STATUSES = frozenset({"pending", "approved", "error", "cancelled", "expired"})
_OAUTH_ERROR_CODES = frozenset({"invalid_callback", "provider_denied", "exchange_failed", "account_save_failed", "session_not_found", "session_closed", "session_limit"})
_QUOTA_GROUPS = frozenset({"claude", "gemini-pro", "gemini-flash"})
_FORBIDDEN_AUTH_QUERY_KEYS = frozenset({"verifier", "project", "credential", "access_token", "refresh_token", "token"})
_REQUIRED_AUTH_QUERY_KEYS = frozenset({
    "client_id", "response_type", "redirect_uri", "scope", "code_challenge",
    "code_challenge_method", "state", "access_type", "prompt",
})
_MAX_AUTH_QUERY_FIELDS = len(_REQUIRED_AUTH_QUERY_KEYS)
_MAX_PUBLIC_MODEL_ID_LENGTH = 96
_PUBLIC_ANTIGRAVITY_MODEL_ID = re.compile(
    r"^(?:"
    r"(?:antigravity-)?gemini-[1-9](?:\.\d)?-(?:pro|flash|ultra|nano|lite)(?:-(?:preview(?:-customtools)?|thinking|experimental|exp|latest))?"
    r"|(?:antigravity-)?claude-(?:(?:opus|sonnet|haiku)-[1-9](?:-[1-9])?(?:-(?:latest|thinking|beta|preview))?|[1-9](?:-[1-9])?-(?:opus|sonnet|haiku)(?:-\d{8})?(?:-(?:latest|thinking|beta|preview))?)"
    r"|antigravity-gpt-oss-120b"
    r")$"
)


def _invalid_request() -> AntigravityBridgeError:
    return AntigravityBridgeError("invalid Antigravity bridge request")


def _invalid_response() -> AntigravityBridgeError:
    return AntigravityBridgeError("invalid Antigravity bridge response")


def _response_boundary(materializer):
    def wrapped(value: Any) -> Any:
        try:
            return materializer(value)
        except Exception:
            raise _invalid_response() from None
    return wrapped


def _is_canonical_uuid(value: Any) -> bool:
    if type(value) is not str or len(value) != 36 or any(value[index] != "-" for index in (8, 13, 18, 23)):
        return False
    hex_value = value.replace("-", "")
    return len(hex_value) == 32 and all(char in "0123456789abcdef" for char in hex_value) and value[14] in "12345" and value[19] in "89ab"


def _is_account_id(value: Any) -> bool:
    return type(value) is str and value.startswith("acct_") and _is_canonical_uuid(value[5:])


def _safe_int(value: Any, *, positive: bool = False) -> int | None:
    if type(value) is not int or value < (1 if positive else 0) or value > _MAX_SAFE_INTEGER:
        return None
    return value


def _safe_text(value: Any, maximum: int, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or len(value) > maximum or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise _invalid_response()
    return value


def _record(value: Any, keys: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise _invalid_response()
    return value


def _materialize_quota(value: Any) -> dict[str, dict[str, Any]]:
    if type(value) is not dict or any(type(group) is not str or group not in _QUOTA_GROUPS for group in value):
        raise _invalid_response()
    result: dict[str, dict[str, Any]] = {}
    for group, raw in value.items():
        row = _record(raw, frozenset({"remainingFraction", "resetTime", "modelCount"}))
        remaining = row["remainingFraction"]
        if type(remaining) not in (int, float) or remaining != remaining or remaining in (float("inf"), float("-inf")) or not 0 <= remaining <= 1:
            raise _invalid_response()
        model_count = _safe_int(row["modelCount"])
        if model_count is None:
            raise _invalid_response()
        result[group] = {"remaining_fraction": float(remaining), "reset_time": _safe_text(row["resetTime"], 256, nullable=True), "model_count": model_count}
    return result


@_response_boundary
def _materialize_snapshot(value: Any) -> dict[str, Any]:
    record = _record(value, frozenset({"total", "enabled", "available", "limited", "connected", "current", "accounts"}))
    current = _record(record["current"], frozenset({"claude", "gemini"}))
    if type(record["connected"]) is not bool or type(record["accounts"]) is not list:
        raise _invalid_response()
    counts = {name: _safe_int(record[name]) for name in ("total", "enabled", "available", "limited")}
    if any(item is None for item in counts.values()) or counts["total"] != len(record["accounts"]):
        raise _invalid_response()
    accounts: list[dict[str, Any]] = []
    for raw in record["accounts"]:
        row = _record(raw, frozenset({"id", "email", "enabled", "priority", "status", "addedAt", "lastUsed", "statusUntil", "quota"}))
        priority, added_at, last_used = _safe_int(row["priority"], positive=True), _safe_int(row["addedAt"]), _safe_int(row["lastUsed"])
        status_until = _safe_int(row["statusUntil"]) if row["statusUntil"] is not None else None
        if not _is_account_id(row["id"]) or type(row["enabled"]) is not bool or type(row["status"]) is not str or row["status"] not in _ACCOUNT_STATUSES or priority is None or added_at is None or last_used is None or (row["statusUntil"] is not None and status_until is None):
            raise _invalid_response()
        accounts.append({"id": row["id"], "email": _safe_text(row["email"], 320, nullable=True), "enabled": row["enabled"], "priority": priority, "status": row["status"], "added_at": added_at, "last_used": last_used, "status_until": status_until, "quota": _materialize_quota(row["quota"])})
    enabled_ids = {account["id"] for account in accounts if account["enabled"]}
    if counts != {
        "total": len(accounts),
        "enabled": len(enabled_ids),
        "available": sum(account["status"] == "available" for account in accounts),
        "limited": sum(account["status"] in {"verification_required", "cooling_down", "limited"} for account in accounts),
    } or record["connected"] is not (len(accounts) > 0):
        raise _invalid_response()
    # A verification challenge disables the account automatically. The reason
    # remains visible while disabled and during an explicit user-directed retry.
    if any(account["status"] != "verification_required"
           and (account["status"] == "disabled") is account["enabled"]
           for account in accounts):
        raise _invalid_response()
    safe_current: dict[str, str | None] = {}
    for family, account_id in current.items():
        if account_id is not None and (not _is_account_id(account_id) or account_id not in enabled_ids):
            raise _invalid_response()
        safe_current[family] = account_id
    return {**counts, "connected": record["connected"], "current": safe_current, "accounts": accounts}


@_response_boundary
def _materialize_auth_url(value: Any) -> str:
    if type(value) is not str or len(value) > 8192 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise _invalid_response()
    if any(char == "%" and (index + 2 >= len(value) or value[index + 1] not in "0123456789abcdefABCDEF" or value[index + 2] not in "0123456789abcdefABCDEF") for index, char in enumerate(value)):
        raise _invalid_response()
    parsed = urlparse(value)
    port = parsed.port
    pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    if (parsed.scheme != "https" or parsed.hostname != "accounts.google.com" or port not in (None, 443)
            or parsed.username is not None or parsed.password is not None or parsed.path != "/o/oauth2/v2/auth"
            or parsed.params or parsed.fragment or len(parsed.query) > 4096 or len(pairs) != _MAX_AUTH_QUERY_FIELDS):
        raise _invalid_response()
    query = dict(pairs)
    if (len(query) != len(pairs) or set(query) != _REQUIRED_AUTH_QUERY_KEYS
            or any(type(key) is not str or type(item) is not str or not item.strip() or len(item) > 1024 for key, item in pairs)
            or query["response_type"] != "code"
            or query["redirect_uri"] != "http://localhost:51121/oauth-callback"
            or query["code_challenge_method"] != "S256"
            or query["access_type"] != "offline" or query["prompt"] != "consent"):
        raise _invalid_response()
    return value


@_response_boundary
def _materialize_oauth_start(value: Any) -> dict[str, Any]:
    record = _record(value, frozenset({"sessionId", "authUrl", "status", "expiresAt", "pollIntervalMs"}))
    expires_at, poll_interval = _safe_int(record["expiresAt"], positive=True), _safe_int(record["pollIntervalMs"], positive=True)
    if not _is_canonical_uuid(record["sessionId"]) or record["status"] != "pending" or expires_at is None or poll_interval is None:
        raise _invalid_response()
    return {"session_id": record["sessionId"], "auth_url": _materialize_auth_url(record["authUrl"]), "flow": "browser_poll", "status": "pending", "expires_at": expires_at, "poll_interval_ms": poll_interval}


@_response_boundary
def _materialize_oauth_status(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) not in ({"status"}, {"status", "errorCode"}):
        raise _invalid_response()
    status = value.get("status")
    if type(status) is not str or status not in _OAUTH_STATUSES or ("errorCode" in value and (status != "error" or type(value["errorCode"]) is not str or value["errorCode"] not in _OAUTH_ERROR_CODES)):
        raise _invalid_response()
    return {"status": status, **({"error_code": value["errorCode"]} if "errorCode" in value else {})}


def filter_antigravity_models(items: list[dict[str, Any]] | None) -> list[str] | None:
    if items is None:
        return None
    result: set[str] = set()
    for item in items:
        model_id = item.get("id") if isinstance(item, dict) else None
        if (
            type(model_id) is str
            and len(model_id) <= _MAX_PUBLIC_MODEL_ID_LENGTH
            and _PUBLIC_ANTIGRAVITY_MODEL_ID.fullmatch(model_id.lower())
        ):
            result.add(model_id)
    return sorted(result, key=lambda value: (value.lower(), value))


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _request_timeout_seconds(value: Any, default: float = 60.0) -> float:
    candidate = getattr(value, "read", value)
    if type(candidate) in (int, float) and candidate > 0:
        return float(candidate)
    return default


class _NamespacedStream:
    def __init__(self, stream: Any):
        self._stream = stream
        self.response = getattr(stream, "response", None)

    def __iter__(self) -> "_NamespacedStream":
        return self

    def __next__(self) -> Any:
        return _namespace(next(self._stream))

    def close(self) -> None:
        self._stream.close()


class _Completions:
    def __init__(self, client: "AntigravityBridgeClient"):
        self.client = client

    def create(self, **kwargs: Any) -> Any:
        payload = dict(kwargs)
        request_timeout = _request_timeout_seconds(payload.pop("timeout", None))
        transport = self.client._get_transport()
        if payload.get("stream") is True:
            # Hermes asks OpenAI-wire providers for usage chunks, but the managed
            # bridge's strict public request schema does not expose that optional
            # SDK extension. Usage remains optional to the shared stream consumer.
            payload.pop("stream_options", None)
            return _NamespacedStream(transport.stream_request(
                "POST", "/v1/chat/completions", payload, timeout=request_timeout
            ))
        response = transport.json_request(
            "POST", "/v1/chat/completions", payload, timeout=request_timeout,
        )
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
        self._startup_condition = threading.Condition(self._lock)
        self._starting = False
        self._starting_process: AntigravityBridgeProcess | None = None
        self.chat = _Chat(self)
        self.is_closed = False

    def _ensure_transport(self) -> AntigravityHTTPTransport:
        with self._startup_condition:
            while True:
                if self.is_closed:
                    raise AntigravityBridgeError("Antigravity bridge client is closed")
                if self.__transport is not None and self._process is not None:
                    return self.__transport
                if not self._starting:
                    self._starting = True
                    break
                self._startup_condition.wait()

        command = self._bridge_command or resolve_antigravity_bridge_command()
        if not command:
            with self._startup_condition:
                self._starting = False
                self._startup_condition.notify_all()
            raise AntigravityBridgeError("Antigravity bridge artifact is not installed")
        process = AntigravityBridgeProcess(command)
        with self._startup_condition:
            self._starting_process = process
            if self.is_closed:
                self._starting_process = None
                self._starting = False
                self._startup_condition.notify_all()
                process.stop()
                raise AntigravityBridgeError("Antigravity bridge client is closed")
        try:
            endpoint = process.start()
            transport = AntigravityHTTPTransport(endpoint)
        except Exception:
            process.stop()
            with self._startup_condition:
                if self._starting_process is process:
                    self._starting_process = None
                self._starting = False
                self._startup_condition.notify_all()
            raise
        with self._startup_condition:
            self._starting_process = None
            self._starting = False
            if self.is_closed:
                self._startup_condition.notify_all()
                process.stop()
                raise AntigravityBridgeError("Antigravity bridge client is closed")
            self._process = process
            self.__transport = transport
            self._startup_condition.notify_all()
            return transport

    def _get_transport(self) -> AntigravityHTTPTransport:
        with self._lock:
            if self.is_closed:
                raise AntigravityBridgeError("Antigravity bridge client is closed")
        return self._ensure_transport()

    def list_models(self) -> list[dict[str, Any]]:
        data = self._get_transport().json_request("GET", "/v1/models")
        if isinstance(data, dict):
            data = data.get("data", [])
        return data if isinstance(data, list) else []

    def list_accounts(self) -> dict[str, Any]:
        return _materialize_snapshot(self._get_transport().json_request("GET", "/v1/accounts"))

    def set_account_enabled(self, account_id: str, enabled: bool) -> dict[str, Any]:
        if not _is_account_id(account_id) or not isinstance(enabled, bool):
            raise _invalid_request()
        path = f"/v1/accounts/{quote(account_id, safe='')}/enabled"
        return _materialize_snapshot(self._get_transport().json_request("PUT", path, {"enabled": enabled}))

    def set_account_priority(self, account_id: str, priority: int) -> dict[str, Any]:
        if not _is_account_id(account_id) or _safe_int(priority, positive=True) is None:
            raise _invalid_request()
        path = f"/v1/accounts/{quote(account_id, safe='')}/priority"
        return _materialize_snapshot(self._get_transport().json_request("PUT", path, {"priority": priority}))

    def remove_account(self, account_id: str) -> dict[str, Any]:
        if not _is_account_id(account_id):
            raise _invalid_request()
        return _materialize_snapshot(self._get_transport().json_request("DELETE", f"/v1/accounts/{quote(account_id, safe='')}"))

    def start_oauth(self, project_id: str = "") -> dict[str, Any]:
        if not isinstance(project_id, str) or len(project_id) > 4096 or any(ord(char) < 32 or ord(char) == 127 for char in project_id):
            raise _invalid_request()
        payload = {"projectId": project_id} if project_id else {}
        return _materialize_oauth_start(self._get_transport().json_request("POST", "/v1/accounts/oauth/start", payload))

    def poll_oauth(self, session_id: str) -> dict[str, Any]:
        if not _is_canonical_uuid(session_id):
            raise _invalid_request()
        return _materialize_oauth_status(self._get_transport().json_request("GET", f"/v1/accounts/oauth/{quote(session_id, safe='')}"))

    def cancel_oauth(self, session_id: str) -> dict[str, Any]:
        if not _is_canonical_uuid(session_id):
            raise _invalid_request()
        return _materialize_oauth_status(self._get_transport().json_request("DELETE", f"/v1/accounts/oauth/{quote(session_id, safe='')}"))

    def close(self) -> None:
        with self._startup_condition:
            if self.is_closed:
                return
            self.is_closed = True
            process, self._process = self._process, None
            starting_process = self._starting_process
            self._starting_process = None
            self.__transport = None
            self._startup_condition.notify_all()
        if process is not None:
            process.stop()
        if starting_process is not None and starting_process is not process:
            starting_process.stop()

    def abort_inflight(self) -> None:
        """Stop the owned child so a blocked request thread can unwind."""
        with self._startup_condition:
            self.is_closed = True
            process, self._process = self._process, None
            starting_process = self._starting_process
            self._starting_process = None
            self.__transport = None
            self._startup_condition.notify_all()
        if process is not None:
            process.stop()
        if starting_process is not None and starting_process is not process:
            starting_process.stop()


class AsyncAntigravityBridgeClient:
    supports_abort_inflight = True

    def __init__(self, sync_client: AntigravityBridgeClient | None = None, **kwargs: Any):
        # Conversion transfers the existing client's lifecycle to this wrapper;
        # do not spawn a second bridge or discard an explicit bridge command.
        self._sync = sync_client if sync_client is not None else AntigravityBridgeClient(**kwargs)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        task = asyncio.create_task(
            asyncio.to_thread(self._sync.chat.completions.create, **kwargs)
        )
        try:
            return await task
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.to_thread(self._sync.abort_inflight))
            raise

    async def list_models(self) -> list[dict[str, Any]]:
        return await self._run_sync(self._sync.list_models)

    async def _run_sync(self, method: Any, *args: Any) -> Any:
        task = asyncio.create_task(asyncio.to_thread(method, *args))
        try:
            return await task
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.to_thread(self._sync.abort_inflight))
            raise

    async def list_accounts(self) -> dict[str, Any]:
        return await self._run_sync(self._sync.list_accounts)

    async def set_account_enabled(self, account_id: str, enabled: bool) -> dict[str, Any]:
        return await self._run_sync(self._sync.set_account_enabled, account_id, enabled)

    async def set_account_priority(self, account_id: str, priority: int) -> dict[str, Any]:
        return await self._run_sync(self._sync.set_account_priority, account_id, priority)

    async def remove_account(self, account_id: str) -> dict[str, Any]:
        return await self._run_sync(self._sync.remove_account, account_id)

    async def start_oauth(self, project_id: str = "") -> dict[str, Any]:
        return await self._run_sync(self._sync.start_oauth, project_id)

    async def poll_oauth(self, session_id: str) -> dict[str, Any]:
        return await self._run_sync(self._sync.poll_oauth, session_id)

    async def cancel_oauth(self, session_id: str) -> dict[str, Any]:
        return await self._run_sync(self._sync.cancel_oauth, session_id)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    def abort_inflight(self) -> None:
        self._sync.abort_inflight()
