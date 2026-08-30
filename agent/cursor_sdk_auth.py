"""Interactive Cursor SDK login for Hermes (`hermes cursor login`).

Implements the same browser PKCE flow the Cursor SDK's ``Cursor.auth.login()``
ships (sdk 1.0.27+): generate a one-time ``verifier``, send the user's browser
to ``cursor.com/loginDeepControl`` with ``challenge = sha256(verifier)``, poll
``api2.cursor.sh/auth/poll`` until the browser completes, then use the session
token **once** to mint a named, expiring user API key via
``aiserver.v1.DashboardService/CreateUserApiKey`` — and drop the session
tokens.  The minted key is the only credential persisted.

Credentials are stored in the active profile's private store
(``$HERMES_HOME/cursor/auth.json``, 0600). This keeps profiles isolated;
explicit ``CURSOR_API_KEY`` compatibility imports remain in the environment.

The login URL can be opened on ANY device logged into cursor.com (it is a
device-code-style flow): only the process holding the verifier can redeem it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import stat
import threading
import time
import urllib.error
import urllib.request
import uuid as uuid_module
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_WEBSITE_URL = "https://cursor.com"
DEFAULT_BACKEND_URL = "https://api2.cursor.sh"
# Matches the SDK's DEFAULT_LOGIN_API_KEY_TTL_MS: 90 days.
DEFAULT_API_KEY_TTL_MS = 90 * 24 * 60 * 60 * 1000

_POLL_MAX_ATTEMPTS = 150
_POLL_BASE_DELAY_S = 1.0
_POLL_MAX_DELAY_S = 10.0
_POLL_BACKOFF = 1.2
_MAX_CONSECUTIVE_ERRORS = 3
_MAX_AUTH_RESPONSE_BYTES = 1 * 1024 * 1024
_MAX_CREDENTIAL_FILE_BYTES = 64 * 1024


class CursorAuthError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CursorAuthError("unexpected Cursor authentication redirect refused")


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


_APPROVED_WEBSITE_HOSTS = frozenset({"cursor.com", "www.cursor.com"})
_APPROVED_BACKEND_HOSTS = frozenset({"api2.cursor.sh", "api.cursor.sh"})


def _approved_base_url(url: str, *, hosts: frozenset[str], label: str) -> str:
    """Validate a Cursor service base URL before it can receive credentials."""
    candidate = (url or "").strip().rstrip("/")
    parsed = urlparse(candidate)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in hosts
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port is not None
        or parsed.path not in {"", "/"}
    ):
        raise CursorAuthError(f"unsupported Cursor {label} host")
    return f"https://{parsed.hostname}"


def resolve_website_url(url: str = "") -> str:
    return _approved_base_url(
        url or os.getenv("CURSOR_WEBSITE_URL", "") or DEFAULT_WEBSITE_URL,
        hosts=_APPROVED_WEBSITE_HOSTS,
        label="website",
    )


def resolve_backend_url(url: str = "") -> str:
    return _approved_base_url(
        url or os.getenv("CURSOR_BACKEND_URL", "") or DEFAULT_BACKEND_URL,
        hosts=_APPROVED_BACKEND_HOSTS,
        label="backend",
    )


# ── Handshake ─────────────────────────────────────────────────────────────


@dataclass
class LoginHandshake:
    uuid: str
    verifier: str
    login_url: str


def create_login_handshake(website_url: str = "") -> LoginHandshake:
    website_url = resolve_website_url(website_url)
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    login_uuid = str(uuid_module.uuid4())
    login_url = (
        f"{website_url}/loginDeepControl?challenge={challenge}"
        f"&uuid={login_uuid}&mode=login&redirectTarget=sdk"
    )
    return LoginHandshake(uuid=login_uuid, verifier=verifier, login_url=login_url)


# ── Poll ──────────────────────────────────────────────────────────────────


def poll_for_login_tokens(
    *,
    api_url: str,
    uuid: str,
    verifier: str,
    on_status: Callable[[str], None] | None = None,
    max_attempts: int = _POLL_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> dict[str, str] | None:
    """Poll ``/auth/poll`` until the browser completes the login.

    POST with the verifier in the JSON body (it is redeemable — it must not
    reach a URL or access log).  Backends predating the POST route answer
    with route-not-found; those get one sticky fallback to GET.  A 404 with
    any other body means the login is still pending.

    Returns ``{"accessToken", "refreshToken"}`` or None on timeout/errors.
    """
    consecutive_errors = 0
    for attempt in range(max_attempts):
        if cancel_event is not None and cancel_event.is_set():
            return None
        if deadline is not None and time.time() >= deadline:
            return None
        delay = min(_POLL_BASE_DELAY_S * (_POLL_BACKOFF ** attempt), _POLL_MAX_DELAY_S)
        try:
            request = urllib.request.Request(
                f"{api_url}/auth/poll",
                method="POST",
                data=json.dumps({"uuid": uuid, "verifier": verifier}).encode(),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            try:
                with _NO_REDIRECT_OPENER.open(request, timeout=15) as response:
                    raw = response.read(_MAX_AUTH_RESPONSE_BYTES + 1)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    # A pending login is represented by a 404; never retry via
                    # GET because the verifier is redeemable and must not enter a URL.
                    consecutive_errors = 0
                    if cancel_event is None:
                        sleep(delay)
                    else:
                        cancel_event.wait(max(0.0, min(delay, deadline - time.time()) if deadline else delay))
                    continue
                raise
            if len(raw) > _MAX_AUTH_RESPONSE_BYTES:
                raise CursorAuthError("Cursor login response exceeds size limit")
            consecutive_errors = 0
            result = json.loads(raw.decode("utf-8"))
            if (
                isinstance(result, dict)
                and isinstance(result.get("accessToken"), str)
                and isinstance(result.get("refreshToken"), str)
            ):
                return {
                    "accessToken": result["accessToken"],
                    "refreshToken": result["refreshToken"],
                }
            return None
        except Exception as exc:
            consecutive_errors += 1
            # Do not log exception text: urllib errors can include the full
            # request URL and future SDK errors may echo request fields.
            logger.debug("auth/poll attempt %d failed (%s)", attempt, type(exc).__name__)
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                return None
            if cancel_event is None:
                sleep(delay)
            else:
                cancel_event.wait(max(0.0, min(delay, deadline - time.time()) if deadline else delay))
    return None


# ── Mint ──────────────────────────────────────────────────────────────────


def _dashboard_rpc(
    backend_url: str, method: str, payload: dict[str, Any], access_token: str
) -> dict[str, Any]:
    """Connect unary POST to aiserver.v1.DashboardService with JSON encoding."""
    request = urllib.request.Request(
        f"{backend_url}/aiserver.v1.DashboardService/{method}",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
            "Connect-Protocol-Version": "1",
        },
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=30) as response:
            raw = response.read(_MAX_AUTH_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as err:
        raise CursorAuthError(
            f"DashboardService/{method} failed (HTTP {err.code}); response redacted"
        ) from err
    if len(raw) > _MAX_AUTH_RESPONSE_BYTES:
        raise CursorAuthError(f"DashboardService/{method}: response exceeds size limit")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise CursorAuthError(f"DashboardService/{method}: invalid JSON response") from exc
    return parsed if isinstance(parsed, dict) else {}


def mint_user_api_key(
    *,
    backend_url: str,
    access_token: str,
    name: str,
    expires_at_ms: int | None = None,
) -> str:
    """Mint a named user API key; the session token is used once and dropped."""
    payload: dict[str, Any] = {"name": name}
    if expires_at_ms is not None:
        # proto3 JSON encodes int64 as a string.
        payload["expiresAt"] = str(int(expires_at_ms))
    response = _dashboard_rpc(backend_url, "CreateUserApiKey", payload, access_token)
    api_key = response.get("apiKey")
    if not isinstance(api_key, str) or not api_key:
        raise CursorAuthError(
            "Login succeeded but CreateUserApiKey returned no key — your team's "
            "settings may restrict user API keys; ask a team admin, or add an "
            "existing CURSOR_API_KEY to ~/.hermes/.env instead."
        )
    return api_key


def get_login_email(backend_url: str, access_token: str) -> str:
    """Best-effort GetMe for a friendly status line."""
    try:
        response = _dashboard_rpc(backend_url, "GetMe", {}, access_token)
        email = response.get("email")
        return email if isinstance(email, str) else ""
    except Exception:
        return ""


# ── Profile-scoped credential store ───────────────────────────────────────


def sdk_auth_path() -> Path:
    """Return the active profile's private Cursor credential store."""
    return get_hermes_home() / "cursor" / "auth.json"


def read_sdk_credentials() -> dict[str, Any] | None:
    """Return valid stored credentials, or None (missing/foreign/expired)."""
    path = sdk_auth_path()
    try:
        if path.is_symlink():
            return None
        if os.name == "nt":
            raw = path.read_bytes()[: _MAX_CREDENTIAL_FILE_BYTES + 1]
        else:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            try:
                file_stat = os.fstat(fd)
                if (
                    not stat.S_ISREG(file_stat.st_mode)
                    or file_stat.st_uid != os.getuid()
                    or file_stat.st_mode & 0o077
                    or file_stat.st_nlink != 1
                ):
                    return None
                raw = os.read(fd, _MAX_CREDENTIAL_FILE_BYTES + 1)
            finally:
                os.close(fd)
        if len(raw) > _MAX_CREDENTIAL_FILE_BYTES:
            return None
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or parsed.get("version") != 1:
        return None
    api_key = parsed.get("apiKey")
    if not isinstance(api_key, str) or not api_key:
        return None
    expires = parsed.get("apiKeyExpiresAtMs")
    if isinstance(expires, (int, float)) and expires <= time.time() * 1000:
        return None
    return parsed


def save_sdk_credentials(
    *,
    backend_url: str,
    api_key: str,
    api_key_expires_at_ms: int | None = None,
    email: str = "",
    source: str = "cursor_login",
) -> Path:
    from utils import atomic_json_write

    backend_url = resolve_backend_url(backend_url)
    path = sdk_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CursorAuthError("Cursor credential path is a symlink; refusing to overwrite")
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    payload: dict[str, Any] = {
        "version": 1,
        "backendUrl": backend_url,
        "apiKey": api_key,
        "createdAtMs": int(time.time() * 1000),
    }
    if api_key_expires_at_ms is not None:
        payload["apiKeyExpiresAtMs"] = int(api_key_expires_at_ms)
    if email:
        payload["email"] = email
    payload["source"] = source
    atomic_json_write(path, payload, indent=2, mode=0o600)
    return path


def clear_sdk_credentials() -> bool:
    try:
        sdk_auth_path().unlink()
        return True
    except OSError:
        return False


# ── Top-level login ───────────────────────────────────────────────────────


def _default_api_key_name() -> str:
    try:
        host = socket.gethostname() or "unknown-host"
    except OSError:
        host = "unknown-host"
    return f"Hermes Agent ({host})"


def login(
    *,
    on_login_url: Callable[[str], None],
    on_status: Callable[[str], None] | None = None,
    api_key_name: str = "",
    api_key_ttl_ms: int = DEFAULT_API_KEY_TTL_MS,
    backend_url: str = "",
    website_url: str = "",
    open_browser: bool = True,
) -> dict[str, Any]:
    """Run the full interactive login; returns the stored credential payload.

    The caller receives the login URL via ``on_login_url`` and may show it
    anywhere — the user can complete it on another device.
    """
    backend_url = resolve_backend_url(backend_url)
    handshake = create_login_handshake(website_url)
    on_login_url(handshake.login_url)
    if open_browser and not os.getenv("NO_OPEN_BROWSER") and not os.getenv("SSH_CONNECTION"):
        try:
            import webbrowser

            webbrowser.open(handshake.login_url)
        except Exception:
            pass

    if on_status:
        on_status("Waiting for the browser login to complete...")
    tokens = poll_for_login_tokens(
        api_url=backend_url,
        uuid=handshake.uuid,
        verifier=handshake.verifier,
        on_status=on_status,
    )
    if tokens is None:
        raise CursorAuthError(
            "Login did not complete (timed out or was cancelled). Run "
            "`hermes cursor login` to try again."
        )

    expires_at_ms = int(time.time() * 1000) + int(api_key_ttl_ms)
    api_key = mint_user_api_key(
        backend_url=backend_url,
        access_token=tokens["accessToken"],
        name=api_key_name or _default_api_key_name(),
        expires_at_ms=expires_at_ms,
    )
    email = get_login_email(backend_url, tokens["accessToken"])
    save_sdk_credentials(
        backend_url=backend_url,
        api_key=api_key,
        api_key_expires_at_ms=expires_at_ms,
        email=email,
        source="cursor_login",
    )
    return {
        "email": email,
        "apiKeyExpiresAtMs": expires_at_ms,
        "path": str(sdk_auth_path()),
    }


def resolve_cursor_api_key() -> tuple[str, str]:
    """Resolve the Cursor credential: explicit env/.env key, else SDK login.

    Returns ``(api_key, source)`` where source is ``"env"``, the stored
    credential's source label, or ``("", "")`` when nothing usable exists.
    """
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv

        env_key = (get_env_value_prefer_dotenv("CURSOR_API_KEY") or "").strip()
        if env_key:
            return env_key, "env"
    except Exception:
        env_key = os.getenv("CURSOR_API_KEY", "").strip()
        if env_key:
            return env_key, "env"
    stored = read_sdk_credentials()
    if stored:
        return str(stored["apiKey"]), str(stored.get("source") or "cursor_login")
    return "", ""
