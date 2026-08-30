"""Cursor SDK bridge process management + Connect JSON transport.

Implements the `sdk.v1` bridge protocol from https://github.com/cursor/sdk-bridge:

* locate or install the ``cursor-sdk-bridge`` launcher,
* spawn it with ``CURSOR_API_KEY`` and parse the ``cursor-sdk-bridge ready``
  stderr discovery line,
* speak Connect over HTTP/1.1 with JSON encoding — unary RPCs are plain JSON
  POSTs, server streams use enveloped ``application/connect+json`` frames
  (1 flags byte + 4-byte big-endian length + payload).

The bridge is HTTP/1.1 only; classic gRPC clients do not work.  JSON encoding
is used so Hermes does not need generated protobuf stubs (proto3 has a
canonical JSON mapping and Connect servers accept it natively).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import stat
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

READY_LINE_PREFIX = "cursor-sdk-bridge ready "
_STARTUP_TIMEOUT_SECONDS = 45.0
_SHUTDOWN_TIMEOUT_SECONDS = 5.0
_MAX_RESPONSE_BYTES = 1 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_AUTH_TOKEN_BYTES = 4096

# Default bridge version installed by `hermes model` when no bridge is found.
# Matches a released @cursor/sdk / cursor-sdk version that includes the
# custom-tools callback surface (landed in SDK 1.0.2x, Aug 2026).  Users can
# override via config.yaml `cursor_bridge.download_version`.
DEFAULT_BRIDGE_VERSION = "1.0.27"
# Primary distribution: exact GitHub release assets.  Digests are pinned in
# source so a compromised release manifest cannot authorize different bytes.
_GITHUB_RELEASE_URL_TEMPLATE = (
    "https://github.com/cursor/sdk-bridge/releases/download/v{version}/"
    "cursor-sdk-bridge-standalone-{os}-{arch}.tar.gz"
)
_TRUSTED_ARCHIVE_SHA256 = {
    ("1.0.27", "darwin", "arm64"): "0d544fd30d5c0f93cb8ade0cb5fbfbea10cf2e55c4669f74f26dd36d6a5bb0ba",
    ("1.0.27", "darwin", "x64"): "f8d6be39cc379420746cc7d09adad51843d095802a9af872858ad5fb3304b1f6",
    ("1.0.27", "linux", "arm64"): "6d2e7b12875003045923d038a56df06b309e80a7d2500a4c5261d5511b1db40c",
    ("1.0.27", "linux", "x64"): "114e6b7b284c31006979e8b4340c66ba60015d2143f62d76ce4d7584f80068c2",
    ("1.0.27", "win32", "x64"): "c373c01da4a8808137cf8578adc6d7f5a4a8a9f0bf5dd288fbba9b6b3e62d9b3",
}


class CursorBridgeError(RuntimeError):
    """Raised for bridge process, transport, or Connect-level failures."""

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class BridgeEndpoint:
    """Where a running bridge listens and how to authenticate to it."""

    url: str
    auth_token: str
    server_version: str = ""
    pid: int | None = None
    workspace_ref: str = ""
    state_root: str = ""


def parse_ready_line(line: str) -> dict[str, Any] | None:
    """Parse one stderr line; return the discovery payload or None.

    Never log the returned payload verbatim — older bridges inline the auth
    token in the discovery JSON.
    """
    if not line.startswith(READY_LINE_PREFIX):
        return None
    raw = line[len(READY_LINE_PREFIX):].strip()
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise CursorBridgeError(f"invalid bridge discovery JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CursorBridgeError("bridge discovery payload is not a JSON object")
    return payload


def validate_discovery(payload: dict[str, Any]) -> None:
    if payload.get("schemaVersion") != 1:
        raise CursorBridgeError(
            f"unsupported bridge discovery schemaVersion={payload.get('schemaVersion')!r}"
        )
    if payload.get("transport") != "tcp":
        raise CursorBridgeError(
            f"unsupported bridge transport={payload.get('transport')!r}"
        )
    if payload.get("protocol") != "connect":
        raise CursorBridgeError(
            f"unsupported bridge protocol={payload.get('protocol')!r}"
        )


def _read_secure_auth_token_file(token_file: str) -> str:
    """Read the bridge's ephemeral token without following a final symlink.

    The official bridge intentionally creates this file in a private
    ``/tmp/cursor-sdk-bridge-*`` directory, not under ``HERMES_HOME``.  Trust is
    therefore based on file type, ownership, permissions, and a no-follow
    descriptor rather than on a project-specific path prefix.
    """
    token_path = Path(token_file).expanduser()
    if not token_path.is_absolute():
        raise CursorBridgeError("bridge auth token file path is not absolute")

    if os.name == "nt":
        if token_path.is_symlink() or not token_path.is_file():
            raise CursorBridgeError("bridge auth token path is not a regular file")
        try:
            raw = token_path.read_bytes()[: _MAX_AUTH_TOKEN_BYTES + 1]
        except OSError as exc:
            raise CursorBridgeError(f"could not read bridge auth token file: {exc}") from exc
    else:
        try:
            parent_stat = token_path.parent.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(parent_stat.st_mode)
                or parent_stat.st_uid != os.getuid()
                or parent_stat.st_mode & 0o077
            ):
                raise CursorBridgeError(
                    "bridge auth token directory ownership or permissions are unsafe"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(token_path, flags)
            try:
                token_stat = os.fstat(fd)
                if (
                    not stat.S_ISREG(token_stat.st_mode)
                    or token_stat.st_uid != os.getuid()
                    or token_stat.st_mode & 0o077
                    or token_stat.st_nlink != 1
                ):
                    raise CursorBridgeError(
                        "bridge auth token file ownership or permissions are unsafe"
                    )
                raw = os.read(fd, _MAX_AUTH_TOKEN_BYTES + 1)
            finally:
                os.close(fd)
        except CursorBridgeError:
            raise
        except OSError as exc:
            raise CursorBridgeError(f"could not read bridge auth token file: {exc}") from exc

    if len(raw) > _MAX_AUTH_TOKEN_BYTES:
        raise CursorBridgeError("bridge auth token file exceeds size limit")
    try:
        token = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise CursorBridgeError("bridge auth token file is not UTF-8") from exc
    if not token:
        raise CursorBridgeError("bridge auth token is empty")
    return token


def endpoint_from_discovery(payload: dict[str, Any]) -> BridgeEndpoint:
    validate_discovery(payload)
    url = str(payload.get("url") or "").strip()
    if not url:
        host = str(payload.get("host") or "").strip()
        port = payload.get("port")
        if not host or not isinstance(port, int):
            raise CursorBridgeError("bridge discovery has no url/host/port")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        url = f"http://{host}:{port}"

    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise CursorBridgeError("bridge endpoint must be loopback HTTP")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CursorBridgeError("bridge endpoint contains unsafe URL components")
    if parsed.port is None or not (1 <= parsed.port <= 65535):
        raise CursorBridgeError("bridge endpoint has invalid port")

    token = str(payload.get("authToken") or "").strip()
    if not token:
        token_file = str(payload.get("authTokenFile") or "").strip()
        if not token_file:
            raise CursorBridgeError("bridge discovery has no authTokenFile")
        token = _read_secure_auth_token_file(token_file)

    return BridgeEndpoint(
        url=url,
        auth_token=token,
        server_version=str(payload.get("serverVersion") or ""),
        pid=payload.get("pid") if isinstance(payload.get("pid"), int) else None,
        workspace_ref=str(payload.get("workspaceRef") or ""),
        state_root=str(payload.get("stateRoot") or ""),
    )


# ── Bridge binary resolution ──────────────────────────────────────────────


def bridge_install_dir() -> Path:
    """Hermes-managed bridge install location (profile-aware)."""
    return get_hermes_home() / "cursor-sdk-bridge"


def _launcher_name() -> str:
    return "cursor-sdk-bridge.exe" if os.name == "nt" else "cursor-sdk-bridge"


def _bridge_from_cursor_sdk_wheel() -> str | None:
    """Locate the bridge bundled inside an installed ``cursor-sdk`` wheel."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("cursor_sdk")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        base = Path(root)
        for candidate in (
            base / "cursor-sdk-bridge" / "bin" / _launcher_name(),
            base / "_bridge" / "bin" / _launcher_name(),
            base / "bridge" / "bin" / _launcher_name(),
        ):
            if candidate.exists():
                return str(candidate)
        # Fall back to a shallow scan — wheel layout is not contractual.
        try:
            for candidate in base.glob(f"**/bin/{_launcher_name()}"):
                return str(candidate)
        except OSError:
            continue
    return None


def resolve_bridge_command(configured_command: str = "") -> str | None:
    """Resolve the bridge launcher path, or None when not installed.

    Resolution order:
      1. ``cursor_bridge.command`` from config.yaml (caller passes it in)
      2. the bridge embedded in an installed ``cursor-sdk`` PyPI wheel
      3. ``CURSOR_SDK_BRIDGE_BIN`` (explicit compatibility override)
      4. the Hermes-managed install under ``$HERMES_HOME/cursor-sdk-bridge/``

    An arbitrary ``cursor-sdk-bridge`` found on PATH is intentionally ignored;
    runtime execution must use a managed or explicitly configured executable.
    """
    configured = (configured_command or "").strip()
    if configured:
        expanded = os.path.expanduser(configured)
        if shutil.which(expanded) or Path(expanded).exists():
            return expanded
        logger.warning("Configured cursor_bridge.command %r not found", configured)

    wheel_bridge = _bridge_from_cursor_sdk_wheel()
    if wheel_bridge:
        return wheel_bridge

    env_bin = os.getenv("CURSOR_SDK_BRIDGE_BIN", "").strip()
    if env_bin and Path(os.path.expanduser(env_bin)).exists():
        return os.path.expanduser(env_bin)

    for managed in (
        bridge_install_dir() / "bin" / _launcher_name(),
        bridge_install_dir() / "cursor-sdk-bridge" / "bin" / _launcher_name(),
    ):
        if managed.exists():
            return str(managed)
    return None


def bridge_platform() -> tuple[str, str]:
    """Return the (os, arch) pair used by the bridge download URL."""
    import platform as _platform
    import sys

    if sys.platform == "win32":
        os_name = "win32"
    elif sys.platform == "darwin":
        os_name = "darwin"
    else:
        os_name = "linux"
    machine = _platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
    if os_name == "win32":
        arch = "x64"  # win32 archives are x64-only
    return os_name, arch


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CursorBridgeError("unexpected HTTP redirect refused")


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)

_ARCHIVE_ORIGIN_HOST = "github.com"
_ARCHIVE_ASSET_HOST = "release-assets.githubusercontent.com"
_ARCHIVE_REDIRECT_TARGETS = {
    _ARCHIVE_ORIGIN_HOST: frozenset({_ARCHIVE_ASSET_HOST}),
    _ARCHIVE_ASSET_HOST: frozenset({_ARCHIVE_ASSET_HOST}),
}
_ARCHIVE_RELEASE_PATH_PREFIX = "/cursor/sdk-bridge/releases/download/"
_MAX_ARCHIVE_REDIRECTS = 3


def _validate_archive_url(url: str, *, source_host: str | None = None) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        raise CursorBridgeError("bridge archive redirect must use HTTPS")
    if parsed.username or parsed.password:
        raise CursorBridgeError("bridge archive redirect contains userinfo")
    if parsed.fragment:
        raise CursorBridgeError("bridge archive redirect contains a fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise CursorBridgeError("bridge archive redirect has an unsafe port") from exc
    if port is not None and port != 443:
        raise CursorBridgeError("bridge archive redirect has an unsafe port")
    if source_host is not None and host not in _ARCHIVE_REDIRECT_TARGETS.get(source_host, ()):
        raise CursorBridgeError("bridge archive redirect destination is untrusted")
    return host


class _ArchiveRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self):
        super().__init__()
        self._redirects = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        source_host = _validate_archive_url(req.full_url)
        _validate_archive_url(newurl, source_host=source_host)
        self._redirects += 1
        if self._redirects > _MAX_ARCHIVE_REDIRECTS:
            raise CursorBridgeError("bridge archive redirect chain exceeds limit")
        # Release assets are public; never copy credentials or arbitrary headers.
        return urllib.request.Request(
            newurl,
            headers={"User-Agent": "hermes-cli"},
            origin_req_host=req.origin_req_host,
            unverifiable=True,
            method="GET",
        )


def _build_archive_download_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_ArchiveRedirectHandler())


def _fetch_url(url: str, timeout: float = 120) -> bytes:
    _validate_archive_url(url)
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() != _ARCHIVE_ORIGIN_HOST or not parsed.path.startswith(
        _ARCHIVE_RELEASE_PATH_PREFIX
    ):
        raise CursorBridgeError("bridge archive source is not a trusted GitHub release")
    request = urllib.request.Request(url, headers={"User-Agent": "hermes-cli"})
    with _build_archive_download_opener().open(request, timeout=timeout) as response:
        data = response.read(_MAX_ARCHIVE_BYTES + 1)
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise CursorBridgeError("bridge archive exceeds download size limit")
    return data


def download_bridge(version: str = "", *, progress: bool = True) -> str:
    """Download and unpack the bridge archive into the Hermes-managed dir.

    Downloads one exact GitHub release asset and checks it against a digest
    pinned in this source tree.  Returns the launcher path.  Called from setup flows
    (`hermes model`) — the runtime client never downloads implicitly.
    """
    import hashlib
    import tarfile
    import tempfile

    version = (version or DEFAULT_BRIDGE_VERSION).strip().lstrip("v")
    os_name, arch = bridge_platform()
    expected = _TRUSTED_ARCHIVE_SHA256.get((version, os_name, arch))
    if not expected:
        raise CursorBridgeError(
            f"Cursor SDK bridge {version!r} has no pinned digest for {os_name}/{arch}"
        )
    dest_root = bridge_install_dir()
    dest_root.parent.mkdir(parents=True, exist_ok=True)

    if progress:
        print(f"  Downloading Cursor SDK bridge {version} ({os_name}/{arch})...")

    github_url = _GITHUB_RELEASE_URL_TEMPLATE.format(
        version=version, os=os_name, arch=arch
    )
    try:
        data = _fetch_url(github_url)
    except CursorBridgeError:
        raise
    except (urllib.error.URLError, OSError) as exc:
        raise CursorBridgeError(f"bridge download failed: {github_url}: {exc}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise CursorBridgeError("bridge archive checksum mismatch")
    if progress:
        print("  ✓ SHA256 verified against the pinned release digest")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp.write(data)
        archive_path = tmp.name

    stage_root = Path(tempfile.mkdtemp(prefix="cursor-sdk-bridge.", dir=dest_root.parent))
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            expanded = 0
            for member in tar.getmembers():
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise CursorBridgeError("bridge archive contains unsafe path")
                if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                    raise CursorBridgeError("bridge archive contains unsupported member")
                expanded += max(0, int(member.size))
                if expanded > 200 * 1024 * 1024:
                    raise CursorBridgeError("bridge archive expands beyond size limit")
            tar.extractall(stage_root, filter="data")

        # Validate the staged tree before it can replace the active install.
        staged_roots = [stage_root, stage_root / "cursor-sdk-bridge"]
        bridge_root = next(
            (root for root in staged_roots if (root / "manifest.json").exists()), None
        )
        if bridge_root is None:
            raise CursorBridgeError("bridge manifest not found after staged extraction")
        try:
            manifest = json.loads((bridge_root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CursorBridgeError(f"bridge manifest unreadable after install: {exc}") from exc
        expected_entrypoint = f"bin/{_launcher_name()}"
        expected_manifest = {
            "sdkVersion": version,
            "os": os_name,
            "arch": arch,
            "entrypoint": expected_entrypoint,
            "protocol": "sdk.v1",
            "distribution": "standalone",
        }
        mismatches = {
            key: (manifest.get(key), expected_value)
            for key, expected_value in expected_manifest.items()
            if manifest.get(key) != expected_value
        }
        if mismatches:
            raise CursorBridgeError(f"bridge manifest does not match pinned asset: {mismatches}")
        launcher = bridge_root / expected_entrypoint
        if not launcher.is_file():
            raise CursorBridgeError(f"bridge launcher missing after install: {launcher}")
        if os.name != "nt":
            launcher.chmod(launcher.stat().st_mode | 0o755)

        # Replace only after every validation succeeds. Keep the install
        # profile-scoped and never execute a partially extracted tree.
        backup = dest_root.with_name(dest_root.name + ".old")
        try:
            if backup.exists():
                shutil.rmtree(backup)
            if dest_root.exists():
                os.replace(dest_root, backup)
            os.replace(bridge_root, dest_root)
            shutil.rmtree(backup, ignore_errors=True)
        except OSError as exc:
            if not dest_root.exists() and backup.exists():
                os.replace(backup, dest_root)
            raise CursorBridgeError(f"bridge install commit failed: {exc}") from exc
    except CursorBridgeError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise CursorBridgeError(f"bridge archive extraction failed: {exc}") from exc
    finally:
        try:
            os.unlink(archive_path)
        except OSError:
            pass
        shutil.rmtree(stage_root, ignore_errors=True)
    launcher = dest_root / "bin" / _launcher_name()
    if progress:
        print(f"  ✓ Installed Cursor SDK bridge → {launcher}")
    return str(launcher)


# ── Bridge process ────────────────────────────────────────────────────────


def _build_subprocess_env(api_key: str) -> dict[str, str]:
    # The bridge is a model-driving executor: it needs the Cursor credential
    # but must not inherit Tier-1 Hermes secrets (gateway bot tokens, etc.).
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    env["CURSOR_API_KEY"] = api_key
    env["CURSOR_SDK_CLIENT_LANGUAGE"] = "python"
    return env


class CursorBridgeProcess:
    """Owns one ``cursor-sdk-bridge`` child process."""

    def __init__(
        self,
        *,
        command: str,
        api_key: str,
        workspace: str,
        tool_callback_url: str = "",
        tool_callback_auth_token: str = "",
    ):
        self._command = command
        self._api_key = api_key
        self._workspace = str(Path(workspace).resolve())
        self._tool_callback_url = tool_callback_url
        self._tool_callback_auth_token = tool_callback_auth_token
        self._process: subprocess.Popen[str] | None = None
        self.endpoint: BridgeEndpoint | None = None

    @property
    def workspace(self) -> str:
        return self._workspace

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _redact_diagnostic(self, line: str) -> str:
        for secret in (self._api_key, self._tool_callback_auth_token):
            if secret:
                line = line.replace(secret, "[REDACTED]")
        try:
            from agent.redact import redact_sensitive_text

            return redact_sensitive_text(line, force=True)
        except Exception:
            return line

    def start(self) -> BridgeEndpoint:
        argv = [self._command, "--workspace", self._workspace]
        if self._tool_callback_url:
            argv += ["--tool-callback-url", self._tool_callback_url]
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            self._process = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={
                    **_build_subprocess_env(self._api_key),
                    "CURSOR_SDK_TOOL_CALLBACK_AUTH_TOKEN": self._tool_callback_auth_token,
                },
                creationflags=windows_hide_flags(),
            )
            atexit.register(self.stop)
        except OSError as exc:
            raise CursorBridgeError(
                f"could not launch Cursor SDK bridge {self._command!r}: {exc}"
            ) from exc

        discovery = self._await_ready_line(self._process)
        try:
            endpoint = endpoint_from_discovery(discovery)
        except CursorBridgeError:
            # Discovery is untrusted child-process output.  Rejecting it must
            # not leave an orphan bridge listening on an attacker-selected
            # endpoint or retaining the credential in its environment.
            self.stop()
            raise
        self.endpoint = endpoint
        logger.info(
            "Cursor SDK bridge ready (pid=%s, version=%s)",
            endpoint.pid,
            endpoint.server_version,
        )
        return endpoint

    def _await_ready_line(self, process: subprocess.Popen[str]) -> dict[str, Any]:
        """Scan stderr for the discovery line; keep draining forever after.

        A full stderr pipe blocks the bridge, so the scanner thread never
        stops reading.
        """
        import queue as _queue

        found: _queue.Queue[dict[str, Any] | Exception] = _queue.Queue(maxsize=1)

        def scan() -> None:
            if process.stderr is None:
                found.put(CursorBridgeError("bridge process exposed no stderr pipe"))
                return
            diagnostics: list[str] = []
            for line in process.stderr:
                line = line.rstrip("\n")
                try:
                    payload = parse_ready_line(line)
                except CursorBridgeError as exc:
                    found.put(exc)
                    payload = None
                if payload is None:
                    if len(diagnostics) < 60:
                        diagnostics.append(self._redact_diagnostic(line))
                    continue
                found.put(payload)
                for _ in process.stderr:  # drain so the bridge never blocks
                    pass
                return
            found.put(
                CursorBridgeError(
                    "Cursor SDK bridge exited before emitting its ready line. "
                    "Stderr tail:\n" + "\n".join(diagnostics[-20:])
                )
            )

        threading.Thread(target=scan, daemon=True, name="cursor-bridge-stderr").start()
        import queue as _queue

        try:
            result = found.get(timeout=_STARTUP_TIMEOUT_SECONDS)
        except _queue.Empty:
            self.stop()
            raise CursorBridgeError(
                f"timed out after {_STARTUP_TIMEOUT_SECONDS:.0f}s waiting for the "
                "Cursor SDK bridge ready line"
            ) from None
        if isinstance(result, Exception):
            self.stop()
            raise result
        return result

    def stop(self) -> None:
        process, self._process = self._process, None
        self.endpoint = None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
            except OSError:
                pass
        try:
            process.wait(timeout=1)
        except (subprocess.TimeoutExpired, OSError):
            pass


# ── Connect JSON transport ────────────────────────────────────────────────


class ConnectJsonTransport:
    """Connect-over-HTTP/1.1 client with JSON message encoding.

    Unary RPCs: ``POST {base}/sdk.v1.<Service>/<Method>`` with
    ``application/json``; errors arrive as non-200 with a Connect error JSON
    body (``{"code": ..., "message": ...}``).

    Server streams: ``application/connect+json`` enveloped frames.  Each frame
    is 1 flags byte + 4-byte big-endian payload length + payload.  A frame
    with flags bit ``0x02`` is the JSON EndStreamResponse (holds ``error``
    when the stream failed).
    """

    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip("/")
        self._token = auth_token

    def _request(self, path: str, content_type: str, body: bytes) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self._token}",
                "Connect-Protocol-Version": "1",
            },
        )

    def _raise_connect_error(self, raw: bytes, http_status: int | None = None) -> None:
        code = None
        message = raw.decode("utf-8", "replace")[:2000]
        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                code = payload.get("code")
                message = str(payload.get("message") or message)
        except ValueError:
            pass
        message = message.replace(self._token, "[REDACTED]")
        message = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", message)
        message = re.sub(r"\b(?:key|sk)[_-][A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
        prefix = f"HTTP {http_status} " if http_status else ""
        raise CursorBridgeError(f"{prefix}connect error [{code}]: {message}", code=code)

    def unary(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        body = json.dumps(request).encode("utf-8")
        raw = b""
        req = self._request(f"/sdk.v1.{service}/{method}", "application/json", body)
        try:
            with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as reply:
                raw = reply.read(_MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as err:
            self._raise_connect_error(err.read(), http_status=err.code)
        except (urllib.error.URLError, OSError) as err:
            raise CursorBridgeError(f"{service}/{method}: {err}") from None
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise CursorBridgeError(f"{service}/{method}: response exceeds size limit")
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise CursorBridgeError(f"{service}/{method}: invalid JSON response: {exc}") from exc
        return payload if isinstance(payload, dict) else {}

    def server_stream(
        self,
        service: str,
        method: str,
        request: dict[str, Any],
        *,
        read_timeout: float = 90.0,
        deadline: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield stream messages as dicts until the EndStreamResponse frame.

        ``read_timeout`` bounds a single socket read; the bridge emits
        keepalives every ~15s so a quiet-but-alive stream never trips it.
        ``deadline`` (monotonic timestamp) bounds the whole stream.
        """
        payload = json.dumps(request).encode("utf-8")
        body = struct.pack(">BI", 0, len(payload)) + payload
        req = self._request(f"/sdk.v1.{service}/{method}", "application/connect+json", body)
        open_timeout = _deadline_read_timeout(deadline, read_timeout, what=f"{service}/{method}")
        try:
            reply = _NO_REDIRECT_OPENER.open(req, timeout=open_timeout)
        except urllib.error.HTTPError as err:
            self._raise_connect_error(err.read(), http_status=err.code)
            return  # unreachable; keeps type-checkers happy
        except (urllib.error.URLError, OSError) as err:
            raise CursorBridgeError(f"{service}/{method}: {err}") from None

        with reply:
            while True:
                timeout = _deadline_read_timeout(
                    deadline, read_timeout, what=f"{service}/{method}"
                )
                _set_reply_read_timeout(reply, timeout)
                header = _read_exact(reply, 5, what=f"{service}/{method} frame header")
                flags, length = struct.unpack(">BI", header)
                if length > 1 * 1024 * 1024:
                    raise CursorBridgeError(f"{service}/{method}: frame exceeds size limit")
                timeout = _deadline_read_timeout(
                    deadline, read_timeout, what=f"{service}/{method}"
                )
                _set_reply_read_timeout(reply, timeout)
                frame = _read_exact(reply, length, what=f"{service}/{method} frame body")
                if flags & 0x02:
                    try:
                        end = json.loads(frame) if frame else {}
                    except (TypeError, ValueError) as exc:
                        raise CursorBridgeError(
                            f"{service}/{method}: malformed EndStreamResponse"
                        ) from exc
                    error = end.get("error") if isinstance(end, dict) else None
                    if error:
                        self._raise_connect_error(json.dumps(error).encode("utf-8"))
                    return
                if not frame:
                    continue
                try:
                    message = json.loads(frame.decode("utf-8"))
                except ValueError as exc:
                    raise CursorBridgeError(
                        f"{service}/{method}: invalid JSON stream frame: {exc}"
                    ) from exc
                if isinstance(message, dict):
                    yield message


def _deadline_read_timeout(deadline: float | None, default: float, *, what: str) -> float:
    if deadline is None:
        return default
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CursorBridgeError(f"{what}: stream deadline exceeded")
    return max(0.001, min(default, remaining))


def _set_reply_read_timeout(reply: Any, timeout: float) -> None:
    """Apply a shrinking deadline to urllib's underlying socket when present."""
    sock = getattr(getattr(getattr(reply, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        try:
            sock.settimeout(timeout)
        except OSError:
            pass


def _read_exact(stream: Any, count: int, *, what: str) -> bytes:
    chunks = b""
    while len(chunks) < count:
        try:
            chunk = stream.read(count - len(chunks))
        except OSError as exc:
            raise CursorBridgeError(f"{what}: stream read failed: {exc}") from None
        if not chunk:
            raise CursorBridgeError(f"{what}: stream ended before EndStreamResponse")
        chunks += chunk
    return chunks
