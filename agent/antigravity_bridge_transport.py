"""Managed loopback transport for the native Antigravity bridge."""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import re
import secrets
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

READY_PREFIXES = ("antigravity-bridge ready ", "antigravity bridge ready ")
PROTOCOL = "antigravity-openai-v1"
TOKEN_ENV = "HERMES_ANTIGRAVITY_BRIDGE_TOKEN"
STARTUP_TIMEOUT_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
MAX_READY_LINE_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class AntigravityBridgeError(RuntimeError):
    """A safe, user-facing bridge failure."""


def redact_antigravity_text(text: str, token: str = "") -> str:
    result = str(text)
    if token:
        result = result.replace(token, "[REDACTED]")
    return re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", result)


def parse_antigravity_ready(line: str) -> dict[str, Any] | None:
    if len(line.encode("utf-8", "replace")) > MAX_READY_LINE_BYTES:
        raise AntigravityBridgeError("bridge readiness line exceeds size limit")
    for prefix in READY_PREFIXES:
        if line.startswith(prefix):
            try:
                payload = json.loads(line[len(prefix):].strip())
            except (TypeError, ValueError) as exc:
                raise AntigravityBridgeError("invalid bridge readiness JSON") from exc
            if not isinstance(payload, dict):
                raise AntigravityBridgeError("bridge readiness payload is not an object")
            return validate_antigravity_ready(payload)
    return None


def validate_antigravity_ready(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("protocol") != PROTOCOL:
        raise AntigravityBridgeError("unsupported Antigravity bridge protocol")
    if payload.get("host") != "127.0.0.1":
        raise AntigravityBridgeError("Antigravity bridge must bind to 127.0.0.1")
    port = payload.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise AntigravityBridgeError("Antigravity bridge readiness has invalid port")
    return {"protocol": PROTOCOL, "host": "127.0.0.1", "port": port}


@dataclass
class AntigravityBridgeEndpoint:
    base_url: str
    auth_token: str


class AntigravityBridgeProcess:
    """Own one explicitly selected bridge child and its ephemeral token."""

    def __init__(self, command: str | list[str], *, startup_timeout: float = STARTUP_TIMEOUT_SECONDS):
        self.command = command
        self.startup_timeout = startup_timeout
        self.auth_token = secrets.token_urlsafe(32)
        self._process: subprocess.Popen[str] | None = None
        self.endpoint: AntigravityBridgeEndpoint | None = None
        self._stop_lock = threading.Lock()

    def start(self) -> AntigravityBridgeEndpoint:
        argv = [self.command] if isinstance(self.command, str) else list(self.command)
        if not argv or not argv[0]:
            raise AntigravityBridgeError("Antigravity bridge command is empty")
        env = dict(os.environ)
        env[TOKEN_ENV] = self.auth_token
        try:
            self._process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=env,
            )
            atexit.register(self.stop)
        except OSError as exc:
            raise AntigravityBridgeError("could not launch Antigravity bridge") from exc
        assert self._process.stdout is not None
        ready: queue.Queue[dict[str, Any] | Exception] = queue.Queue(maxsize=1)

        def scan() -> None:
            assert self._process is not None and self._process.stdout is not None
            for raw in self._process.stdout:
                try:
                    payload = parse_antigravity_ready(raw.rstrip("\r\n"))
                except Exception as exc:
                    ready.put(exc)
                    return
                if payload is not None:
                    ready.put(payload)
                    for _ in self._process.stdout:
                        pass
                    return
            ready.put(AntigravityBridgeError("bridge exited before readiness"))

        def drain_stderr() -> None:
            assert self._process is not None and self._process.stderr is not None
            for _ in self._process.stderr:
                pass
        threading.Thread(target=drain_stderr, daemon=True, name="antigravity-bridge-log").start()
        threading.Thread(target=scan, daemon=True, name="antigravity-bridge-ready").start()
        try:
            result = ready.get(timeout=self.startup_timeout)
        except queue.Empty:
            self.stop()
            raise AntigravityBridgeError("timed out waiting for Antigravity bridge readiness") from None
        if isinstance(result, Exception):
            self.stop()
            raise result
        self.endpoint = AntigravityBridgeEndpoint(
            f"http://{result['host']}:{result['port']}", self.auth_token
        )
        return self.endpoint

    def stop(self) -> None:
        with self._stop_lock:
            process, self._process = self._process, None
            self.endpoint = None
            if process is None:
                return
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
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


def resolve_antigravity_bridge_command() -> str | None:
    """Return only a Hermes-managed executable; never search PATH or download."""
    from hermes_constants import get_hermes_home
    candidate = get_hermes_home() / "antigravity-bridge" / "bin" / (
        "antigravity-bridge.exe" if os.name == "nt" else "antigravity-bridge"
    )
    return str(candidate) if candidate.is_file() else None


class AntigravityHTTPTransport:
    def __init__(self, endpoint: AntigravityBridgeEndpoint):
        self.endpoint = endpoint

    def request(self, method: str, path: str, body: bytes | None = None, *, timeout: float = 60.0) -> bytes:
        import urllib.error
        import urllib.request
        request = urllib.request.Request(
            self.endpoint.base_url + path, data=body, method=method,
            headers={"Authorization": f"Bearer {self.endpoint.auth_token}",
                     "Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raw = exc.read(4096).decode("utf-8", "replace")
            raise AntigravityBridgeError(redact_antigravity_text(f"bridge HTTP {exc.code}: {raw}", self.endpoint.auth_token)) from None
        except (urllib.error.URLError, OSError) as exc:
            raise AntigravityBridgeError("Antigravity bridge request failed") from exc
        if len(data) > MAX_RESPONSE_BYTES:
            raise AntigravityBridgeError("Antigravity bridge response exceeds size limit")
        return data

    def json_request(self, method: str, path: str, payload: Any = None, *, timeout: float = 60.0) -> Any:
        raw = self.request(method, path, json.dumps(payload).encode() if payload is not None else None, timeout=timeout)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, ValueError) as exc:
            raise AntigravityBridgeError("Antigravity bridge returned invalid JSON") from exc
