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
import time
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

READY_PREFIXES = ("antigravity-bridge ready ", "antigravity bridge ready ")
PROTOCOL = "antigravity-openai-v1"
SHARED_PROTOCOL = "antigravity-openai-shared-v1"
TOKEN_ENV = "HERMES_ANTIGRAVITY_BRIDGE_TOKEN"
STARTUP_TIMEOUT_SECONDS = 30.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
MAX_READY_LINE_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class AntigravityBridgeError(RuntimeError):
    """A safe, user-facing bridge failure."""


class AntigravityBridgeHTTPError(AntigravityBridgeError):
    """A bridge HTTP failure with only its safe status available to callers."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"bridge HTTP {status_code}")


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
    if payload.get("protocol") not in (PROTOCOL, SHARED_PROTOCOL):
        raise AntigravityBridgeError("unsupported Antigravity bridge protocol")
    if payload.get("host") != "127.0.0.1":
        raise AntigravityBridgeError("Antigravity bridge must bind to 127.0.0.1")
    port = payload.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise AntigravityBridgeError("Antigravity bridge readiness has invalid port")
    return {"protocol": payload["protocol"], "host": "127.0.0.1", "port": port}


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
        self._process: subprocess.Popen[Any] | None = None
        self.endpoint: AntigravityBridgeEndpoint | None = None
        self._stop_lock = threading.Lock()
        self._stopped = False

    def start(self) -> AntigravityBridgeEndpoint:
        argv = [self.command] if isinstance(self.command, str) else list(self.command)
        if not argv or not argv[0]:
            raise AntigravityBridgeError("Antigravity bridge command is empty")
        env = dict(os.environ)
        env[TOKEN_ENV] = self.auth_token
        # Managed clients share one installation-owned pool, including callers
        # that pass the resolved executable explicitly (auth/catalog probes).
        # Keep custom commands private and never mutate the parent environment.
        env.pop("HERMES_ANTIGRAVITY_SHARED_ROOT", None)
        shared = isinstance(self.command, str) and self.command == resolve_antigravity_bridge_command()
        if shared:
            from hermes_constants import get_default_hermes_root
            root = str(get_default_hermes_root().expanduser().resolve())
            env["HERMES_HOME"] = root
            env["HERMES_ANTIGRAVITY_SHARED_ROOT"] = root
        try:
            with self._stop_lock:
                if self._stopped:
                    raise AntigravityBridgeError("Antigravity bridge process is stopped")
                process = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=False, env=env,
                )
                self._process = process
                atexit.register(self.stop)
        except OSError as exc:
            raise AntigravityBridgeError("could not launch Antigravity bridge") from exc
        assert process.stdout is not None
        ready: queue.Queue[dict[str, Any] | Exception] = queue.Queue(maxsize=1)

        def scan() -> None:
            assert process.stdout is not None
            line = bytearray()
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                raw = process.stdout.read(1)
                if not raw:
                    ready.put(AntigravityBridgeError("bridge exited before readiness"))
                    return
                line.extend(raw)
                if len(line) > MAX_READY_LINE_BYTES:
                    ready.put(AntigravityBridgeError("bridge readiness line exceeds size limit"))
                    return
                if raw in (b"\n", b"\r"):
                    try:
                        payload = parse_antigravity_ready(line.decode("utf-8", "replace").rstrip("\r\n"))
                    except Exception as exc:
                        ready.put(exc)
                        return
                    line.clear()
                    if payload is not None:
                        ready.put(payload)
                        return
            ready.put(AntigravityBridgeError("timed out waiting for Antigravity bridge readiness"))

        def drain_stderr() -> None:
            assert process is not None and process.stderr is not None
            for _ in process.stderr:
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
        if shared and result["protocol"] != SHARED_PROTOCOL:
            self.stop()
            raise AntigravityBridgeError("Antigravity bridge update required for shared accounts")
        self.endpoint = AntigravityBridgeEndpoint(
            f"http://{result['host']}:{result['port']}", self.auth_token
        )
        return self.endpoint

    def stop(self) -> None:
        with self._stop_lock:
            self._stopped = True
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
    from hermes_constants import get_default_hermes_root
    candidate = get_default_hermes_root().expanduser().resolve() / "antigravity-bridge" / "bin" / (
        "antigravity-bridge.exe" if os.name == "nt" else "antigravity-bridge"
    )
    return str(candidate) if candidate.is_file() else None


class AntigravitySSEStream(Iterator[dict[str, Any]]):
    """Bounded iterator over OpenAI-compatible SSE records."""

    def __init__(self, response: Any):
        self.response = response
        self._closed = False
        self._bytes_read = 0
        self._data_lines: list[str] = []

    def __iter__(self) -> "AntigravitySSEStream":
        return self

    def __next__(self) -> dict[str, Any]:
        while not self._closed:
            try:
                raw = self.response.readline(MAX_RESPONSE_BYTES + 1)
            except OSError as exc:
                self.close()
                raise AntigravityBridgeError("Antigravity bridge stream failed") from exc
            if not raw:
                self.close()
                raise AntigravityBridgeError(
                    "Antigravity bridge stream ended before [DONE]"
                )
            self._bytes_read += len(raw)
            if self._bytes_read > MAX_RESPONSE_BYTES:
                self.close()
                raise AntigravityBridgeError(
                    "Antigravity bridge response exceeds size limit"
                )
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                self.close()
                raise AntigravityBridgeError(
                    "Antigravity bridge returned invalid stream data"
                ) from exc
            if line == "":
                if not self._data_lines:
                    continue
                data = "\n".join(self._data_lines)
                self._data_lines.clear()
                if data.strip() == "[DONE]":
                    self.close()
                    raise StopIteration
                try:
                    payload = json.loads(data)
                except (TypeError, ValueError) as exc:
                    self.close()
                    raise AntigravityBridgeError(
                        "Antigravity bridge returned invalid stream data"
                    ) from exc
                if not isinstance(payload, dict):
                    self.close()
                    raise AntigravityBridgeError(
                        "Antigravity bridge returned invalid stream data"
                    )
                return payload
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if separator and field == "data":
                self._data_lines.append(value[1:] if value.startswith(" ") else value)
        raise StopIteration

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.response.close()
        except OSError:
            pass


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
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            code = exc.code
            if isinstance(code, bool) or not isinstance(code, int) or not 100 <= code <= 599:
                raise AntigravityBridgeError("Antigravity bridge request failed") from None
            raise AntigravityBridgeHTTPError(code) from None
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

    def stream_request(self, method: str, path: str, payload: Any = None, *,
                       timeout: float = 60.0) -> AntigravitySSEStream:
        import urllib.error
        import urllib.request
        request = urllib.request.Request(
            self.endpoint.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers={"Authorization": f"Bearer {self.endpoint.auth_token}",
                     "Accept": "text/event-stream", "Content-Type": "application/json"},
        )

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            code = exc.code
            if isinstance(code, bool) or not isinstance(code, int) or not 100 <= code <= 599:
                raise AntigravityBridgeError("Antigravity bridge request failed") from None
            raise AntigravityBridgeHTTPError(code) from None
        except (urllib.error.URLError, OSError) as exc:
            raise AntigravityBridgeError("Antigravity bridge request failed") from exc
        content_type = str(response.headers.get("Content-Type") or "")
        if content_type.partition(";")[0].strip().lower() != "text/event-stream":
            response.close()
            raise AntigravityBridgeError(
                "Antigravity bridge did not return an event stream"
            )
        return AntigravitySSEStream(response)
