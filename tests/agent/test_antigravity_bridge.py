"""Contract tests for the native Antigravity bridge provider."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def test_antigravity_profile_exposes_identity_and_curated_catalog():
    from providers import get_provider_profile

    profile = get_provider_profile("antigravity")

    assert profile is not None
    assert profile.name == "antigravity"
    assert profile.display_name == "Google Antigravity"
    assert "google-antigravity" in profile.aliases
    assert "google" not in profile.aliases
    assert profile.base_url == "sdkbridge://antigravity"
    assert any("gemini" in model.lower() for model in profile.fallback_models)
    assert any("claude" in model.lower() for model in profile.fallback_models)


def test_antigravity_catalog_filters_supported_families_and_preserves_live_empty():
    from agent.antigravity_bridge_client import filter_antigravity_models

    assert filter_antigravity_models([
        {"id": "antigravity-claude-sonnet-4-6"},
        {"id": "antigravity-gemini-3-pro"},
        {"id": "claude-sonnet-4"},
        {"id": "gemini-2.5-pro"},
        {"id": "gpt-5"},
        {"id": "claude-sonnet-4"},
    ]) == [
        "antigravity-claude-sonnet-4-6",
        "antigravity-gemini-3-pro",
        "claude-sonnet-4",
        "gemini-2.5-pro",
    ]
    assert filter_antigravity_models([]) == []
    assert filter_antigravity_models(None) is None


def test_antigravity_fallback_models_are_bridge_owned_supported_families():
    from agent.antigravity_bridge_client import CURATED_FALLBACK_MODELS

    assert CURATED_FALLBACK_MODELS
    assert all(model.startswith(("antigravity-gemini-", "antigravity-claude-"))
               for model in CURATED_FALLBACK_MODELS)


def test_antigravity_ready_payload_requires_loopback_contract():
    from agent.antigravity_bridge_transport import validate_antigravity_ready

    valid = {"protocol": "antigravity-openai-v1", "host": "127.0.0.1", "port": 43123}
    assert validate_antigravity_ready(valid) == valid
    for invalid in (
        {**valid, "host": "0.0.0.0"},
        {**valid, "protocol": "other"},
        {**valid, "port": 0},
        {**valid, "port": 65536},
    ):
        with pytest.raises(Exception):
            validate_antigravity_ready(invalid)


def test_antigravity_error_redacts_bearer_token():
    from agent.antigravity_bridge_transport import redact_antigravity_text

    token = "ephemeral-secret-token"
    output = redact_antigravity_text(f"Authorization: Bearer {token}", token)
    assert token not in output
    assert "[REDACTED]" in output


def test_antigravity_http_errors_never_expose_upstream_body_or_token():
    from agent.antigravity_bridge_transport import (
        AntigravityBridgeEndpoint, AntigravityBridgeError, AntigravityHTTPTransport,
    )

    class ErrorHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(502)
            self.end_headers()
            self.wfile.write(b"private-account-id bearer super-secret")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ErrorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = "super-secret"
    try:
        transport = AntigravityHTTPTransport(
            AntigravityBridgeEndpoint(f"http://127.0.0.1:{server.server_port}", token)
        )
        with pytest.raises(AntigravityBridgeError) as exc_info:
            transport.request("GET", "/v1/models")
        assert str(exc_info.value) == "bridge HTTP 502"
        assert "private-account-id" not in str(exc_info.value)
        assert token not in str(exc_info.value)
    finally:
        server.shutdown()


def test_antigravity_redirect_does_not_forward_authenticated_request():
    from agent.antigravity_bridge_transport import (
        AntigravityBridgeEndpoint, AntigravityBridgeError, AntigravityHTTPTransport,
    )

    destination_requests = []

    class DestinationHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            destination_requests.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    destination = ThreadingHTTPServer(("127.0.0.1", 0), DestinationHandler)

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{destination.server_port}/sink")
            self.end_headers()

        def log_message(self, *args):
            pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (destination, origin)]
    for thread in threads:
        thread.start()
    try:
        transport = AntigravityHTTPTransport(
            AntigravityBridgeEndpoint(f"http://127.0.0.1:{origin.server_port}", "token")
        )
        with pytest.raises(AntigravityBridgeError):
            transport.request("GET", "/v1/models")
        assert destination_requests == []
    finally:
        origin.shutdown()
        destination.shutdown()


def test_antigravity_close_is_terminal_and_does_not_start_bridge(monkeypatch):
    from agent.antigravity_bridge_client import AntigravityBridgeClient

    client = AntigravityBridgeClient(bridge_command=[sys.executable, "-c", "raise SystemExit"])
    client.close()
    monkeypatch.setattr(client, "_ensure_transport", lambda: pytest.fail("bridge started after close"))
    with pytest.raises(Exception, match="closed"):
        client.list_models()


def test_antigravity_abort_inflight_reaps_blocked_child(tmp_path):
    from agent.antigravity_bridge_client import AntigravityBridgeClient

    child = tmp_path / "blocking_bridge.py"
    child.write_text(
        """import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({'data': []}).encode(); self.send_response(200)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        while True: time.sleep(1)
    def log_message(self, *args): pass
ns = ThreadingHTTPServer(('127.0.0.1', 0), H)
print('antigravity-bridge ready ' + json.dumps({'protocol': 'antigravity-openai-v1', 'host': '127.0.0.1', 'port': ns.server_port}), flush=True)
ns.serve_forever()
""",
        encoding="utf-8",
    )
    client = AntigravityBridgeClient(bridge_command=[sys.executable, str(child)])
    request_thread = None
    try:
        client.list_models()
        def make_request():
            try:
                client.chat.completions.create(model="x", messages=[])
            except Exception:
                pass

        request_thread = threading.Thread(target=make_request, daemon=True)
        request_thread.start()
        time.sleep(0.2)
        client.abort_inflight()
        request_thread.join(timeout=3)
        assert not request_thread.is_alive()
        assert client._process is None
    finally:
        client.close()


def test_antigravity_readiness_rejects_oversized_unterminated_line(tmp_path):
    from agent.antigravity_bridge_transport import (
        MAX_READY_LINE_BYTES, AntigravityBridgeError, AntigravityBridgeProcess,
    )

    child = tmp_path / "no_newline_bridge.py"
    child.write_text(
        f"import sys, time; sys.stdout.write('x' * {MAX_READY_LINE_BYTES + 1}); sys.stdout.flush(); time.sleep(5)\n",
        encoding="utf-8",
    )
    process = AntigravityBridgeProcess([sys.executable, str(child)], startup_timeout=2)
    with pytest.raises(AntigravityBridgeError, match="exceeds size limit"):
        process.start()
    process.stop()


def test_antigravity_client_crosses_real_subprocess_and_http_boundary(tmp_path):
    from agent.antigravity_bridge_client import AntigravityBridgeClient

    child = tmp_path / "fake_bridge.py"
    child.write_text(
        """import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
token = os.environ['HERMES_ANTIGRAVITY_BRIDGE_TOKEN']
class H(BaseHTTPRequestHandler):
    def _reply(self, body):
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.headers.get('Authorization') != 'Bearer ' + token: self.send_error(401); return
        self._reply(json.dumps({'data': [{'id': 'gemini-test'}, {'id': 'claude-test'}, {'id': 'gpt-hidden'}]}).encode())
    def do_POST(self):
        if self.headers.get('Authorization') != 'Bearer ' + token: self.send_error(401); return
        self._reply(json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'bridge reply'}}]}).encode())
    def log_message(self, *args): pass
ns = ThreadingHTTPServer(('127.0.0.1', 0), H)
print('antigravity-bridge ready ' + json.dumps({'protocol': 'antigravity-openai-v1', 'host': '127.0.0.1', 'port': ns.server_port}), flush=True)
ns.serve_forever()
""",
        encoding="utf-8",
    )
    client = AntigravityBridgeClient(bridge_command=[sys.executable, str(child)])
    try:
        assert [item["id"] for item in client.list_models()] == ["gemini-test", "claude-test", "gpt-hidden"]
        reply = client.chat.completions.create(model="gemini-test", messages=[])
        assert reply.choices[0].message.content == "bridge reply"
    finally:
        client.close()
        client.close()
    assert client._process is None


def _wait_for_file(path, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).is_file():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for marker {path}")


def _reap_ledger_pids(ledger):
    if not ledger.is_file():
        return
    for raw_pid in ledger.read_text(encoding="utf-8").splitlines():
        try:
            os.kill(int(raw_pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass


def _write_blocking_bridge(path, *, entered_path=None, ledger_path=None,
                           emit_readiness=True, readiness_delay: float = 0):
    ledger_setup = (
        f"from pathlib import Path; marker = Path({str(ledger_path)!r}).open('a', encoding='utf-8'); marker.write(str(__import__('os').getpid()) + '\\n'); marker.close()\n"
        if ledger_path else ""
    )
    entered_setup = (
        f"from pathlib import Path; Path({str(entered_path)!r}).touch()\n"
        if entered_path else ""
    )
    readiness = (
        f"time.sleep({readiness_delay}); print('antigravity-bridge ready ' + json.dumps({{'protocol': 'antigravity-openai-v1', 'host': '127.0.0.1', 'port': ns.server_port}}), flush=True)\n"
        if emit_readiness else ""
    )
    path.write_text(
        f"""import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
{ledger_setup}
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({{'data': []}}).encode(); self.send_response(200)
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        {entered_setup}
        while True: time.sleep(1)
    def log_message(self, *args): pass
ns = ThreadingHTTPServer(('127.0.0.1', 0), H)
{readiness}
ns.serve_forever()
""",
        encoding="utf-8",
    )


def test_antigravity_async_cancellation_reaps_concrete_child(tmp_path):
    from agent.antigravity_bridge_client import AsyncAntigravityBridgeClient

    child = tmp_path / "blocking_async_bridge.py"
    entered = tmp_path / "post-entered"
    _write_blocking_bridge(child, entered_path=entered)

    async def exercise():
        client = AsyncAntigravityBridgeClient(bridge_command=[sys.executable, str(child)])
        await client.list_models()
        process = client._sync._process
        assert process is not None and process._process is not None
        child_process = process._process
        task = asyncio.create_task(client.chat.completions.create(model="x", messages=[]))
        await asyncio.to_thread(_wait_for_file, entered)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert child_process.poll() is not None
        await client.close()
        await client.close()

    asyncio.run(exercise())


def test_antigravity_concurrent_first_use_has_one_owned_child(tmp_path):
    from agent.antigravity_bridge_client import AntigravityBridgeClient

    child = tmp_path / "blocking_concurrent_bridge.py"
    ledger = tmp_path / "bridge-pids"
    _write_blocking_bridge(child, ledger_path=ledger, readiness_delay=0.3)
    client = AntigravityBridgeClient(bridge_command=[sys.executable, str(child)])
    barrier = threading.Barrier(3)
    results = []

    def call_list_models():
        barrier.wait()
        try:
            results.append(client.list_models())
        except Exception as exc:
            results.append(exc)

    threads = [threading.Thread(target=call_list_models) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
    passed = False
    try:
        assert all(not isinstance(result, Exception) for result in results)
        pids = {int(raw_pid) for raw_pid in ledger.read_text(encoding="utf-8").splitlines()}
        assert len(pids) == 1
        process = client._process
        assert process is not None and process._process is not None
        child_process = process._process
        assert child_process.poll() is None
        client.close()
        assert child_process.poll() is not None
        passed = True
    finally:
        client.close()
        if not passed:
            _reap_ledger_pids(ledger)


def test_antigravity_abort_pending_readiness_reaps_starter_and_wakes_waiter(tmp_path):
    from agent.antigravity_bridge_client import (
        AntigravityBridgeClient, AntigravityBridgeError,
    )
    child = tmp_path / "pending_readiness_bridge.py"
    started = tmp_path / "bridge-started"
    _write_blocking_bridge(child, ledger_path=started, emit_readiness=False)
    client = AntigravityBridgeClient(bridge_command=[sys.executable, str(child)])
    barrier = threading.Barrier(3)
    results = []

    def call_list_models():
        barrier.wait()
        try:
            results.append(client.list_models())
        except Exception as exc:
            results.append(exc)

    threads = [threading.Thread(target=call_list_models) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    starting_process = None
    passed = False
    try:
        _wait_for_file(started)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            wrapper = client._starting_process
            if wrapper is not None and wrapper._process is not None:
                starting_process = wrapper._process
                break
            time.sleep(0.02)
        assert starting_process is not None
        client.abort_inflight()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert all(isinstance(result, AntigravityBridgeError) for result in results)
        assert starting_process.poll() is not None
        assert client.is_closed
        assert client._process is None
        passed = True
    finally:
        client.close()
        if starting_process is not None and starting_process.poll() is None:
            starting_process.terminate()
        if not passed:
            _reap_ledger_pids(started)


def test_antigravity_loopback_bypasses_environment_proxy(monkeypatch):
    from agent.antigravity_bridge_transport import (
        AntigravityBridgeEndpoint, AntigravityHTTPTransport,
    )

    origin_requests = []
    proxy_requests = []

    class OriginHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            origin_requests.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            proxy_requests.append(self.headers.get("Authorization"))
            self.send_response(502)
            self.end_headers()

        def log_message(self, *args):
            pass

    origin = ThreadingHTTPServer(("127.0.0.1", 0), OriginHandler)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (origin, proxy)]
    for thread in threads:
        thread.start()
    try:
        for name in ("HTTP_PROXY", "http_proxy"):
            monkeypatch.setenv(name, f"http://127.0.0.1:{proxy.server_port}")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        transport = AntigravityHTTPTransport(
            AntigravityBridgeEndpoint(f"http://127.0.0.1:{origin.server_port}", "private-bearer")
        )
        assert transport.request("GET", "/v1/models") == b"{}"
        assert origin_requests == ["Bearer private-bearer"]
        assert proxy_requests == []
    finally:
        origin.shutdown()
        proxy.shutdown()


def test_antigravity_runtime_factory_selects_client_without_starting_bridge(monkeypatch):
    from types import SimpleNamespace
    from agent.agent_runtime_helpers import create_openai_client
    from agent.antigravity_bridge_client import AntigravityBridgeClient

    agent = SimpleNamespace(provider="antigravity", model="gemini-3-pro")
    client = create_openai_client(
        agent,
        {"base_url": "sdkbridge://antigravity", "bridge_command": [sys.executable, "-c", "raise SystemExit"]},
        reason="test",
        shared=False,
    )
    assert isinstance(client, AntigravityBridgeClient)
    assert client._process is None
    client.close()


def test_antigravity_auxiliary_router_returns_sync_and_async_clients(monkeypatch):
    from agent import antigravity_bridge_transport
    from agent.antigravity_bridge_client import AntigravityBridgeClient, AsyncAntigravityBridgeClient
    from agent.auxiliary_client import resolve_provider_client

    monkeypatch.setattr(
        antigravity_bridge_transport,
        "resolve_antigravity_bridge_command",
        lambda: "/tmp/fake-antigravity-bridge",
    )
    sync_client, sync_model = resolve_provider_client("antigravity", model="gemini-3-pro")
    async_client, async_model = resolve_provider_client(
        "antigravity", model="antigravity-claude-sonnet-4-6", async_mode=True
    )
    assert isinstance(sync_client, AntigravityBridgeClient)
    assert isinstance(async_client, AsyncAntigravityBridgeClient)
    assert sync_model == "gemini-3-pro"
    assert async_model == "antigravity-claude-sonnet-4-6"
    sync_client.close()
    async_client._sync.close()


def test_antigravity_command_resolution_fails_closed_for_missing_managed_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-hermes"))
    from agent.antigravity_bridge_transport import resolve_antigravity_bridge_command

    assert resolve_antigravity_bridge_command() is None
