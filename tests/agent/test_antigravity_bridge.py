"""Contract tests for the native Antigravity bridge provider."""

from __future__ import annotations

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
