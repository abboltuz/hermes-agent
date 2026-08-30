"""Contract tests for the native Antigravity bridge provider."""

from __future__ import annotations

import sys

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
        {"id": "claude-sonnet-4"},
        {"id": "gemini-2.5-pro"},
        {"id": "gpt-5"},
        {"id": "claude-sonnet-4"},
    ]) == ["claude-sonnet-4", "gemini-2.5-pro"]
    assert filter_antigravity_models([]) == []
    assert filter_antigravity_models(None) is None


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
