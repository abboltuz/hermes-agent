from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import cursor_bridge_transport as transport


def _standalone_archive() -> bytes:
    manifest = {
        "bridgeVersion": "1.0.27",
        "sdkVersion": "1.0.27",
        "protocol": "sdk.v1",
        "os": "darwin",
        "arch": "arm64",
        "entrypoint": "bin/cursor-sdk-bridge",
        "distribution": "standalone",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload, mode in (
            (
                "cursor-sdk-bridge/manifest.json",
                json.dumps(manifest).encode(),
                0o644,
            ),
            ("cursor-sdk-bridge/bin/cursor-sdk-bridge", b"#!/bin/sh\n", 0o755),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def test_archive_download_follows_trusted_signed_asset_redirect_without_credentials(monkeypatch):
    payload = b"trusted bridge archive"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, _limit):
            return payload

    class FakeOpener:
        def open(self, request, timeout):
            assert request.full_url.startswith("https://github.com/")
            assert dict(request.header_items()) == {"User-agent": "hermes-cli"}
            assert timeout == 120
            return FakeResponse()

    monkeypatch.setattr(transport, "_build_archive_download_opener", lambda: FakeOpener())
    monkeypatch.setattr(
        transport._NO_REDIRECT_OPENER,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            transport.CursorBridgeError("unexpected HTTP redirect refused")
        ),
    )

    assert transport._fetch_url(
        "https://github.com/cursor/sdk-bridge/releases/download/v1.0.27/"
        "cursor-sdk-bridge-standalone-darwin-arm64.tar.gz"
    ) == payload


def test_archive_redirect_preserves_signed_query_and_strips_credentials():
    handler = transport._ArchiveRedirectHandler()
    request = transport.urllib.request.Request(
        "https://github.com/cursor/sdk-bridge/releases/download/v1.0.27/asset",
        headers={
            "Authorization": "Bearer secret",
            "Cookie": "session=secret",
            "Proxy-Authorization": "Basic secret",
            "X-Custom": "must-not-forward",
        },
    )

    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://release-assets.githubusercontent.com/asset?X-Amz-Signature=signed",
    )

    assert redirected.full_url.endswith("?X-Amz-Signature=signed")
    assert dict(redirected.header_items()) == {"User-agent": "hermes-cli"}


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://github.com/cursor/sdk-bridge/releases/download/v1/asset", "HTTPS"),
        ("https://evil.example/asset", "trusted GitHub"),
        ("https://user:pass@github.com/cursor/sdk-bridge/releases/download/v1/asset", "userinfo"),
        ("https://github.com:444/cursor/sdk-bridge/releases/download/v1/asset", "unsafe port"),
        ("https://github.com/cursor/sdk-bridge/releases/download/v1/asset#fragment", "fragment"),
    ],
)
def test_archive_download_rejects_unsafe_initial_urls(url, message):
    with pytest.raises(transport.CursorBridgeError, match=message):
        transport._fetch_url(url)


def test_archive_redirect_rejects_untrusted_destination():
    handler = transport._ArchiveRedirectHandler()
    request = transport.urllib.request.Request(
        "https://github.com/cursor/sdk-bridge/releases/download/v1/asset"
    )
    with pytest.raises(transport.CursorBridgeError, match="destination is untrusted"):
        handler.redirect_request(request, None, 302, "Found", {}, "https://evil.example/asset")


def test_archive_redirect_rejects_downgrade_and_relative_destination():
    handler = transport._ArchiveRedirectHandler()
    request = transport.urllib.request.Request(
        "https://github.com/cursor/sdk-bridge/releases/download/v1/asset"
    )
    for destination, message in (
        ("http://release-assets.githubusercontent.com/asset", "HTTPS"),
        ("/relative-asset", "HTTPS"),
    ):
        with pytest.raises(transport.CursorBridgeError, match=message):
            handler.redirect_request(request, None, 302, "Found", {}, destination)


def test_archive_redirect_rejects_excessive_chain():
    handler = transport._ArchiveRedirectHandler()
    request = transport.urllib.request.Request(
        "https://github.com/cursor/sdk-bridge/releases/download/v1/asset"
    )
    for _ in range(transport._MAX_ARCHIVE_REDIRECTS):
        request = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/asset?sig=signed",
        )
    with pytest.raises(transport.CursorBridgeError, match="exceeds limit"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://release-assets.githubusercontent.com/asset?sig=signed",
        )


def test_download_bridge_uses_embedded_digest_and_atomic_install(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(transport, "bridge_platform", lambda: ("darwin", "arm64"))
    payload = _standalone_archive()
    monkeypatch.setitem(
        transport._TRUSTED_ARCHIVE_SHA256,
        ("1.0.27", "darwin", "arm64"),
        hashlib.sha256(payload).hexdigest(),
    )
    requested = []

    def fetch(url: str) -> bytes:
        requested.append(url)
        return payload

    monkeypatch.setattr(transport, "_fetch_url", fetch)
    from pathlib import Path

    launcher = Path(transport.download_bridge("1.0.27"))

    assert launcher.is_file()
    assert launcher.stat().st_mode & stat.S_IXUSR
    assert requested == [
        "https://github.com/cursor/sdk-bridge/releases/download/v1.0.27/"
        "cursor-sdk-bridge-standalone-darwin-arm64.tar.gz"
    ]
    assert not list((tmp_path / "cursor" / "bridge").glob(".install-*"))


def test_download_bridge_rejects_bad_or_unknown_digest_before_install(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(transport, "bridge_platform", lambda: ("darwin", "arm64"))
    monkeypatch.setattr(transport, "_fetch_url", lambda _url: _standalone_archive())
    monkeypatch.setitem(
        transport._TRUSTED_ARCHIVE_SHA256,
        ("1.0.27", "darwin", "arm64"),
        "0" * 64,
    )

    with pytest.raises(transport.CursorBridgeError, match="checksum mismatch"):
        transport.download_bridge("1.0.27")
    assert not transport.bridge_install_dir().exists()

    called = False

    def should_not_fetch(_url: str) -> bytes:
        nonlocal called
        called = True
        return b""

    monkeypatch.setattr(transport, "_fetch_url", should_not_fetch)
    with pytest.raises(transport.CursorBridgeError, match="no pinned digest"):
        transport.download_bridge("9.9.9")
    assert called is False


def test_discovery_accepts_official_secure_temp_token_file(tmp_path):
    token_dir = tmp_path / "cursor-sdk-bridge-test"
    token_dir.mkdir(mode=0o700)
    token_dir.chmod(0o700)
    token_path = token_dir / "auth-token"
    token_path.write_text("local-capability", encoding="utf-8")
    token_path.chmod(0o600)

    endpoint = transport.endpoint_from_discovery(
        {
            "schemaVersion": 1,
            "protocolVersion": "sdk.v1",
            "transport": "tcp",
            "protocol": "connect",
            "url": "http://127.0.0.1:43123",
            "authTokenFile": str(token_path),
        }
    )

    assert endpoint.url == "http://127.0.0.1:43123"
    assert endpoint.auth_token == "local-capability"


def test_discovery_rejects_unsafe_token_file_and_non_loopback(tmp_path):
    token_dir = tmp_path / "cursor-sdk-bridge-test"
    token_dir.mkdir(mode=0o700)
    token_dir.chmod(0o700)
    token_path = token_dir / "auth-token"
    token_path.write_text("local-capability", encoding="utf-8")
    token_path.chmod(0o644)

    with pytest.raises(transport.CursorBridgeError, match="permissions are unsafe"):
        transport.endpoint_from_discovery(
            {
                "schemaVersion": 1,
                "protocolVersion": "sdk.v1",
                "transport": "tcp",
                "protocol": "connect",
                "url": "http://127.0.0.1:43123",
                "authTokenFile": str(token_path),
            }
        )
    with pytest.raises(transport.CursorBridgeError, match="loopback"):
        transport.endpoint_from_discovery(
            {
                "schemaVersion": 1,
                "protocolVersion": "sdk.v1",
                "transport": "tcp",
                "protocol": "connect",
                "url": "http://192.0.2.1:43123",
                "authToken": "x",
            }
        )
    with pytest.raises(transport.CursorBridgeError, match="unsafe URL components"):
        transport.endpoint_from_discovery(
            {
                "schemaVersion": 1,
                "protocolVersion": "sdk.v1",
                "transport": "tcp",
                "protocol": "connect",
                "url": "http://127.0.0.1:43123?token=x",
                "authToken": "x",
            }
        )


def _start_server(handler_type):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_connect_transport_refuses_http_redirects():
    redirected = threading.Event()

    class Target(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            redirected.set()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    target, target_thread = _start_server(Target)

    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            self.send_response(307)
            self.send_header(
                "Location", f"http://127.0.0.1:{target.server_port}/stolen"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()

    redirect, redirect_thread = _start_server(Redirect)
    try:
        client = transport.ConnectJsonTransport(
            f"http://127.0.0.1:{redirect.server_port}", "local-capability"
        )
        with pytest.raises(transport.CursorBridgeError, match="redirect refused"):
            client.unary("SdkBridgeControlService", "Ping", {}, timeout=1.0)
        assert not redirected.wait(0.1)
    finally:
        redirect.shutdown()
        target.shutdown()
        redirect.server_close()
        target.server_close()
        redirect_thread.join(timeout=1)
        target_thread.join(timeout=1)


def test_server_stream_enforces_total_deadline():
    class SlowBody(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/connect+json")
            self.end_headers()
            self.wfile.write(struct.pack(">BI", 0, 64))
            self.wfile.flush()
            time.sleep(3)

    server, thread = _start_server(SlowBody)
    try:
        client = transport.ConnectJsonTransport(
            f"http://127.0.0.1:{server.server_port}", "local-capability"
        )
        started = time.monotonic()
        with pytest.raises(transport.CursorBridgeError):
            list(
                client.server_stream(
                    "SdkAgentService",
                    "Send",
                    {},
                    deadline=time.monotonic() + 0.2,
                    read_timeout=5.0,
                )
            )
        assert time.monotonic() - started < 2.5
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)
