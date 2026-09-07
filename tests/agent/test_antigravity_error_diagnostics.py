"""Real loopback coverage for the bridge's untrusted diagnostic boundary."""
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent.antigravity_bridge_transport import (
    AntigravityBridgeEndpoint,
    AntigravityBridgeHTTPError,
    AntigravityHTTPTransport,
    DIAGNOSTIC_HEADER,
    MAX_ERROR_BYTES,
)


DIAGNOSTIC_ID = "a63536c5-3b73-4e96-a339-8513e46460ab"
SECRET = "private@example.invalid Bearer test-secret prompt-content"


@contextmanager
def error_server(body, *, status=403, headers=None, stall=False, trickle=False,
                 chunked=False, declared_length=None):
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Transfer-Encoding" if chunked else "Content-Length",
                             "chunked" if chunked else str(len(body) if declared_length is None else declared_length))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if stall:
                release.wait(5)
                return
            try:
                if trickle:
                    for byte in body:
                        if release.wait(0.01):
                            return
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                else:
                    self.wfile.write((f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n") if chunked else body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield AntigravityHTTPTransport(AntigravityBridgeEndpoint(
            f"http://127.0.0.1:{server.server_port}", "test-secret",
        ))
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def call(transport, stream, timeout=5):
    if stream:
        transport.stream_request("GET", "/v1/chat/completions", timeout=timeout)
    else:
        transport.request("GET", "/v1/chat/completions", timeout=timeout)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason,status", [
    ("VALIDATION_REQUIRED", 403), ("PERMISSION_DENIED", 403),
    ("UNAUTHENTICATED", 401), ("RATE_LIMIT_EXCEEDED", 429),
    ("QUOTA_EXHAUSTED", 429), ("MODEL_CAPACITY_EXHAUSTED", 503),
    ("MODEL_NOT_FOUND", 404), ("INVALID_REQUEST", 400),
    ("UPSTREAM_UNAVAILABLE", 502), ("UNKNOWN", 403),
    ("SERVICE_DISABLED", 403), ("ACCESS_TOKEN_SCOPE_INSUFFICIENT", 403),
    ("INTERNAL_ERROR", 500),
])
def test_safe_reason_reaches_exception_and_log(stream, reason, status, caplog):
    body = json.dumps({"error": {
        "code": reason, "diagnostic_id": DIAGNOSTIC_ID,
        "message": SECRET, "type": SECRET, "details": {"email": SECRET},
    }}).encode()
    with error_server(body, status=status) as transport:
        with pytest.raises(AntigravityBridgeHTTPError) as caught:
            call(transport, stream)
    error = caught.value
    assert error.status_code == status
    assert error.code == reason
    assert error.diagnostic_id == DIAGNOSTIC_ID
    assert reason in str(error)
    assert DIAGNOSTIC_ID in str(error)
    assert SECRET not in str(error) + caplog.text
    assert "test-secret" not in str(error) + caplog.text
    record = next(r for r in caplog.records if r.message.startswith("antigravity_http_error "))
    assert json.loads(record.message.split(" ", 1)[1]) == {
        "status": status, "code": reason, "diagnostic_id": DIAGNOSTIC_ID,
    }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("body", [
    SECRET.encode(), b"{invalid", b"\xff", b"[]", b"null",
    json.dumps({"error": {"code": SECRET, "diagnostic_id": SECRET, "message": SECRET}}).encode(),
    json.dumps({"error": {"code": [], "diagnostic_id": {"secret": SECRET}}}).encode(),
    b"[" * 2000 + b"]" * 2000,
    b"x" * (MAX_ERROR_BYTES + 1),
])
def test_malformed_old_or_oversize_errors_remain_status_only(stream, body, caplog):
    with error_server(body, headers={DIAGNOSTIC_HEADER: "not-a-uuid"}) as transport:
        with pytest.raises(AntigravityBridgeHTTPError) as caught:
            call(transport, stream)
    assert str(caught.value) == "bridge HTTP 403"
    assert caught.value.code is None
    assert caught.value.diagnostic_id is None
    assert SECRET not in caplog.text


@pytest.mark.parametrize("stream", [False, True])
def test_valid_header_survives_old_body(stream):
    with error_server(SECRET.encode(), headers={DIAGNOSTIC_HEADER: DIAGNOSTIC_ID}) as transport:
        with pytest.raises(AntigravityBridgeHTTPError) as caught:
            call(transport, stream)
    assert caught.value.code is None
    assert caught.value.diagnostic_id == DIAGNOSTIC_ID


@pytest.mark.parametrize("stream", [False, True])
def test_disagreeing_correlation_is_not_trusted(stream):
    body = json.dumps({"error": {"code": "VALIDATION_REQUIRED", "diagnostic_id": DIAGNOSTIC_ID}}).encode()
    with error_server(body, headers={DIAGNOSTIC_HEADER: "b63536c5-3b73-4e96-a339-8513e46460ab"}) as transport:
        with pytest.raises(AntigravityBridgeHTTPError) as caught:
            call(transport, stream)
    assert str(caught.value) == "bridge HTTP 403"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("behavior", [{"stall": True}, {"trickle": True}])
@pytest.mark.parametrize("timeout", [0.1, 5])
def test_error_body_cannot_extend_caller_deadline(stream, behavior, timeout):
    with error_server(b"x" * 4000, **behavior) as transport:
        started = time.monotonic()
        with pytest.raises(AntigravityBridgeHTTPError) as caught:
            call(transport, stream, timeout=timeout)
        # A stalled/dripping diagnostic must not turn a known HTTP error into
        # another timeout, or wait to consume the body (~40 seconds here).
        assert time.monotonic() - started < 2
    assert caught.value.status_code == 403
    assert str(caught.value) == "bridge HTTP 403"


@pytest.mark.parametrize("stream", [False, True])
def test_chunked_error_does_not_wait_for_chunk_framing(stream):
    with error_server(SECRET.encode(), chunked=True, stall=True,
                      headers={DIAGNOSTIC_HEADER: DIAGNOSTIC_ID}) as transport:
        started = time.monotonic()
        with pytest.raises(AntigravityBridgeHTTPError) as caught:
            call(transport, stream)
        assert time.monotonic() - started < 2
    assert caught.value.code is None
    assert caught.value.diagnostic_id == DIAGNOSTIC_ID


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("declared_length", [1, 2000, -1, "bad"])
def test_wrong_content_length_cannot_establish_body_diagnostic(stream, declared_length):
    body = json.dumps({"error": {"code": "VALIDATION_REQUIRED"}}).encode()
    with error_server(body, declared_length=declared_length,
                      headers={DIAGNOSTIC_HEADER: DIAGNOSTIC_ID}) as transport:
        with pytest.raises(AntigravityBridgeHTTPError) as caught:
            call(transport, stream)
    assert caught.value.code is None
    assert caught.value.diagnostic_id == DIAGNOSTIC_ID
