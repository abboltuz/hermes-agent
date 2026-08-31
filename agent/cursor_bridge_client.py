"""OpenAI-compatible client that runs Hermes turns through the Cursor SDK bridge.

This adapter lets Hermes treat a user's Cursor subscription as a chat backend
(``model.provider: cursor``).  Each request spawns/reuses a local
``cursor-sdk-bridge`` process (the official `sdk.v1` contract from
https://github.com/cursor/sdk-bridge), creates a short-lived Cursor agent,
sends the formatted conversation as one prompt, and converts the result back
into the minimal OpenAI-response shape Hermes expects.

Tool handling keeps Hermes's harness intact. Cursor's built-in tools are
always disabled and Hermes tools
  are declared as SDK *custom tools*.  When the Cursor agent invokes one, the
  bridge round-trips to the loopback callback server below; Hermes captures
  the call, cancels the run, and returns it to ``run_conversation()`` as a
  normal OpenAI ``tool_call``.  Every Hermes mechanism (approvals, budget,
  interrupts, agent-level todo/memory interception) keeps working because the
  tool executes inside Hermes's own loop, exactly like any other provider.

Billing goes to the user's own ``CURSOR_API_KEY`` — Hermes never proxies or
resells Cursor inference.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from agent.cursor_bridge_transport import (
    ConnectJsonTransport,
    CursorBridgeError,
    CursorBridgeProcess,
    resolve_bridge_command,
)
from agent.cursor_bridge_wire import (
    decode_call_custom_tool_request,
    encode_call_custom_tool_response,
)
from agent.openai_sdk_imports import load_chat_completion_message_tool_call_types

logger = logging.getLogger(__name__)

BRIDGE_MARKER_BASE_URL = "sdkbridge://cursor"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_CALLBACK_PATH = "/sdk.v1.SdkCustomToolCallbackService/CallCustomTool"
_TERMINAL_STATUSES = {
    "RUN_LIFECYCLE_STATUS_FINISHED",
    "RUN_LIFECYCLE_STATUS_ERROR",
    "RUN_LIFECYCLE_STATUS_CANCELLED",
    "RUN_LIFECYCLE_STATUS_EXPIRED",
}

_DEFERRED_TOOL_RESULT = {
    "status": "deferred",
    "detail": "Hermes executes this tool and will continue the conversation.",
}


def load_bridge_settings() -> dict[str, Any]:
    """Read the ``cursor_bridge`` config section with safe defaults."""
    settings: dict[str, Any] = {
        "command": "",
        "download_version": "",
    }
    try:
        # Lazy import: hermes_cli.config imports provider plugins which may
        # import agent modules — importing it at module load would cycle.
        from hermes_cli.config import load_config

        section = load_config().get("cursor_bridge")
        if isinstance(section, dict):
            for key in settings:
                if key in section and section[key] is not None:
                    settings[key] = section[key]
    except Exception as exc:  # config errors must never break the client
        logger.debug("cursor_bridge config load failed: %s", exc)
    return settings


# ── Prompt formatting ─────────────────────────────────────────────────────


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def format_messages_as_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten an OpenAI-shaped conversation into one Cursor prompt.

    The Cursor SDK does not allow customizing the system prompt, so the
    Hermes system message and history travel as user-message text.  Tool
    calls/results from earlier Hermes loop iterations are rendered inline so
    the model has the full working context each turn.
    """
    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool result",
        }.get(role, role.title() or "Context")

        rendered = _render_message_content(message.get("content"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            call_lines = []
            for call in tool_calls:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict):
                    call_lines.append(
                        f"[tool call] {fn.get('name')}({fn.get('arguments', '{}')})"
                    )
            if call_lines:
                rendered = "\n".join(filter(None, [rendered, *call_lines]))
        if not rendered:
            continue
        transcript.append(f"{label}:\n{rendered}")

    sections = [
        "You are the model backend for the Hermes agent. The transcript below "
        "is the full conversation so far, including earlier tool activity.",
        "When you need a tool, call one of your available tools directly — "
        "do not describe or simulate tool calls in text.",
    ]
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(sections)


def convert_openai_tools_to_custom_tools(
    tools: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """OpenAI function schemas → sdk.v1 ``custom_tools`` map."""
    custom: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        parameters = fn.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            parameters = {"type": "object", "properties": {}}
        entry: dict[str, Any] = {"inputSchema": parameters}
        description = fn.get("description")
        if isinstance(description, str) and description.strip():
            entry["description"] = description.strip()
        custom[name] = entry
    return custom


# ── Tool callback server ──────────────────────────────────────────────────


class _CallbackHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "hermes-cursor-callback"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("cursor tool callback: " + fmt, *args)

    def handle_one_request(self) -> None:
        # Loop mode cancels the Cursor run after capturing a tool call, so the
        # bridge routinely resets in-flight/idle keep-alive callback
        # connections. Swallow the resulting reset instead of letting
        # socketserver dump a ConnectionResetError traceback to stderr.
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            self.close_connection = True
        except OSError as exc:
            logger.debug("cursor tool callback connection closed: %s", exc)
            self.close_connection = True

    def _read_body(self) -> bytes:
        transfer_encoding = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in transfer_encoding:
            # Minimal chunked decoder — http.server does not decode chunked
            # request bodies, and the bridge may send them (see sdk-bridge
            # docs/services.md).
            body = bytearray()
            while True:
                size_line = self.rfile.readline(65536).strip()
                if not size_line:
                    raise ValueError("invalid chunk size")
                if b";" in size_line:
                    size_line = size_line.split(b";", 1)[0]
                try:
                    size = int(size_line, 16)
                except ValueError as exc:
                    raise ValueError("invalid chunk size") from exc
                if size == 0:
                    self.rfile.readline(65536)
                    return bytes(body)
                if len(body) + size > 1 * 1024 * 1024:
                    raise ValueError("callback body exceeds size limit")
                chunk = self.rfile.read(size)
                if len(chunk) != size:
                    raise ValueError("truncated callback chunk")
                body.extend(chunk)
                if self.rfile.read(2) != b"\r\n":
                    raise ValueError("invalid chunk terminator")
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0 or length > 1 * 1024 * 1024:
            raise ValueError("callback body exceeds size limit")
        return self.rfile.read(length) if length else b""

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_connect_error(self, http_status: int, code: str, message: str) -> None:
        payload = json.dumps({"code": code, "message": message}).encode("utf-8")
        self._respond(http_status, "application/json", payload)

    def do_POST(self) -> None:  # noqa: N802 — http.server API
        server: _ToolCallbackServer = self.server  # type: ignore[assignment]
        auth = self.headers.get("Authorization") or ""
        token = auth[len("Bearer "):].strip() if auth.startswith("Bearer ") else ""
        if not token or not secrets.compare_digest(token, server.auth_token):
            self._respond_connect_error(401, "unauthenticated", "invalid callback token")
            return
        if self.path.rstrip("/") != _CALLBACK_PATH:
            self._respond_connect_error(404, "unimplemented", f"unknown RPC {self.path}")
            return

        try:
            raw = self._read_body()
        except (OSError, ValueError) as exc:
            self._respond_connect_error(400, "invalid_argument", f"unreadable body: {exc}")
            return

        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        is_proto = content_type == "application/proto"
        try:
            if is_proto:
                request = decode_call_custom_tool_request(raw)
            else:
                request = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(request, dict):
                    raise ValueError("request body is not a JSON object")
        except (ValueError, UnicodeDecodeError) as exc:
            self._respond_connect_error(400, "invalid_argument", f"bad request body: {exc}")
            return

        try:
            result = server.handler(request)
        except Exception as exc:  # tool failures must not kill the stream
            logger.warning(
                "cursor tool callback handler failed (%s)", type(exc).__name__
            )
            result = {"error": "Hermes rejected the tool callback"}
        if not isinstance(result, dict):
            result = {"value": str(result)}

        if is_proto:
            self._respond(200, "application/proto", encode_call_custom_tool_response(result))
        else:
            body = json.dumps({"result": result}).encode("utf-8")
            self._respond(200, "application/json", body)


class _ToolCallbackServer(ThreadingHTTPServer):
    """Loopback Connect server implementing SdkCustomToolCallbackService."""

    daemon_threads = True

    def __init__(self, handler: Callable[[dict[str, Any]], dict[str, Any]]):
        super().__init__(("127.0.0.1", 0), _CallbackHandler)
        self.handler = handler
        self.auth_token = secrets.token_urlsafe(32)
        self._thread = threading.Thread(
            target=self.serve_forever, daemon=True, name="cursor-tool-callback"
        )

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            self.shutdown()
            self.server_close()
        except OSError:
            pass

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Backstop for the whole request lifecycle (read, dispatch, response
        # flush): a bridge that resets a callback connection — which loop mode
        # provokes on every CancelRun — must not print a traceback to stderr.
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        logger.debug("cursor tool callback server error from %s", client_address, exc_info=True)


# ── Active-run bookkeeping ────────────────────────────────────────────────


class _ActiveRun:
    """Per-request state shared between the stream loop and tool callbacks."""

    def __init__(self, agent_id: str, tool_names: set[str], deadline: float):
        self.agent_id = agent_id
        self.tool_names = tool_names
        self.deadline = deadline
        self.lock = threading.Lock()
        self.captured_calls: list[dict[str, Any]] = []
        self.cancel_requested = False

    def capture(self, request: dict[str, Any]) -> None:
        with self.lock:
            self.captured_calls.append(request)


# ── OpenAI-compatible facade ──────────────────────────────────────────────


class _BridgeChatCompletions:
    def __init__(self, client: "CursorBridgeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _BridgeChatNamespace:
    def __init__(self, client: "CursorBridgeClient"):
        self.completions = _BridgeChatCompletions(client)


class CursorBridgeClient:
    """Minimal OpenAI-client-compatible facade for the Cursor SDK bridge."""

    supports_abort_inflight = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        workspace: str | None = None,
        bridge_command: str | None = None,
        tool_mode: str | None = None,
        builtin_tools: bool | None = None,
        tool_dispatcher: Callable[[str, dict[str, Any], str | None], Any] | None = None,
        **_: Any,
    ):
        settings = load_bridge_settings()
        self.api_key = (api_key or os.getenv("CURSOR_API_KEY", "")).strip()
        self.base_url = base_url or BRIDGE_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._workspace = str(Path(workspace or os.getcwd()).resolve())
        self._bridge_command = bridge_command or str(settings.get("command") or "")
        requested_mode = (tool_mode or "loop").strip().lower()
        requested_builtin_tools = (
            bool(builtin_tools)
            if builtin_tools is not None
            else False
        )
        if requested_mode != "loop":
            raise CursorBridgeError(
                "Cursor harness mode is unsupported because it bypasses Hermes tool controls"
            )
        if requested_builtin_tools:
            raise CursorBridgeError(
                "Cursor built-in tools are unsupported because they bypass Hermes approvals"
            )
        if tool_dispatcher is not None:
            raise CursorBridgeError("inline Cursor tool dispatch is unsupported")

        self.chat = _BridgeChatNamespace(self)
        self.is_closed = False
        self._lock = threading.Lock()
        self._process: CursorBridgeProcess | None = None
        self._transport: ConnectJsonTransport | None = None
        self._callback_server: _ToolCallbackServer | None = None
        self._active_runs: dict[str, _ActiveRun] = {}
        self._runs_lock = threading.Lock()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────

    def _ensure_bridge(self) -> ConnectJsonTransport:
        with self._lock:
            if self._transport is not None and self._process is not None and self._process.is_alive():
                return self._transport

            if not self.api_key or self.api_key == "cursor":
                # Fall back to Hermes's profile-scoped Cursor credential,
                # filled by `hermes cursor login`.
                from agent.cursor_sdk_auth import read_sdk_credentials

                stored = read_sdk_credentials()
                if stored:
                    self.api_key = str(stored["apiKey"])
                else:
                    raise CursorBridgeError(
                        "No Cursor credential available. Run `hermes cursor login` "
                        "(browser login on your Cursor account), or add "
                        "CURSOR_API_KEY to ~/.hermes/.env "
                        "(cursor.com/dashboard → API Keys)."
                    )

            command = resolve_bridge_command(self._bridge_command)
            if not command:
                raise CursorBridgeError(
                    "Cursor SDK bridge not found. Run `hermes model` and pick Cursor "
                    "to install it, `pip install cursor-sdk`, or set "
                    "cursor_bridge.command in config.yaml."
                )

            if self._callback_server is None:
                self._callback_server = _ToolCallbackServer(self._handle_tool_callback)
                self._callback_server.start()

            process: CursorBridgeProcess | None = None
            try:
                process = CursorBridgeProcess(
                    command=command,
                    api_key=self.api_key,
                    workspace=self._workspace,
                    tool_callback_url=self._callback_server.url,
                    tool_callback_auth_token=self._callback_server.auth_token,
                )
                endpoint = process.start()
                transport = ConnectJsonTransport(endpoint.url, endpoint.auth_token)
                self._verify_capabilities(transport)
            except Exception:
                self._transport = None
                self._process = None
                if process is not None:
                    process.stop()
                if self._callback_server is not None:
                    self._callback_server.stop()
                    self._callback_server = None
                raise
            assert process is not None
            self._process = process
            self._transport = transport
            self.is_closed = False
            return transport

    @staticmethod
    def _verify_capabilities(transport: ConnectJsonTransport) -> None:
        """Negotiate the sdk.v1 control surface before serving requests."""
        try:
            result = transport.unary(
                "SdkBridgeControlService", "Ping", {}, timeout=5.0
            )
        except CursorBridgeError as exc:
            raise CursorBridgeError(
                "Cursor SDK bridge capability handshake failed"
            ) from exc
        if result.get("ok") is False or result.get("ready") is False:
            raise CursorBridgeError("Cursor SDK bridge did not pass capability handshake")
        try:
            version = transport.unary(
                "SdkBridgeControlService", "GetVersion", {}, timeout=5.0
            )
        except CursorBridgeError as exc:
            raise CursorBridgeError(
                "Cursor SDK bridge version negotiation failed"
            ) from exc
        if version.get("protocolVersion") != "sdk.v1":
            raise CursorBridgeError(
                "Cursor SDK bridge does not advertise the required sdk.v1 protocol"
            )

    def close(self) -> None:
        with self._lock:
            self.is_closed = True
            transport, self._transport = self._transport, None
            process, self._process = self._process, None
            callback_server, self._callback_server = self._callback_server, None
        if transport is not None:
            try:
                transport.unary(
                    "SdkBridgeControlService",
                    "Shutdown",
                    {"graceSeconds": 0},
                    timeout=3.0,
                )
            except CursorBridgeError:
                pass
        if process is not None:
            process.stop()
        if callback_server is not None:
            callback_server.stop()

    def abort_inflight(self) -> None:
        """Abort an in-flight bridge request from a watchdog/control thread.

        The bridge is a child process rather than an httpx socket pool. Killing
        that child unblocks the request-owning thread without cross-thread file
        descriptor closes; the owner later runs the ordinary cached-client
        teardown path.
        """
        with self._lock:
            self.is_closed = True
            self._transport = None
            process, self._process = self._process, None
            callback_server, self._callback_server = self._callback_server, None
        if process is not None:
            process.stop()
        if callback_server is not None:
            callback_server.stop()

    # ── tool callbacks ───────────────────────────────────────────────────

    def _handle_tool_callback(self, request: dict[str, Any]) -> dict[str, Any]:
        tool_name = str(request.get("toolName") or "").strip()
        agent_id = str(request.get("agentId") or "").strip()
        args = request.get("args")
        if not isinstance(args, dict):
            args = {}

        with self._runs_lock:
            run = self._active_runs.get(agent_id)
        if run is None:
            return {"error": f"no active Hermes run for agent {agent_id!r}"}

        if tool_name not in run.tool_names:
            return {"error": "tool is not declared for this Hermes run"}

        # Capture the call, hand it back to Hermes's loop, and
        # cancel the Cursor run so the model does not keep iterating on the
        # placeholder result below.
        run.capture(
            {
                "toolName": tool_name,
                "args": args,
                "toolCallId": request.get("toolCallId"),
            }
        )
        self._request_cancel(run)
        return dict(_DEFERRED_TOOL_RESULT)

    def _request_cancel(self, run: _ActiveRun) -> None:
        with run.lock:
            if run.cancel_requested:
                return
            run.cancel_requested = True

        def cancel() -> None:
            try:
                transport = self._transport
                if transport is None:
                    return
                remaining = run.deadline - time.monotonic()
                if remaining <= 0:
                    return
                listing = transport.unary(
                    "SdkAgentService",
                    "ListRuns",
                    {"agentId": run.agent_id, "options": {"limit": 5}},
                    timeout=max(0.1, min(5.0, remaining)),
                )
                for item in listing.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status") or "")
                    run_id = str(item.get("runId") or "")
                    if run_id and status not in _TERMINAL_STATUSES:
                        remaining = run.deadline - time.monotonic()
                        if remaining <= 0:
                            return
                        transport.unary(
                            "SdkAgentService",
                            "CancelRun",
                            {"runId": run_id, "agentId": run.agent_id},
                            timeout=max(0.1, min(5.0, remaining)),
                        )
                        return
            except CursorBridgeError as exc:
                logger.debug("cursor run cancel failed (continuing): %s", exc)

        threading.Thread(target=cancel, daemon=True, name="cursor-run-cancel").start()

    # ── request execution ────────────────────────────────────────────────

    def _next_call_id(self) -> str:
        with self._call_counter_lock:
            self._call_counter += 1
            return f"cursor_call_{self._call_counter}"

    @staticmethod
    def _normalize_timeout(timeout: Any) -> float:
        if timeout is None:
            return _DEFAULT_TIMEOUT_SECONDS
        if isinstance(timeout, (int, float)):
            return float(timeout)
        candidates = [
            getattr(timeout, attr, None)
            for attr in ("read", "write", "connect", "pool", "timeout")
        ]
        numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
        return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        del tool_choice  # the Cursor harness decides tool use on its own
        transport = self._ensure_bridge()
        effective_timeout = self._normalize_timeout(timeout)
        deadline = time.monotonic() + effective_timeout

        model_id = (model or "").strip()
        if not model_id or model_id.lower() == "cursor":
            model_id = "auto"

        custom_tools = convert_openai_tools_to_custom_tools(tools)
        options: dict[str, Any] = {
            "model": {"id": model_id},
            "name": "hermes",
            "local": {"cwd": [self._workspace]},
            "mode": "AGENT_MODE_OPTION_AGENT",
            "apiKey": self.api_key,
        }
        if custom_tools:
            options["local"]["customTools"] = custom_tools
            # Cursor SDK 1.0.27 exposes custom tools through its built-in MCP
            # capability; isolate settings so external MCP servers cannot enter
            # Hermes's tool channel.
            options["local"]["settingSources"] = []
            options["tools"] = {"names": ["mcp"]}
        else:
            # An empty tool allowlist keeps tool-free runs text-only.
            options["tools"] = {"names": []}

        create_timeout = min(60.0, max(0.1, deadline - time.monotonic()))
        created = transport.unary(
            "SdkAgentService", "CreateAgent", {"options": options}, timeout=create_timeout
        )
        agent_id = str(created.get("agentId") or "")
        if not agent_id:
            raise CursorBridgeError("CreateAgent returned no agentId")

        run = _ActiveRun(agent_id, set(custom_tools), deadline)
        with self._runs_lock:
            self._active_runs[agent_id] = run
        prompt_text = format_messages_as_prompt(messages or [])

        final_text = ""
        usage_payload: dict[str, Any] = {}
        status = ""
        error_code = None
        stream_error: CursorBridgeError | None = None
        try:
            stream_iter = transport.server_stream(
                "SdkAgentService",
                "Send",
                {
                    "agentId": agent_id,
                    "message": {"text": prompt_text},
                    "options": {},
                },
                deadline=deadline,
            )
            for message in stream_iter:
                result_env = message.get("result")
                if isinstance(result_env, dict):
                    status = str(result_env.get("status") or "")
                    error_code = result_env.get("errorCode")
                    run_result = result_env.get("result")
                    if isinstance(run_result, dict):
                        final_text = str(run_result.get("result") or "")
                        maybe_usage = run_result.get("usage")
                        if isinstance(maybe_usage, dict):
                            usage_payload = maybe_usage
                    continue
                if "done" in message:
                    break
                # sdk_message / interaction_update / keepalives: ignored.
        except CursorBridgeError as exc:
            stream_error = exc
        finally:
            with self._runs_lock:
                self._active_runs.pop(agent_id, None)
            self._cleanup_agent(transport, agent_id, deadline)

        captured = list(run.captured_calls)
        if stream_error is not None and not captured:
            raise stream_error
        if not captured and status and status not in {
            "RUN_LIFECYCLE_STATUS_FINISHED",
        }:
            code_note = f" (code={error_code})" if error_code else ""
            raise CursorBridgeError(
                f"Cursor run ended with status {status}{code_note}: "
                f"{final_text or 'no result text'}",
                code=str(error_code) if error_code else None,
            )

        tool_calls = []
        if captured:
            tool_call_cls, function_cls = load_chat_completion_message_tool_call_types()
        for call in captured:
            call_id = str(call.get("toolCallId") or self._next_call_id())
            tool_calls.append(tool_call_cls(
                id=call_id,
                response_item_id=None,
                type="function",
                function=function_cls(
                    name=str(call.get("toolName") or ""),
                    arguments=json.dumps(call.get("args") or {}, ensure_ascii=False),
                ),
            ))

        usage = self._usage_namespace(usage_payload)
        assistant_message = SimpleNamespace(
            content=(final_text or None) if not tool_calls else None,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=assistant_message, finish_reason=finish_reason)],
            usage=usage,
            model=model_id,
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _cleanup_agent(
        self, transport: ConnectJsonTransport, agent_id: str, deadline: float
    ) -> None:
        """Delete the per-request agent so bridge-store state does not pile up."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.debug("cursor agent cleanup skipped after request deadline for %s", agent_id)
            return
        try:
            transport.unary(
                "SdkAgentService",
                "DeleteAgent",
                {"agentId": agent_id, "options": {"cwd": self._workspace}},
                timeout=max(0.1, min(5.0, remaining)),
            )
        except CursorBridgeError as exc:
            logger.debug("cursor agent cleanup failed for %s: %s", agent_id, exc)

    @staticmethod
    def _usage_namespace(usage_payload: dict[str, Any]) -> SimpleNamespace:
        def as_int(key: str) -> int:
            value = usage_payload.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        input_tokens = as_int("inputTokens")
        output_tokens = as_int("outputTokens")
        total = as_int("totalTokens") or (input_tokens + output_tokens)
        return SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total,
            prompt_tokens_details=SimpleNamespace(cached_tokens=as_int("cacheReadTokens")),
        )

    # ── catalog ──────────────────────────────────────────────────────────

    def list_models(self) -> list[dict[str, Any]]:
        """Return the account's model catalog (SdkModel dicts)."""
        transport = self._ensure_bridge()
        response = transport.unary(
            "SdkCursorService",
            "ListModels",
            {"options": {"apiKey": self.api_key}},
            timeout=30.0,
        )
        items = response.get("items")
        return [item for item in items or [] if isinstance(item, dict)]


class AsyncCursorBridgeClient:
    """Cancellation-aware async facade over the synchronous bridge client.

    The bridge transport is blocking by design; every operation runs in a worker
    thread so auxiliary async callers never block the event loop. Cancellation
    stops awaiting the worker and ``close`` tears down the child/callback server.
    """

    def __init__(self, **kwargs: Any):
        import asyncio

        self._asyncio = asyncio
        self._sync = CursorBridgeClient(**kwargs)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    async def _create_completion(self, **kwargs: Any) -> Any:
        task = self._asyncio.create_task(
            self._asyncio.to_thread(self._sync.chat.completions.create, **kwargs)
        )
        try:
            return await task
        except self._asyncio.CancelledError:
            await self._asyncio.shield(
                self._asyncio.to_thread(self._sync.abort_inflight)
            )
            raise

    async def list_models(self) -> list[dict[str, Any]]:
        return await self._asyncio.to_thread(self._sync.list_models)

    async def close(self) -> None:
        await self._asyncio.to_thread(self._sync.close)


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot bridge response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = [
            SimpleNamespace(
                index=index,
                id=getattr(tool_call, "id", None),
                type=getattr(tool_call, "type", "function"),
                function=SimpleNamespace(
                    name=getattr(tool_call.function, "name", None),
                    arguments=getattr(tool_call.function, "arguments", None),
                ),
            )
            for index, tool_call in enumerate(message.tool_calls)
        ]

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=None,
        reasoning=None,
    )
    data_chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(choices=[], model=completion.model, usage=completion.usage)
    return [data_chunk, usage_chunk]
