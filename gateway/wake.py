"""Wake an existing agent session from a background completion event.

Two delivery strategies, selected by the target adapter's
``supports_async_delivery`` capability flag:

* Push-capable adapters (telegram, discord, plugin platforms, ...): inject a
  synthetic ``MessageEvent(internal=True)`` through ``adapter.handle_message``
  — the pre-existing wake path, preserved exactly.

* Stateless request/response adapters (the API server,
  ``supports_async_delivery = False``): ``handle_message`` would run the wake
  turn under a ``build_session_key()``-derived key
  (``agent:main:api_server:group:<sid>``) that NEVER matches the raw
  ``X-Hermes-Session-Id`` key real gateway/HQ turns run under
  (``_bind_api_server_session``), so the wake lands in a parallel, invisible
  session. Instead we self-POST ``/v1/chat/completions`` on the in-pod API
  server with the raw session id in the ``X-Hermes-Session-Id`` header — the
  exact entry point real turns use — so the wake turn resumes the REAL
  session, with full history, and its result is visible the next time the
  client polls/reopens the conversation.

Failures RAISE (after bounded retries on transient errors) so callers can
rewind cursors / retry instead of silently losing the event.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A wake self-post runs the entire agent turn synchronously (stream=false);
# generous ceiling so long tool-using turns aren't killed mid-flight.
WAKE_TURN_TIMEOUT_SECONDS = 600.0

# Backoff delays between retries on transient failures (429 concurrency cap,
# connection errors). The API server has no per-session lock — concurrent
# turns on one session are last-writer-wins — but it DOES enforce a global
# max_concurrent_runs cap via HTTP 429, which is worth waiting out.
_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)


def adapter_supports_push(adapter: Any) -> bool:
    """Whether this adapter can push a message to the user after a turn ends.

    Mirrors ``gateway.session_context.async_delivery_supported`` but reads the
    capability off the adapter class (``supports_async_delivery``) instead of
    the request-scoped contextvar — background watchers run outside any bound
    session context. Adapters that don't declare the flag are push-capable.
    """
    return bool(getattr(adapter, "supports_async_delivery", True))


async def deliver_wake(
    adapter: Any,
    *,
    text: str,
    session_id: str = "",
    source: Any = None,
    profile: str = "",
    route_profile: str = "",
    display_kind: str = "internal_notification",
    display_metadata: dict[str, Any] | None = None,
) -> None:
    """Deliver a wake turn to the session behind ``adapter``.

    ``session_id`` is the RAW session id (the ``X-Hermes-Session-Id`` value /
    ``state.db`` key) — required for non-push adapters. ``source`` is the
    ``SessionSource`` used to build the synthetic event — required for
    push-capable adapters.

    Raises on failure (bad arguments, exhausted retries, HTTP error) so the
    caller can rewind/retry instead of treating the wake as delivered.
    """
    from agent.message_provenance import provenance_for_gateway_ingress
    from gateway.internal_turn import build_internal_turn_envelope

    display_envelope = build_internal_turn_envelope(
        display_kind,
        display_metadata
        or {
            "source": "gateway_wake",
            "internal": True,
            "kind": "wake",
        },
    )
    bounded_display = display_envelope["display_metadata"]
    source_platform = getattr(getattr(source, "platform", None), "value", None)
    platform = str(source_platform or "api_server")
    provenance_metadata: dict[str, Any] = {
        "producer": "gateway_wake",
        "source": bounded_display["source"],
        "event_kind": bounded_display["kind"],
        "platform": platform,
    }
    for key in (
        "event_id",
        "task_id",
        "job_id",
        "process_id",
        "delegation_id",
        "run_id",
        "generation_id",
    ):
        value = bounded_display.get(key)
        if value is not None and value != "":
            provenance_metadata[key] = value
    if session_id:
        provenance_metadata["session_id"] = session_id
    provenance = provenance_for_gateway_ingress(
        platform=platform,
        internal=True,
        internal_source=bounded_display["source"],
        event_kind=bounded_display["kind"],
        metadata=provenance_metadata,
    )
    envelope = build_internal_turn_envelope(
        display_envelope["display_kind"],
        bounded_display,
        provenance.as_message_fields(),
    )

    if adapter_supports_push(adapter):
        if source is None:
            raise ValueError(
                "deliver_wake: push-capable adapter requires a SessionSource"
            )
        from gateway.platforms.base import MessageEvent, MessageType

        synth_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
            metadata=envelope["display_metadata"],
        )
        await adapter.handle_message(synth_event)
        return

    if not session_id:
        raise ValueError(
            "deliver_wake: non-push adapter (supports_async_delivery=False) "
            "requires the raw session id to self-post the wake turn"
        )
    await _self_post_chat_completion(
        adapter,
        text=text,
        session_id=session_id,
        profile=profile,
        route_profile=route_profile,
        internal_turn=envelope,
    )


async def _self_post_chat_completion(
    adapter: Any,
    *,
    text: str,
    session_id: str,
    internal_turn: dict[str, Any],
    profile: str = "",
    route_profile: str = "",
) -> None:
    """POST the wake text to the in-pod API server as a normal session turn.

    Uses the adapter's own bind host/port/key (``ApiServerAdapter.__init__``).
    Session continuation via ``X-Hermes-Session-Id`` is 403-gated on
    ``API_SERVER_KEY`` being configured, so a missing key is a hard error —
    raise loudly rather than run the wake in a fresh fingerprint-derived
    session nobody is looking at.
    """
    import aiohttp

    host = str(getattr(adapter, "_host", "") or "127.0.0.1")
    if host in ("0.0.0.0", "::", "*"):
        # Wildcard bind address — connect over loopback.
        host = "127.0.0.1"
    port = int(getattr(adapter, "_port", 0) or 8642)
    target_resolver = getattr(adapter, "_wake_request_target", None)
    if callable(target_resolver):
        path, api_key = target_resolver(
            profile=profile,
            route_profile=route_profile,
        )
    else:
        # Compatibility for adapter-like test doubles and third-party
        # stateless adapters. They may use the historical default route, but
        # can never claim to deliver a named-profile wake with the listener
        # owner's credential.
        if str(profile or "").strip() not in {"", "default"} or str(
            route_profile or ""
        ).strip() not in {"", "default"}:
            raise RuntimeError(
                "wake self-post adapter cannot resolve named-profile "
                "credentials; refusing default-profile fallback"
            )
        api_key = str(getattr(adapter, "_api_key", "") or "")
        path = (
            "/p/default/v1/chat/completions"
            if str(route_profile or "").strip() == "default"
            else "/v1/chat/completions"
        )
        if not api_key:
            raise RuntimeError(
                "wake self-post requires API_SERVER_KEY: session continuation via "
                "X-Hermes-Session-Id is rejected (403) on an unauthenticated API "
                "server, so the wake cannot reach the target session"
            )

    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Hermes-Session-Id": session_id,
    }
    from gateway.internal_turn import INTERNAL_TURN_FIELD

    payload = {
        "model": str(getattr(adapter, "_model_name", "") or "hermes-agent"),
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        INTERNAL_TURN_FIELD: internal_turn,
    }
    idempotency_material = {
        "version": 1,
        "session_id": session_id,
        "profile": str(profile or "").strip(),
        "route_profile": str(route_profile or "").strip(),
        "path": path,
        "text": text,
        "internal_turn": internal_turn,
    }
    canonical = json.dumps(
        idempotency_material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    # Stable for every retry of one logical wake (including an outer queue
    # rewind), distinct across event identity/destination, and bounded well
    # below common proxy header limits. The digest prevents session ids or
    # internal metadata from becoming observable header values.
    headers["Idempotency-Key"] = (
        "hermes-wake-v1-" + hashlib.sha256(canonical).hexdigest()
    )

    last_err: Optional[BaseException] = None
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            timeout = aiohttp.ClientTimeout(total=WAKE_TURN_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 429:
                        # Global concurrency cap (max_concurrent_runs) —
                        # transient; back off and retry.
                        last_err = RuntimeError(
                            f"wake self-post got HTTP 429 (concurrency cap) "
                            f"for session {session_id}"
                        )
                        logger.warning(
                            "%s; attempt %d/%d", last_err, attempt + 1, attempts
                        )
                        continue
                    if resp.status >= 400:
                        body = (await resp.text())[:300]
                        # Non-transient (auth/validation) — fail immediately.
                        raise RuntimeError(
                            f"wake self-post failed for session {session_id}: "
                            f"HTTP {resp.status}: {body}"
                        )
                    await resp.read()
                    logger.info(
                        "wake self-post delivered for session %s (attempt %d)",
                        session_id,
                        attempt + 1,
                    )
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_err = exc
            logger.warning(
                "wake self-post transient failure for session %s "
                "(attempt %d/%d): %s",
                session_id,
                attempt + 1,
                attempts,
                exc,
            )
            continue
    raise RuntimeError(
        f"wake self-post gave up for session {session_id} after "
        f"{attempts} attempts: {last_err}"
    ) from last_err
