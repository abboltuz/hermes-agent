"""Behavioral contracts for the continuable-session compression coordinator."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from agent.compression_v3 import (
    CompressionBudget,
    ContextProjectionUnfit,
    CompressionCandidate,
    CompressionCoordinator,
    CompressionRequest,
    BackgroundCompressionConfig,
    build_background_snapshot,
    build_policy_capsule,
    compression_route_is_eligible,
    ensure_compression_coordinator,
    emergency_context_cut,
    validate_projection,
    prepare_api_request,
    prune_tool_pressure_projection,
    estimate_projection_tokens,
    provider_request_budget,
    resolve_background_compression_config,
    run_background_compression_worker,
)
from agent.compression_v3 import _provider_wire_token_bound
from agent.compression_v3 import _bind_recovery_identity
from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request


HUMAN = {"origin_kind": "human_user", "turn_kind": "prompt", "trust_kind": "user_authorized"}


def test_provider_wire_estimate_stays_in_token_domain_with_explicit_overhead():
    request = {
        "instructions": 'ASCII compact id 😀 中文 "quotes" \\n',
        "input": [{"role": "user", "content": [{"type": "text", "text": "\\ud800"}]}],
        "tools": [{"type": "function", "name": "x", "parameters": {"required": ["x"]}}],
        "max_output_tokens": 17,
    }
    serialized = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    serialized_bytes = len(serialized.encode("utf-8"))
    estimate = _provider_wire_token_bound(request)

    assert estimate >= (serialized_bytes + 2) // 3
    assert estimate < serialized_bytes


def test_provider_wire_estimate_does_not_measure_unicode_as_ascii_escapes():
    request = {
        "instructions": "Отвечай по-русски. " * 1_000,
        "input": [{"role": "user", "content": "продолжай работу " * 2_000}],
        "max_output_tokens": 1_024,
    }
    escaped_bytes = len(
        json.dumps(request, ensure_ascii=True, separators=(",", ":")).encode()
    )
    utf8_bytes = len(
        json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    )

    estimate = _provider_wire_token_bound(request)

    assert escaped_bytes > utf8_bytes * 2
    assert estimate < escaped_bytes // 3
    assert estimate >= (utf8_bytes + 2) // 3


def test_provider_request_budget_uses_active_model_window_and_output_reserve():
    agent = SimpleNamespace(
        _config_context_length=12_000,
        context_compressor=SimpleNamespace(context_length=12_000),
        _compression_safety_margin=500,
    )
    request = {
        "input": [{"role": "user", "content": "x" * 6_000}],
        "max_output_tokens": 2_000,
    }

    decision = provider_request_budget(agent, request)

    assert decision.context_window == 12_000
    assert decision.output_reserve == 2_000
    assert decision.safety_margin == 500
    assert decision.safe_input_budget == 9_500
    assert decision.fits is True


def test_provider_request_budget_honors_runtime_context_downgrade():
    agent = SimpleNamespace(
        _config_context_length=1_000_000,
        context_compressor=SimpleNamespace(context_length=200_000),
        _compression_safety_margin=1_024,
    )
    request = {
        "input": [{"role": "user", "content": "x" * 600_000}],
        "max_output_tokens": 16_000,
    }

    decision = provider_request_budget(agent, request)

    assert decision.context_window == 200_000
    assert decision.safe_input_budget == 182_976
    assert decision.estimated_input_tokens > decision.safe_input_budget
    assert decision.fits is False


def test_codex_responses_incident_sized_wire_reaches_transport_once():
    """Serialized JSON bytes are not provider tokens for the final fit gate."""
    clients = []
    codex_calls = []
    response = object()
    client = object()

    def make_client(reason, **_kwargs):
        clients.append(reason)
        return client

    def run_codex(request, **kwargs):
        codex_calls.append((request, kwargs))
        return response

    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        _config_context_length=272_000,
        _compression_safety_margin=1_024,
        _run_codex_stream=run_codex,
    )
    request = {
        "model": "gpt-5.6-luna",
        "instructions": "policy " * 5_000,
        "input": [{
            "role": "user",
            "content": [{"type": "input_text", "text": "result " * 20_000}],
        }],
        "tools": [{
            "type": "function",
            "name": "write_file",
            "description": "schema " * 15_000,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        }],
        "max_output_tokens": 16_000,
    }
    serialized_bytes = len(
        json.dumps(request, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    )
    assert serialized_bytes > agent._config_context_length

    assert _dispatch_nonstreaming_api_request(
        agent, request, make_client=make_client
    ) is response
    assert clients == ["codex_stream_request"]
    assert len(codex_calls) == 1
    prepared, kwargs = codex_calls[0]
    assert prepared["instructions"] == request["instructions"]
    assert prepared["input"] == request["input"]
    assert prepared["tools"] == request["tools"]
    assert kwargs["client"] is client


def test_prepare_api_request_strips_private_sidecars_recursively():
    agent = type("Agent", (), {"_config_context_length": 100_000, "_compression_safety_margin": 0})()
    request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "ok", "_row_id": 3}], "_row_id": 2, "_db_persisted": True}],
        "tools": [{"type": "function", "function": {"name": "x", "_compression_capsule": True}}],
        "input": [{"role": "user", "content": {"nested": {"_row_id": 4}}}],
        "max_tokens": 1,
    }
    prepared = prepare_api_request(agent, request)

    def walk(value):
        if isinstance(value, dict):
            assert not any(key in {"_row_id", "_db_persisted", "_compression_capsule"} for key in value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(prepared)
    assert request["messages"][0]["_row_id"] == 2


@pytest.mark.parametrize("api_mode", ["codex_responses", "anthropic_messages", "chat_completions"])
def test_shared_dispatch_refuses_oversized_native_wire_before_client_call(api_mode):
    agent = type("Agent", (), {
        "api_mode": api_mode,
        "provider": "openrouter",
        "_config_context_length": 32,
        "_compression_safety_margin": 0,
    })()
    calls = []

    def make_client(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("provider client must not be created")

    request = {
        "instructions": "irreducible policy " * 40,
        "input": [{"role": "user", "content": "irreducible input " * 40}],
        "messages": None,
        "tools": [{"type": "function", "name": "large", "parameters": {"type": "object"}}],
        "max_output_tokens": 8,
    }
    with pytest.raises(ContextProjectionUnfit) as exc_info:
        _dispatch_nonstreaming_api_request(agent, request, make_client=make_client)
    assert exc_info.value.outcome == "context_projection_unfit"
    assert calls == []


def test_codex_responses_final_boundary_refuses_instructions_input_before_sdk(monkeypatch):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    calls = []
    codex_calls = []
    agent = SimpleNamespace(
        api_mode="codex_responses", provider="openai-codex",
        _config_context_length=32, _compression_safety_margin=0,
        _run_codex_stream=lambda request, **_kwargs: codex_calls.append(request),
    )
    request = {
        "instructions": "policy " * 100,
        "input": [{"role": "user", "content": "task " * 100}],
        "tools": [], "max_output_tokens": 8,
    }

    with pytest.raises(ContextProjectionUnfit):
        _dispatch_nonstreaming_api_request(
            agent, request, make_client=lambda *args, **kwargs: calls.append(args)
        )
    assert calls == []
    assert codex_calls == []


def test_anthropic_messages_final_boundary_refuses_system_messages_tools_before_sdk():
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    calls = []
    anthropic_calls = []
    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider="anthropic",
        _config_context_length=32, _compression_safety_margin=0,
        _anthropic_messages_create=lambda request, **_kwargs: anthropic_calls.append(request),
    )
    request = {
        "system": "policy " * 100,
        "messages": [{"role": "user", "content": "task " * 100}],
        "tools": [{"name": "large", "input_schema": {"type": "object"}}],
        "max_tokens": 8,
    }

    with pytest.raises(ContextProjectionUnfit):
        _dispatch_nonstreaming_api_request(
            agent, request, make_client=lambda *args, **kwargs: calls.append(args)
        )
    assert calls == []
    assert anthropic_calls == []


def test_bedrock_converse_final_boundary_refuses_control_wire_before_boto3(monkeypatch):
    from agent import bedrock_adapter
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    calls = []
    converse_calls = []
    agent = SimpleNamespace(
        api_mode="bedrock_converse", provider="bedrock",
        _config_context_length=32, _compression_safety_margin=0,
    )
    request = {
        "__bedrock_region__": "us-east-1", "__bedrock_converse__": True,
        "system": [{"text": "policy " * 100}],
        "messages": [{"role": "user", "content": [{"text": "task " * 100}]}],
        "inferenceConfig": {"maxTokens": 8},
    }

    monkeypatch.setattr(
        bedrock_adapter, "_get_bedrock_runtime_client",
        lambda _region: SimpleNamespace(
            converse=lambda **kwargs: converse_calls.append(kwargs)
        ),
    )
    with pytest.raises(ContextProjectionUnfit):
        _dispatch_nonstreaming_api_request(
            agent, request, make_client=lambda *args, **kwargs: calls.append(args)
        )
    assert calls == []
    assert converse_calls == []


def test_bedrock_nested_output_reserve_is_checked_before_boto3(monkeypatch):
    """Converse maxTokens consumes context even when serialized input fits."""
    from agent import bedrock_adapter
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    calls = []
    request = {
        "__bedrock_region__": "us-east-1", "__bedrock_converse__": True,
        "system": [{"text": "policy"}],
        "messages": [{"role": "user", "content": [{"text": "task"}]}],
        "inferenceConfig": {"maxTokens": 100},
    }
    # Leave room for the request wire but not for its nested output reserve.
    from agent.compression_v3 import _provider_wire_token_bound, _strip_provider_private
    wire = _provider_wire_token_bound(_strip_provider_private(request))
    agent = SimpleNamespace(
        api_mode="bedrock_converse", provider="bedrock",
        _config_context_length=wire + 99, _compression_safety_margin=0,
    )
    monkeypatch.setattr(
        bedrock_adapter, "_get_bedrock_runtime_client",
        lambda _region: SimpleNamespace(converse=lambda **kwargs: calls.append(kwargs)),
    )
    with pytest.raises(ContextProjectionUnfit):
        _dispatch_nonstreaming_api_request(
            agent, request, make_client=lambda *args, **kwargs: calls.append(args)
        )
    assert calls == []


def test_openai_chat_nonstreaming_final_boundary_refuses_before_completions(monkeypatch):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    calls = []
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="openrouter",
        _config_context_length=32, _compression_safety_margin=0,
    )
    request = {
        "messages": [{"role": "system", "content": "policy " * 100},
                     {"role": "user", "content": "task " * 100}],
        "tools": [{"type": "function", "function": {"name": "large", "parameters": {}}}],
        "max_tokens": 8,
    }
    with pytest.raises(ContextProjectionUnfit):
        _dispatch_nonstreaming_api_request(
            agent, request, make_client=lambda *args, **kwargs: calls.append(args)
        )
    assert calls == []


def test_openai_chat_streaming_final_boundary_refuses_before_completions():
    from agent.chat_completion_helpers import interruptible_streaming_api_call

    calls = []
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="openrouter", platform="cli",
        _config_context_length=32, _compression_safety_margin=0,
        _interrupt_requested=False,
        _interruptible_api_call=lambda request: calls.append(request),
    )
    request = {
        "messages": [{"role": "system", "content": "policy " * 100},
                     {"role": "user", "content": "task " * 100}],
        "tools": [{"type": "function", "function": {"name": "large", "parameters": {}}}],
        "max_tokens": 8,
    }
    with pytest.raises(ContextProjectionUnfit):
        interruptible_streaming_api_call(agent, request)
    assert calls == []


def test_reducible_chat_dispatch_calls_once_with_sanitized_final_wire():
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    captured = []
    response = object()
    completions = SimpleNamespace(create=lambda **kwargs: (captured.append(kwargs), response)[1])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="openrouter",
        _config_context_length=100_000, _compression_safety_margin=0,
    )
    request = {
        "messages": [{"role": "user", "content": "small", "_row_id": 9,
                      "_compression_capsule": "private"}],
        "tools": [], "max_tokens": 8,
    }
    assert _dispatch_nonstreaming_api_request(
        agent, request, make_client=lambda *_args, **_kwargs: client
    ) is response
    assert len(captured) == 1
    assert "_row_id" not in json.dumps(captured[0])
    assert "_compression_capsule" not in json.dumps(captured[0])


def test_oversized_reducible_history_is_cut_before_one_chat_dispatch():
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    captured = []
    response = object()
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: (captured.append(kwargs), response)[1]
            )
        )
    )
    messages = [{"role": "user", "content": "latest request", "_row_id": 1}]
    for number in range(8):
        call_id = f"old-{number}"
        messages.extend([
            {"role": "assistant", "_row_id": 2 + number * 2, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "tool", "_row_id": 3 + number * 2, "tool_call_id": call_id, "content": "old result " * 500},
        ])
    # The original wire is oversized, but the retained/capsule projection is
    # reducible and must reach the SDK exactly once.
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="openrouter",
        _config_context_length=2200, _compression_safety_margin=0,
        session_id="reducible-dispatch",
        _session_db=SimpleNamespace(register_compression_recovery=lambda *_args, **_kwargs: "recovery-1"),
    )
    assert _dispatch_nonstreaming_api_request(
        agent, {"messages": messages, "tools": [], "max_tokens": 8},
        make_client=lambda *_args, **_kwargs: client,
    ) is response
    assert len(captured) == 1
    assert len(captured[0]["messages"]) < len(messages)
    assert (
        _provider_wire_token_bound(captured[0]) + captured[0]["max_tokens"]
        <= agent._config_context_length
    )
    assert "_compression" not in json.dumps(captured[0])


def test_transport_added_tools_make_canonical_fit_unfit_before_openai_sdk():
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    calls = []
    agent = SimpleNamespace(
        api_mode="chat_completions", provider="openrouter",
        _config_context_length=256, _compression_safety_margin=0,
    )
    request = {
        "messages": [{"role": "user", "content": "fits canonically"}],
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {"description": "z" * 500}}}],
        "max_tokens": 8,
    }
    tool_tokens = estimate_projection_tokens(request["tools"])
    canonical_context = estimate_projection_tokens(request["messages"]) + tool_tokens + 8
    agent._config_context_length = canonical_context
    assert CompressionBudget(
        canonical_context, 8, 0, tool_schema_tokens=tool_tokens
    ).fits(request["messages"])
    assert _provider_wire_token_bound(request) + 8 > canonical_context
    with pytest.raises(ContextProjectionUnfit):
        _dispatch_nonstreaming_api_request(
            agent, request, make_client=lambda *args, **kwargs: calls.append(args)
        )
    assert calls == []


def _round(number: int, *, body: str = "result"):
    call_id = f"call-{number}"
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def test_compression_requests_share_logical_owner_and_admit_once():
    db = object()
    first = SimpleNamespace(session_id="physical-a", _session_db=db, _conversation_root_id=lambda: "logical-1")
    second = SimpleNamespace(session_id="physical-b", _session_db=db, _conversation_root_id=lambda: "logical-1")

    owner_a = ensure_compression_coordinator(first, trigger="preflight", urgency=1)
    owner_b = ensure_compression_coordinator(second, trigger="gateway_hygiene", urgency=3)

    assert owner_a is owner_b
    assert owner_b.attempt is not None
    assert owner_b.attempt.trigger == "gateway_hygiene"
    assert owner_b.attempt.urgency == 3
    assert owner_b.admit_execution(CompressionRequest(
        session_id="logical-1", generation=0, trigger="gateway_hygiene",
        urgency=3, source_fingerprint="same", force=False,
    )).outcome == "admitted"
    assert owner_b.admit_execution(CompressionRequest(
        session_id="logical-1", generation=0, trigger="preflight",
        urgency=1, source_fingerprint="same", force=False,
    )).outcome == "joined"


def test_compression_requests_rearm_only_for_changed_source_or_force():
    owner = CompressionCoordinator(session_id="logical-2")
    first = CompressionRequest("logical-2", 4, "preflight", source_fingerprint="a")
    assert owner.admit_execution(first).outcome == "admitted"
    owner.finish_execution(first, "no_progress")
    assert owner.admit_execution(first).outcome == "no_progress_suppressed"
    changed = CompressionRequest("logical-2", 4, "overflow", urgency=3, source_fingerprint="b")
    assert owner.admit_execution(changed).outcome == "admitted"
    owner.finish_execution(changed, "timed_out")
    forced = CompressionRequest("logical-2", 4, "manual", urgency=3, source_fingerprint="b", force=True)
    assert owner.admit_execution(forced).outcome == "admitted"


def test_terminal_outcome_is_scoped_to_source_and_strategy():
    owner = CompressionCoordinator(session_id="logical-strategy")
    semantic = CompressionRequest(
        "logical-strategy", 7, "preflight", source_fingerprint="same",
        strategy="semantic",
    )
    forced = CompressionRequest(
        "logical-strategy", 7, "wire_recovery", source_fingerprint="same",
        strategy="forced_semantic",
    )

    assert owner.admit_execution(semantic).outcome == "admitted"
    owner.finish_execution(semantic, "timed_out")

    assert owner.terminal_outcome(semantic) == "timed_out"
    assert owner.admit_execution(semantic).outcome == "no_progress_suppressed"
    assert owner.terminal_outcome(forced) is None
    assert owner.admit_execution(forced).outcome == "admitted"


def test_transient_abort_does_not_poison_unchanged_source():
    owner = CompressionCoordinator(session_id="logical-abort")
    request = CompressionRequest(
        "logical-abort", 2, "preflight", source_fingerprint="same"
    )

    assert owner.admit_execution(request).outcome == "admitted"
    owner.finish_execution(request, "aborted")

    assert owner.terminal_outcome(request) is None
    assert owner.admit_execution(request).outcome == "admitted"


def test_background_policy_is_ratio_based_and_bounded():
    policy = resolve_background_compression_config(
        {"enabled": True, "start_ratio": 99, "deadline_seconds": 999}
    )

    assert isinstance(policy, BackgroundCompressionConfig)
    assert policy.enabled is True
    assert policy.start_ratio < 0.85
    assert policy.deadline_seconds == 120


def test_background_snapshot_stops_before_incomplete_tool_round():
    messages = [
        {"role": "user", **HUMAN, "content": "first"},
        *_round(1, body="done"),
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "pending",
                "type": "function",
                "function": {"name": "tool", "arguments": "{}"},
            }],
        },
    ]

    snapshot = build_background_snapshot(
        "logical", 3, messages, route={"provider": "p", "model": "m"}
    )

    assert snapshot is not None
    assert snapshot.source_length == 3


def test_background_worker_preserves_latest_human_verbatim_after_sampling(
    monkeypatch,
):
    latest = {"role": "user", **HUMAN, "content": "KEEP THIS EXACTLY"}
    messages = [{"role": "user", **HUMAN, "content": "first"}]
    messages.extend(
        {"role": "assistant", "content": f"bulk-{index}-" + "x" * 20_000}
        for index in range(8)
    )
    messages.extend([latest, {"role": "assistant", "content": "ack"}])
    snapshot = build_background_snapshot(
        "sampled",
        1,
        messages,
        route={
            "resolution": "auxiliary_auto",
            "certified_fast": True,
            "reasoning": False,
            "context_length": 12_000,
        },
    )
    assert snapshot is not None
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))]
        ),
    )

    candidate = run_background_compression_worker(snapshot)

    assert candidate.messages[-1] == latest


def test_background_job_coalesces_and_adopts_append_only_tail():
    messages = [{"role": "user", **HUMAN, "content": "current task"}]
    snapshot = build_background_snapshot(
        "logical-bg", 3, messages,
        route={"provider": "p", "model": "m"},
        schema_hash="tools-v1",
    )
    assert snapshot is not None
    release = threading.Event()
    candidate = CompressionCandidate(
        "logical-bg",
        3,
        snapshot.source_length,
        snapshot.prefix_fingerprint,
        "tools-v1",
        [
            {"role": "assistant", "content": "summary"},
            dict(messages[0]),
        ],
    )

    def worker(_snapshot):
        release.wait(1)
        return candidate

    owner = CompressionCoordinator(session_id="logical-bg")
    first = owner.start_background(snapshot, worker)
    second = owner.start_background(snapshot, worker)
    assert first is second
    assert owner.poll_background() is None

    release.set()
    assert first is not None
    first.result(timeout=1)
    messages[0]["_db_persisted"] = True
    live = messages + [{"role": "assistant", "content": "new answer"}]
    projected = owner.project_background(
        live, generation=3, schema_hash="tools-v1"
    )

    assert projected is not None
    assert [item["content"] for item in projected] == [
        "summary", "current task", "new answer"
    ]
    assert owner.project_background(
        live, generation=3, schema_hash="tools-v1"
    ) == projected


def test_background_candidate_rejects_changed_prefix_or_schema():
    messages = [{"role": "user", **HUMAN, "content": "original"}]
    snapshot = build_background_snapshot(
        "logical-stale", 1, messages,
        route={"provider": "p", "model": "m"},
        schema_hash="schema-a",
    )
    assert snapshot is not None
    candidate = CompressionCandidate(
        "logical-stale", 1, 1, snapshot.prefix_fingerprint, "schema-a",
        [{"role": "assistant", "content": "summary"}, dict(messages[0])],
    )
    owner = CompressionCoordinator(session_id="logical-stale")
    future = owner.start_background(snapshot, lambda _: candidate)
    assert future is not None
    future.result(timeout=1)

    assert owner.project_background(
        [{"role": "user", **HUMAN, "content": "edited"}],
        generation=1,
        schema_hash="schema-a",
    ) is None
    assert owner.project_background(
        messages, generation=1, schema_hash="schema-b"
    ) is None


def test_fit_request_starts_background_work_without_waiting(monkeypatch):
    from agent import compression_v3

    release = threading.Event()
    entered = threading.Event()

    def worker(snapshot):
        entered.set()
        release.wait(1)
        return CompressionCandidate(
            snapshot.session_id,
            snapshot.generation,
            snapshot.source_length,
            snapshot.prefix_fingerprint,
            snapshot.schema_hash,
            [{"role": "assistant", "content": "summary"}],
        )

    monkeypatch.setattr(compression_v3, "run_background_compression_worker", worker)
    agent = SimpleNamespace(
        session_id="nonblocking-fit",
        _config_context_length=1_000,
        _compression_safety_margin=0,
        _compression_generation=0,
        _compression_v3_background_config=BackgroundCompressionConfig(
            True, 0.05, 30
        ),
        _compression_v3_route={
            "provider": "aux",
            "model": "fast",
            "certified_fast": True,
            "reasoning": False,
        },
    )
    request = {
        "messages": [{"role": "user", **HUMAN, "content": "x" * 300}],
        "tools": [],
        "max_tokens": 8,
    }
    try:
        assert prepare_api_request(agent, request)["messages"] == request["messages"]
        assert entered.wait(1)
        assert agent._compression_coordinator.poll_background() is None
    finally:
        release.set()


def test_unfit_request_uses_deterministic_cut_while_background_is_pending(
    monkeypatch,
):
    from agent import compression_v3

    release = threading.Event()

    def worker(snapshot):
        release.wait(1)
        raise AssertionError("pending semantic result must not be awaited")

    monkeypatch.setattr(compression_v3, "run_background_compression_worker", worker)
    messages = [{"role": "user", **HUMAN, "content": "latest task"}]
    for index in range(8):
        messages.extend(_round(index, body="bulk " * 500))
    for row_id, message in enumerate(messages, start=1):
        message["_row_id"] = row_id
    agent = SimpleNamespace(
        session_id="nonblocking-unfit",
        _config_context_length=2_200,
        _compression_safety_margin=0,
        _compression_generation=0,
        _session_db=SimpleNamespace(
            register_compression_recovery=lambda *_args, **_kwargs: "recovery-1"
        ),
        _compression_v3_background_config=BackgroundCompressionConfig(
            True, 0.05, 30
        ),
        _compression_v3_route={
            "provider": "aux",
            "model": "fast",
            "certified_fast": True,
            "reasoning": False,
        },
    )
    try:
        prepared = prepare_api_request(
            agent, {"messages": messages, "tools": [], "max_tokens": 8}
        )
        assert len(prepared["messages"]) < len(messages)
        assert agent._compression_coordinator.poll_background() is None
    finally:
        release.set()


def test_budget_includes_wire_floor_and_blocks_unfit_projection():
    budget = CompressionBudget(context_window=100, output_reserve=20, safety_margin=10, system_tokens=25, tool_schema_tokens=15)
    assert budget.safe_input_budget == 70
    assert budget.history_budget == 30
    assert budget.fits([{"role": "user", "content": "x" * 200}]) is False


def test_emergency_cut_recomputes_final_wire_fit_while_demoting():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    for i in range(7):
        messages.extend(_round(i, body="bulk-" + ("x" * 800)))
    budget = CompressionBudget(6000, 10, 10)
    wire_sizes = []
    def wire_fit(candidate):
        size = len(json.dumps(candidate, ensure_ascii=True))
        wire_sizes.append(size)
        return size <= 8000
    cut = emergency_context_cut(
        messages,
        budget,
        session_id="wire-fit",
        generation=1,
        wire_fit=wire_fit,
    )
    assert cut.provider_call_allowed is True
    assert len(json.dumps(cut.messages, ensure_ascii=True)) <= 8000
    assert wire_sizes
    assert wire_sizes[-1] <= 8000


def test_emergency_cut_refuses_when_final_wire_floor_is_irreducible():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    cut = emergency_context_cut(
        messages,
        CompressionBudget(6000, 10, 10),
        session_id="wire-floor",
        generation=1,
        wire_fit=lambda candidate: False,
    )
    assert cut.provider_call_allowed is False
    assert cut.outcome == "context_projection_unfit"

def test_emergency_cut_preserves_current_group_and_latest_six_rounds():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    for i in range(8):
        messages.extend(_round(i, body=("old" if i == 0 else "recent") * 100))
    cut = emergency_context_cut(messages, CompressionBudget(1500, 10, 10, 10, 10), session_id="s", generation=3)
    assert cut.outcome == "emergency_context_cut"
    assert "latest task" in [m.get("content") for m in cut.messages]
    assert sum(m.get("role") == "tool" for m in cut.messages) >= 6
    assert all(m.get("role") != "tool" or m.get("tool_call_id") for m in cut.messages)
    assert validate_projection(cut.messages).valid


def test_emergency_cut_demotes_only_oldest_bodies_needed_to_fit():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    bodies = [f"body-{i}-" + ("x" * 1190) for i in range(6)]
    for i, body in enumerate(bodies):
        messages.extend(_round(i, body=body))

    cut = emergency_context_cut(messages, CompressionBudget(1800, 10, 10, 10, 10), session_id="s", generation=3)

    assert cut.outcome == "emergency_context_cut"
    assert cut.provider_call_allowed is True
    assert CompressionBudget(1800, 10, 10, 10, 10).fits(cut.messages)
    tool_bodies = [m["content"] for m in cut.messages if m.get("role") == "tool"]
    assert tool_bodies[0].startswith("[COMPACTION RECOVERY PENDING]")
    assert tool_bodies[-1] == bodies[-1]
    assert all(m.get("tool_call_id") for m in cut.messages if m.get("role") == "tool")
    assert validate_projection(cut.messages).valid


def test_emergency_cut_does_not_demote_already_fitting_candidate():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    body = "verbatim body"
    messages.extend(_round(0, body=body))

    cut = emergency_context_cut(messages, CompressionBudget(1000, 10, 10, 10, 10), session_id="s", generation=3)

    assert cut.provider_call_allowed is True
    assert [m["content"] for m in cut.messages if m.get("role") == "tool"] == [body]


def test_emergency_cut_can_demote_exactly_one_oldest_body():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    bodies = [f"body-{i}-" + ("x" * 1190) for i in range(6)]
    for i, body in enumerate(bodies):
        messages.extend(_round(i, body=body))

    cut = emergency_context_cut(messages, CompressionBudget(1950, 10, 10, 10, 10), session_id="s", generation=3)

    tool_bodies = [m["content"] for m in cut.messages if m.get("role") == "tool"]
    assert sum(body.startswith("[COMPACTION RECOVERY PENDING]") for body in tool_bodies) == 1
    assert tool_bodies[1:] == bodies[1:]


def test_emergency_cut_demotes_all_retained_bodies_when_required():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    for i in range(6):
        messages.extend(_round(i, body=f"body-{i}-" + ("x" * 1190)))

    cut = emergency_context_cut(messages, CompressionBudget(1000, 10, 10, 10, 10), session_id="s", generation=3)

    tool_bodies = [m["content"] for m in cut.messages if m.get("role") == "tool"]
    assert cut.provider_call_allowed is True
    assert all(body.startswith("[COMPACTION RECOVERY PENDING]") for body in tool_bodies)
    assert CompressionBudget(1000, 10, 10, 10, 10).fits(cut.messages)


def test_emergency_cut_honors_minimum_reclaim_after_candidate_fits():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    bodies = [f"body-{i}-" + ("x" * 10_000) for i in range(6)]
    for i, body in enumerate(bodies):
        messages.extend(_round(i, body=body))

    original_tokens = estimate_projection_tokens(messages)
    cut = emergency_context_cut(
        messages,
        CompressionBudget(100_000, 10, 10),
        session_id="s",
        generation=3,
        min_reclaim_tokens=8_192,
    )

    reclaimed = original_tokens - estimate_projection_tokens(cut.messages)
    assert cut.provider_call_allowed is True
    assert reclaimed >= 8_192
    tool_bodies = [m["content"] for m in cut.messages if m.get("role") == "tool"]
    assert all(body.startswith("[COMPACTION RECOVERY PENDING]") for body in tool_bodies[:4])
    assert tool_bodies[4:] == bodies[4:]
    assert validate_projection(cut.messages).valid


def test_emergency_cut_reports_fit_but_insufficient_reclaim_when_minimum_is_impossible():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "latest task"}]
    for i in range(6):
        messages.extend(_round(i, body=f"small-{i}"))

    cut = emergency_context_cut(
        messages,
        CompressionBudget(100_000, 10, 10),
        session_id="s",
        generation=3,
        min_reclaim_tokens=8_192,
    )

    assert cut.provider_call_allowed is False
    assert cut.outcome == "context_projection_min_reclaim_unmet"
    assert estimate_projection_tokens(messages) - estimate_projection_tokens(cut.messages) < 8_192
    assert validate_projection(cut.messages).valid


def test_capsule_uses_human_intent_not_synthetic_wakes_and_keeps_identifiers():
    messages = [
        {"role": "user", "content": "synthetic wake", "_internal_wake": True},
        {"role": "user", **HUMAN, "content": "ship task at /tmp/a.py on abc1234; PR #42"},
        {"role": "assistant", "content": "MUST preserve approvals; NEVER reset"},
    ]
    capsule = build_policy_capsule(messages, session_id="s", watermark=9, generation=2)
    assert capsule.latest_human_intent == "ship task at /tmp/a.py on abc1234; PR #42"
    assert "abc1234" in capsule.identifiers and "#42" in capsule.identifiers
    assert "synthetic wake" not in capsule.latest_human_intent


def test_coordinator_coalesces_generation_and_rejects_stale_candidate():
    coordinator = CompressionCoordinator(session_id="s")
    first = coordinator.request(CompressionRequest("s", 4, "tool_pressure", 1))
    second = coordinator.request(CompressionRequest("s", 4, "hard_limit", 3))
    assert first.attempt_id == second.attempt_id
    assert second.urgency == 3
    candidate = CompressionCandidate("s", 4, 0, "wrong", "", [{"role": "user", "content": "stale"}])
    assert coordinator.adopt(candidate) is False
    assert coordinator.outcome is None


def test_snapshot_adoption_splices_post_watermark_tail_once():
    coordinator = CompressionCoordinator(session_id="s")
    coordinator.request(CompressionRequest("s", 2, "background", 1))
    snapshot = coordinator.snapshot(watermark=7)
    assert snapshot.session_id == "s" and snapshot.watermark == 7
    candidate = CompressionCandidate("s", 2, 7, "", "", [{"role": "user", "content": "old"}, {"role": "assistant", "content": "summary"}])
    tail = [{"role": "user", "content": "new"}, {"role": "assistant", "content": "answer"}]
    assert coordinator.adopt_with_tail(candidate, tail, current_watermark=8) is True
    assert [m["content"] for m in coordinator.active_projection] == ["old", "summary", "new", "answer"]


def test_background_route_requires_explicit_certification():
    assert compression_route_is_eligible({"provider": "p", "model": "m", "certified_fast": True, "reasoning": False})
    assert compression_route_is_eligible({"resolution": "auxiliary_auto", "certified_fast": True, "reasoning": False})
    assert not compression_route_is_eligible({"provider": "p", "model": "m", "reasoning": False})
    assert not compression_route_is_eligible({"provider": "p", "model": "m", "certified_fast": True, "reasoning": True})


def test_agent_coordinator_is_recreated_only_when_session_identity_changes():
    agent = type("Agent", (), {"session_id": "s", "_compression_generation": 4})()
    first = ensure_compression_coordinator(agent, trigger="automatic")
    second = ensure_compression_coordinator(agent, trigger="hard_limit", urgency=3)
    assert first is second and second.attempt.urgency == 3
    agent.session_id = "s-next"
    assert ensure_compression_coordinator(agent, trigger="automatic") is not first


def test_irreducible_floor_returns_typed_unfit_without_provider_call():
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "task"}]
    result = emergency_context_cut(messages, CompressionBudget(20, 5, 5, 15, 15), session_id="s", generation=1)
    assert result.outcome == "context_projection_unfit"
    assert result.provider_call_allowed is False
    assert result.messages == messages


def test_recovery_binding_uses_exact_sidecar_ids_not_history_content():
    registered = {}

    class DB:
        def register_compression_recovery(self, session_id, ids, **kwargs):
            registered["ids"] = ids
            return "canonical"

    source = [
        {"role": "assistant", "content": "duplicate", "reasoning": "current", "_row_id": 3},
        {"role": "tool", "content": "same", "tool_call_id": "c", "_row_id": 4},
    ]
    retained = [dict(source[1])]
    retained[0]["content"] = "[COMPACTION RECOVERY PENDING] session=s anchor=opaque"
    agent = SimpleNamespace(_session_db=DB(), session_id="s")

    assert _bind_recovery_identity(agent, source, retained, "opaque", 1, 3) == "canonical"
    assert registered["ids"] == [3]


def test_responses_wire_floor_is_rejected_before_provider():
    agent = type("Agent", (), {"session_id": "s", "_config_context_length": 100, "_compression_safety_margin": 10})()
    request = {
        "instructions": "i" * 1000,
        "input": [{"role": "user", "content": "u" * 1000}],
        "max_output_tokens": 10,
    }
    with pytest.raises(ContextProjectionUnfit):
        prepare_api_request(agent, request)


def test_responses_emergency_projection_keeps_current_task_and_fits_wire():
    agent = SimpleNamespace(
        session_id="s",
        _config_context_length=1_200,
        _compression_safety_margin=0,
        _provider_wire_emergency_projection=True,
    )
    current_task = "finish the current investigation"
    request = {
        "model": "gpt-5.6-sol",
        "instructions": "policy",
        "input": [
            {"role": "user", "content": "old context " * 4_000},
            {"role": "assistant", "content": "old answer " * 4_000},
            {"role": "user", "content": current_task},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "web_search",
                "arguments": "{\"q\":\"context windows\"}",
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "large result " * 4_000,
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "context_management": [{"type": "compaction", "compact_threshold": 900}],
        "max_output_tokens": 4_000,
        "store": False,
    }

    prepared = prepare_api_request(agent, request)

    assert provider_request_budget(agent, prepared).fits is True
    assert any(
        item.get("role") == "user" and item.get("content") == current_task
        for item in prepared["input"]
    )
    call_ids = {
        item.get("call_id")
        for item in prepared["input"]
        if item.get("type") == "function_call"
    }
    output_ids = {
        item.get("call_id")
        for item in prepared["input"]
        if item.get("type") == "function_call_output"
    }
    assert call_ids == output_ids
    assert prepared["tools"] == request["tools"]
    assert prepared["tool_choice"] == request["tool_choice"]
    assert prepared["parallel_tool_calls"] is request["parallel_tool_calls"]
    assert prepared["context_management"] == request["context_management"]
    assert prepared["max_output_tokens"] < request["max_output_tokens"]


def test_responses_emergency_projection_preserves_transport_output_shape():
    """A late fit recovery must not invent controls omitted by transport."""
    agent = SimpleNamespace(
        session_id="s",
        _config_context_length=1_200,
        _compression_safety_margin=0,
        _provider_wire_emergency_projection=True,
    )
    current_task = "continue the current task without changing the request schema"
    request = {
        "model": "gpt-5.6-sol",
        "instructions": "policy",
        "input": [
            {"role": "user", "content": "old context " * 4_000},
            {"role": "assistant", "content": "old answer " * 4_000},
            {"role": "user", "content": current_task},
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "context_management": [{"type": "compaction", "compact_threshold": 900}],
        # Codex backend intentionally omits max_output_tokens. The context
        # layer receives the already-built wire shape and must preserve that
        # transport decision while reducing optional replay history.
        "store": False,
    }

    prepared = prepare_api_request(agent, request)

    assert provider_request_budget(agent, prepared).fits is True
    assert "max_output_tokens" not in prepared
    assert any(
        item.get("role") == "user" and item.get("content") == current_task
        for item in prepared["input"]
    )
    assert prepared["tools"] == request["tools"]
    assert prepared["tool_choice"] == request["tool_choice"]
    assert prepared["parallel_tool_calls"] is request["parallel_tool_calls"]
    assert prepared["context_management"] == request["context_management"]


def test_responses_emergency_projection_refuses_to_drop_oversized_toolset():
    agent = SimpleNamespace(
        session_id="s",
        _config_context_length=300,
        _compression_safety_margin=0,
        _provider_wire_emergency_projection=True,
    )
    request = {
        "model": "gpt-5.6-sol",
        "instructions": "policy",
        "input": [{"role": "user", "content": "continue"}],
        "tools": [
            {
                "type": "function",
                "name": "required_tool",
                "description": "schema " * 2_000,
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": "auto",
        "max_output_tokens": 128,
    }

    with pytest.raises(ContextProjectionUnfit):
        prepare_api_request(agent, request)

    assert request["tools"][0]["name"] == "required_tool"


def test_responses_emergency_projection_reduces_output_to_positive_minimum():
    agent = SimpleNamespace(
        session_id="s",
        _config_context_length=1_000,
        _compression_safety_margin=0,
        _provider_wire_emergency_projection=True,
    )
    request = {
        "model": "gpt-5.6-sol",
        "instructions": "policy",
        "input": [{"role": "user", "content": "continue"}],
        "tools": [
            {
                "type": "function",
                "name": "required_tool",
                "description": "x" * 2_720,
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": "auto",
        "max_output_tokens": 128,
    }

    assert provider_request_budget(agent, request).fits is False

    prepared = prepare_api_request(agent, request)

    assert provider_request_budget(agent, prepared).fits is True
    assert prepared["max_output_tokens"] == 1
    assert prepared["tools"] == request["tools"]
    assert prepared["input"] == request["input"]


def test_responses_emergency_projection_never_rewrites_current_user_task():
    agent = SimpleNamespace(
        session_id="s",
        _config_context_length=1_000,
        _compression_safety_margin=0,
        _provider_wire_emergency_projection=True,
    )
    current_task = "start:" + (" preserve every detail" * 700) + ":end"
    request = {
        "model": "gpt-5.6-sol",
        "instructions": "policy",
        "input": [{"role": "user", "content": current_task}],
        "max_output_tokens": 128,
    }

    with pytest.raises(ContextProjectionUnfit):
        prepare_api_request(agent, request)

    assert request["input"] == [{"role": "user", "content": current_task}]


def test_pre_send_gate_without_persistence_returns_typed_unfit():
    agent = type("Agent", (), {"session_id": "s", "_compression_generation": 2, "_config_context_length": 300, "_compression_safety_margin": 10})()
    call_id = "call-1"
    request = {"messages": [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "task"}, {"role": "assistant", "tool_calls": [{"id": call_id}]}, {"role": "tool", "tool_call_id": call_id, "content": "x" * 3000}], "max_tokens": 10}
    with pytest.raises(ContextProjectionUnfit):
        prepare_api_request(agent, request)
    assert request["messages"][-1]["content"] == "x" * 3000


def test_pre_send_gate_refuses_irreducible_request_without_mutating_input():
    agent = type("Agent", (), {"session_id": "s", "_compression_generation": 1, "_config_context_length": 20, "_compression_safety_margin": 5})()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "task"}]
    request = {"messages": messages, "max_tokens": 5}
    try:
        prepare_api_request(agent, request)
    except Exception as exc:
        assert getattr(exc, "outcome", None) == "context_projection_unfit"
    else:
        raise AssertionError("irreducible request was accepted")
    assert request["messages"] == messages


def test_tool_pressure_projection_keeps_six_parallel_semantic_rounds():
    agent = type(
        "Agent",
        (),
        {
            "session_id": "s",
            "_compression_generation": 4,
            "_config_context_length": 20_000,
            "max_tokens": 100,
            "_compression_safety_margin": 100,
        },
    )()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "do the work"}]
    for number in range(8):
        messages.append({"role": "assistant", "tool_calls": [
            {"id": f"r{number}-a", "type": "function", "function": {"name": "tool", "arguments": "{}"}},
            {"id": f"r{number}-b", "type": "function", "function": {"name": "tool", "arguments": "{}"}},
        ]})
        messages.extend([
            {"role": "tool", "tool_call_id": f"r{number}-b", "content": f"r{number}-b-" + "x" * 10000},
            {"role": "tool", "tool_call_id": f"r{number}-a", "content": f"r{number}-a-" + "x" * 5000},
        ])
    projection, reclaimed = prune_tool_pressure_projection(agent, messages)
    assert reclaimed == 0
    assert projection is messages


def test_tool_pressure_projection_prunes_completed_rounds_before_next_request():
    agent = type(
        "Agent",
        (),
        {
            "session_id": "s",
            "_compression_generation": 4,
            "_config_context_length": 20_000,
            "max_tokens": 100,
            "_compression_safety_margin": 100,
        },
    )()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "do the work"}]
    for number in range(12):
        messages.extend(_round(number, body=f"result-{number}-" + "x" * 10_000))
    projection, reclaimed = prune_tool_pressure_projection(agent, messages)
    assert reclaimed == 0
    assert projection is messages


def test_tool_pressure_projection_accounts_for_external_request_floor():
    from agent.compression_v3 import estimate_projection_tokens

    agent = type(
        "Agent",
        (),
        {
            "session_id": "s",
            "_compression_generation": 4,
            "_config_context_length": 20_000,
            "max_tokens": 100,
            "_compression_safety_margin": 100,
        },
    )()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "do the work"}]
    for number in range(12):
        messages.extend(_round(number, body=f"result-{number}-" + "x" * 5_500))

    # Message projection is below the whole-request soft threshold, but tool
    # schemas/wire overhead put the observed request above it.
    before = estimate_projection_tokens(messages)
    assert before < 20_000 * 0.85
    projection, reclaimed = prune_tool_pressure_projection(agent, messages, current_tokens=18_000)

    assert projection is messages
    assert reclaimed == 0


def test_tool_pressure_projection_does_not_split_incomplete_group():
    agent = type("Agent", (), {"session_id": "s", "_config_context_length": 20_000})()
    messages = [{"role": "user", **HUMAN, "content": "task"}, {"role": "assistant", "tool_calls": [{"id": "pending"}]}]
    projection, reclaimed = prune_tool_pressure_projection(agent, messages, current_tokens=19_000)
    assert reclaimed == 0
    assert projection == messages


def test_tool_pressure_projection_keeps_cache_stable_below_soft_threshold():
    agent = type("Agent", (), {"session_id": "s", "_config_context_length": 20_000})()
    messages = [{"role": "user", **HUMAN, "content": "small completed turn"}]
    projection, reclaimed = prune_tool_pressure_projection(agent, messages, current_tokens=1_000)
    assert reclaimed == 0
    assert projection is messages


def test_production_turn_executes_tools_prunes_projection_and_preserves_sessiondb(
    monkeypatch, tmp_path
):
    """The mid-turn boundary must be exercised through AIAgent, not its helper."""
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.registry import registry
    import agent.compression_v3 as compression_v3

    tool_name = "compression_v3_e2e_tool"
    original = registry.get_entry(tool_name)
    registry.register(
        name=tool_name,
        toolset="test",
        schema={
            "name": tool_name,
            "description": "deterministic compression test tool",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kwargs: json.dumps(
            {"ok": True, "body": "canonical-" + "x" * 3_000}
        ),
        check_fn=lambda: True,
        max_result_size_chars=100_000,
    )
    db = SessionDB(db_path=Path(tmp_path) / "session.db")
    session_id = "compression-v3-production-e2e"
    agent = None
    calls = []
    try:
        prior = [{"role": "system", "content": "policy"}]
        for number in range(7):
            prior.extend(_round(number, body=f"durable-{number}-" + "y" * 6_000))

        db.create_session(session_id=session_id, source="cli")
        db.append_messages_batch(session_id, prior)

        monkeypatch.setattr(
            "run_agent.get_tool_definitions",
            lambda **kwargs: [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "deterministic compression test tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        agent = AIAgent(
            api_key="test-key",
            base_url="http://compression-v3.test",
            provider="openrouter",
            model="test-model",
            session_id=session_id,
            session_db=db,
            max_iterations=3,
            max_tokens=100,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.compression_enabled = True
        agent._config_context_length = 100_000
        agent.context_compressor.context_length = 100_000
        agent.context_compressor.should_compress = lambda tokens: False
        agent._compress_context = lambda messages, system_message, **kwargs: (
            messages,
            system_message,
        )
        agent._compression_safety_margin = 100
        agent._disable_streaming = True
        agent._cached_system_prompt = "policy"
        production_prune_calls = []
        production_prune_results = []
        production_pressure = {}
        original_prune = compression_v3.prune_tool_pressure_projection

        def production_prune(*args, **kwargs):
            args[0]._config_context_length = 60_000
            args[0].context_compressor.context_length = 60_000
            message_before = estimate_projection_tokens(args[1])
            soft_budget = int(args[0]._config_context_length * 0.85)
            non_message_floor = min(
                soft_budget - 1,
                max(1, soft_budget - message_before + 4_096),
            )
            kwargs["current_tokens"] = message_before + non_message_floor
            production_pressure.update(
                message_before=message_before,
                non_message_floor=non_message_floor,
                soft_budget=soft_budget,
            )
            production_prune_calls.append(kwargs["current_tokens"])
            result = original_prune(*args, **kwargs)
            production_prune_results.append(result)
            return result

        monkeypatch.setattr(compression_v3, "prune_tool_pressure_projection", production_prune)

        tool_call = SimpleNamespace(
            id="live-call",
            type="function",
            function=SimpleNamespace(name=tool_name, arguments="{}"),
        )
        scripted = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant", content=None, tool_calls=[tool_call]
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            role="assistant", content="final answer", tool_calls=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            ),
        ]

        def scripted_call(kwargs):
            calls.append(json.loads(json.dumps(kwargs["messages"])))
            if len(calls) == 1:
                agent._config_context_length = 60_000
                agent.context_compressor.context_length = 60_000
            return scripted.pop(0)

        agent._interruptible_api_call = scripted_call
        result = agent.run_conversation(
            "task", system_message="policy", conversation_history=prior, task_id="compression-e2e"
        )

        assert result["final_response"] == "final answer"
        assert production_prune_calls
        assert production_prune_results
        assert len(calls) == 2
        message_before = production_pressure["message_before"]
        non_message_floor = production_pressure["non_message_floor"]
        soft_budget = production_pressure["soft_budget"]
        projected, reclaimed = production_prune_results[-1]
        message_after = estimate_projection_tokens(projected)
        assert reclaimed >= 8_192
        assert projected != prior
        assert 0 < non_message_floor < soft_budget
        assert message_before + non_message_floor > soft_budget
        assert message_after + non_message_floor <= soft_budget
        second_tools = [m for m in calls[1] if m.get("role") == "tool"]
        first_tools = [m for m in calls[0] if m.get("role") == "tool"]
        assert len(first_tools) >= 7
        assert estimate_projection_tokens(calls[1]) < message_before
        assert any(message.get("tool_call_id") == "live-call" for message in second_tools)
        assert any(
            message.get("role") == "system"
            and "[COMPACTION RECOVERY]" in message.get("content", "")
            for message in calls[1]
        )
        assert {
            message["tool_call_id"]
            for message in calls[1]
            if message.get("role") == "tool"
        } == {f"call-{number}" for number in range(2, 7)} | {"live-call"}
        assert any(
            message.get("content") == "durable-0-" + "y" * 6_000
            for message in calls[0]
        )
        assert not any(
            message.get("content") == "durable-0-" + "y" * 6_000
            for message in calls[1]
        )
        assert validate_projection(calls[1]).valid


        loaded = db.get_messages(session_id, include_inactive=True)
        durable_bodies = [
            message["content"]
            for message in loaded
            if message.get("role") == "tool"
        ]
        canonical_body = json.dumps({"ok": True, "body": "canonical-" + "x" * 3_000})
        assert durable_bodies.count(canonical_body) == 1
        assert all(
            durable_bodies.count(f"durable-{number}-" + "y" * 6_000) == 1
            for number in range(7)
        )
    finally:
        if original is None:
            registry.deregister(tool_name)
        if agent is not None:
            agent.close()
        db.close()
