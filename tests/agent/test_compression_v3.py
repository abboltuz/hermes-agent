"""Behavioral contracts for the continuable-session compression coordinator."""

import json
from pathlib import Path
from types import SimpleNamespace

from agent.compression_v3 import (
    CompressionBudget,
    CompressionCandidate,
    CompressionCoordinator,
    CompressionRequest,
    build_policy_capsule,
    compression_route_is_eligible,
    ensure_compression_coordinator,
    emergency_context_cut,
    validate_projection,
    prepare_api_request,
    prune_tool_pressure_projection,
    estimate_projection_tokens,
)


HUMAN = {"origin_kind": "human_user", "turn_kind": "prompt", "trust_kind": "user_authorized"}


def _round(number: int, *, body: str = "result"):
    call_id = f"call-{number}"
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": body},
    ]


def test_budget_includes_wire_floor_and_blocks_unfit_projection():
    budget = CompressionBudget(context_window=100, output_reserve=20, safety_margin=10, system_tokens=25, tool_schema_tokens=15)
    assert budget.safe_input_budget == 70
    assert budget.history_budget == 30
    assert budget.fits([{"role": "user", "content": "x" * 200}]) is False


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


def test_pre_send_gate_cuts_reducible_request_and_binds_session_coordinator():
    agent = type("Agent", (), {"session_id": "s", "_compression_generation": 2, "_config_context_length": 300, "_compression_safety_margin": 10})()
    call_id = "call-1"
    request = {"messages": [{"role": "system", "content": "policy"}, {"role": "user", **HUMAN, "content": "task"}, {"role": "assistant", "tool_calls": [{"id": call_id}]}, {"role": "tool", "tool_call_id": call_id, "content": "x" * 3000}], "max_tokens": 10}
    prepared = prepare_api_request(agent, request)
    assert prepared["messages"] != request["messages"]
    assert agent._compression_coordinator.outcome is None
    assert validate_projection(prepared["messages"]).valid


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
    assert reclaimed >= 8192
    assert {m["tool_call_id"] for m in projection if m.get("role") == "tool"} == {
        f"r{number}-{suffix}" for number in range(2, 8) for suffix in ("a", "b")
    }


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
    assert reclaimed >= 8192
    assert projection != messages
    assert validate_projection(projection).valid
    assert {
        message["tool_call_id"]
        for message in projection
        if message.get("role") == "tool"
    } == {f"call-{number}" for number in range(6, 12)}


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

    assert projection != messages
    assert reclaimed >= 8192
    assert estimate_projection_tokens(projection) + (18_000 - before) <= 20_000 * 0.85


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
