"""Behavioral contracts for the continuable-session compression coordinator."""

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
    assert all(
        any(str(message.get("content", "")).startswith(f"result-{number}-") for message in projection)
        for number in range(6, 12)
    )


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
