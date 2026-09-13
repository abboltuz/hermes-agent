"""Worker route and typed-failure contracts; no live providers or profiles."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import goals, kanban_db as kb
from hermes_cli.kanban_model_route import (
    REQUIRED_MODEL, WORKER_ROUTE_ENV, card_model_route, judge_route_for_task,
)


def task(**overrides):
    fields = dict(id="task-1", title="Implement", body="Verify", goal_mode=True,
                  goal_max_turns=3, model_override="test-model", provider_override="antigravity")
    fields.update(overrides)
    return SimpleNamespace(**fields)


def loop(**overrides):
    kwargs = dict(task_id="task-1", goal_text="Implement", first_response="progress",
                  task_status_fn=lambda: "running", run_turn=Mock(), block_fn=Mock())
    kwargs.update(overrides)
    return goals.run_kanban_goal_loop(**kwargs)


@pytest.mark.parametrize("reason,exit_code", [("provider_error", 1), ("rate_limit", 75), ("billing", 75)])
def test_initial_provider_failure_does_not_enter_goal_loop(monkeypatch, reason, exit_code):
    import cli

    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    failure = {"failed": True, "failure_reason": reason, "final_response": "Provider error"}
    assert not cli._should_run_kanban_goal_loop(failure)
    assert cli._single_query_exit_code(failure) == exit_code
    judge = Mock()
    monkeypatch.setattr(goals, "judge_goal", judge)
    result = loop(first_result=failure)
    assert result["failure_result"] is failure
    judge.assert_not_called()


def test_continuation_provider_failure_retains_result_and_stops(monkeypatch):
    judge = Mock(return_value=("continue", "more work", False, None, False))
    monkeypatch.setattr(goals, "judge_goal", judge)
    failure = {"failed": True, "failure_reason": "billing", "billing_block": {"code": "credits"}}
    run = Mock(return_value=failure)
    block = Mock()
    route = {"provider": "antigravity", "model": "test-model"}
    result = loop(run_turn=run, block_fn=block, judge_route=route)
    assert result["outcome"] == "failed_provider"
    assert result["failure_result"] is failure
    assert result["turns_used"] == 2
    judge.assert_called_once_with("Implement", "progress", route=route)
    run.assert_called_once()
    block.assert_not_called()


@pytest.mark.parametrize("parse_failed,transport_failed", [(False, True), (True, False)])
def test_unavailable_judge_stops_without_more_worker_turns(monkeypatch, parse_failed, transport_failed):
    judge = Mock(return_value=("continue", "unavailable", parse_failed, None, transport_failed))
    monkeypatch.setattr(goals, "judge_goal", judge)
    run, block = Mock(), Mock()
    result = loop(run_turn=run, block_fn=block)
    assert result["outcome"] == "blocked_judge_unavailable"
    run.assert_not_called()
    block.assert_called_once()
    judge.assert_called_once()


@pytest.mark.parametrize("status", ["done", "review", "changes_requested", "blocked"])
def test_committed_terminal_status_takes_precedence(monkeypatch, status):
    judge = Mock()
    monkeypatch.setattr(goals, "judge_goal", judge)
    result = loop(task_status_fn=lambda: status, first_result={"failed": True})
    assert "failure_result" not in result
    judge.assert_not_called()


def test_quiet_wrapper_returns_typed_failure_and_pinned_route(monkeypatch):
    import cli

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.delenv(WORKER_ROUTE_ENV, raising=False)
    monkeypatch.setattr(kb, "connect", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(kb, "get_task", lambda *_: task())
    monkeypatch.setattr(kb, "goal_run_status", lambda *_: "running")
    failure = {"failed": True, "failure_reason": "rate_limit", "final_response": "limited"}
    agent = SimpleNamespace(session_id="session-1", run_conversation=Mock(return_value=failure))
    shell = SimpleNamespace(agent=agent, session_id="session-1", conversation_history=[])
    judge = Mock(return_value=("continue", "more work", False, None, False))
    monkeypatch.setattr(goals, "judge_goal", judge)
    assert cli._run_kanban_goal_loop_q(shell, "progress") is failure
    assert cli._single_query_exit_code(failure) == 75
    assert judge.call_args.kwargs["route"] == {"provider": "antigravity", "model": "test-model"}
    assert agent.run_conversation.call_args.kwargs["persist_user_provenance"]["turn_kind"] == "continuation"


def test_snapshot_wins_only_for_own_task(monkeypatch):
    pinned = {"provider": "antigravity", "model": "original-model"}
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv(WORKER_ROUTE_ENV, json.dumps(pinned))
    assert judge_route_for_task(task()) == pinned
    assert judge_route_for_task(task(id="task-2"))["model"] == "test-model"


@pytest.mark.parametrize("raw", ['{', '{}', '[]', '{"provider":"auto","model":"x"}',
                                  '{"provider":"antigravity","model":"auto"}'])
def test_invalid_worker_snapshot_is_rejected(monkeypatch, raw):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv(WORKER_ROUTE_ENV, raw)
    with pytest.raises(ValueError):
        judge_route_for_task(task())


@pytest.mark.parametrize("model,provider", [(None, None), ("test-model", None),
                                            (None, "antigravity"), (REQUIRED_MODEL, "antigravity")])
def test_required_profile_needs_complete_card_route(tmp_path, model, provider):
    (tmp_path / "config.yaml").write_text(f"model:\n  default: {REQUIRED_MODEL}\n")
    with pytest.raises(ValueError):
        card_model_route(task(model_override=model, provider_override=provider), tmp_path)


def test_ordinary_profile_inheritance_remains_compatible(tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  default: default-model\n")
    assert card_model_route(task(model_override=None, provider_override=None), tmp_path) is None
    assert card_model_route(task(provider_override=None), tmp_path) is None
    assert card_model_route(task(), tmp_path) == {"provider": "antigravity", "model": "test-model"}


@pytest.mark.parametrize(
    "contents",
    [
        "model: [unterminated\n",
        "- model\n- is-not-a-mapping\n",
    ],
)
def test_corrupt_profile_blocks_worker_route_resolution(tmp_path, contents):
    (tmp_path / "config.yaml").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to load configuration"):
        card_model_route(
            task(model_override=None, provider_override=None), tmp_path
        )


def test_corrupt_profile_cannot_reuse_cached_route_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"model:\n  default: {REQUIRED_MODEL}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="explicit provider and model"):
        card_model_route(
            task(model_override=None, provider_override=None), tmp_path
        )

    config_path.write_text("model: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load configuration"):
        card_model_route(
            task(model_override=None, provider_override=None), tmp_path
        )


def test_profile_route_uses_effective_managed_config(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text(
        "model:\n  default: ${PROFILE_MODEL}\n", encoding="utf-8"
    )
    managed_home = tmp_path / "managed"
    managed_home.mkdir()
    (managed_home / "config.yaml").write_text(
        "model:\n  default: managed-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("PROFILE_MODEL", REQUIRED_MODEL)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    managed_scope.invalidate_managed_cache()

    assert card_model_route(
        task(model_override=None, provider_override=None), profile_home
    ) is None
    managed_scope.invalidate_managed_cache()


@pytest.mark.parametrize("provider", ["custom:auto", "custom:main", "custom:moa",
                                    "actual-computer", "actualcomputer", "aci", "custom:aci",
                                    "custom", "custom:", "custom:custom", "ollama"])
def test_dynamic_alias_cannot_be_a_card_pin_or_snapshot(monkeypatch, provider):
    with pytest.raises(ValueError):
        card_model_route(task(provider_override=provider))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv(WORKER_ROUTE_ENV, json.dumps({"provider": provider, "model": "test-model"}))
    with pytest.raises(ValueError):
        judge_route_for_task(task())


@pytest.mark.parametrize("provider", ["myrelay", "custom:myrelay"])
def test_named_custom_card_route_is_preserved(provider):
    assert card_model_route(task(provider_override=provider))["provider"] == provider


@pytest.mark.parametrize("surface", ["cli", "tool"])
def test_handoff_judge_uses_pinned_route_not_auxiliary_default(monkeypatch, surface):
    from agent import auxiliary_client
    from hermes_cli import kanban
    from tools import kanban_tools

    monkeypatch.delenv(WORKER_ROUTE_ENV, raising=False)
    probe = Mock(side_effect=AssertionError("must not probe default judge"))
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", probe)
    monkeypatch.setattr(kanban_tools, "_goal_judge_available", probe)
    judge = Mock(return_value=("blocked", "impossible", False, None, False))
    monkeypatch.setattr(goals, "judge_goal", judge)
    monkeypatch.setattr(kanban_tools, "judge_goal", judge)
    module = kanban if surface == "cli" else kanban_tools
    assert module._goal_mode_handoff_rejection(task(), "evidence") == ("blocked", "impossible")
    assert judge.call_args.kwargs["route"] == {"provider": "antigravity", "model": "test-model"}
    probe.assert_not_called()


def test_judge_explicit_route_disables_fallback_without_losing_cap(monkeypatch):
    from agent import auxiliary_client

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"verdict":"done","reason":"verified"}'))])
    call = Mock(return_value=response)
    monkeypatch.setattr(auxiliary_client, "call_llm", call)
    monkeypatch.setattr(goals, "_goal_judge_max_tokens", lambda: 4096)
    route = {"provider": "antigravity", "model": "test-model"}
    assert goals.judge_goal("goal", "answer", route=route)[0] == "done"
    assert call.call_args.kwargs["allow_fallback"] is False
    assert call.call_args.kwargs["provider"] == route["provider"]
    assert call.call_args.kwargs["model"] == route["model"]
    assert call.call_args.kwargs["max_tokens"] == 4096


@pytest.mark.parametrize("pinned", [True, False])
def test_primary_worker_fallback_policy_is_scoped_to_pinned_run(monkeypatch, pinned):
    import cli

    chain = [{"provider": "other", "model": "other-model"}]
    monkeypatch.setattr(cli, "get_fallback_chain", lambda _config: chain)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.delenv(WORKER_ROUTE_ENV, raising=False)
    if pinned:
        monkeypatch.setenv(WORKER_ROUTE_ENV, json.dumps({"provider": "antigravity", "model": "test-model"}))
    shell = cli.HermesCLI(compact=True)
    assert shell._fallback_model == ([] if pinned else chain)
