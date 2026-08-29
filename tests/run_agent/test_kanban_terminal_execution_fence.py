"""Behavioral proof for dispatcher-worker terminal execution fences."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


_SENTINEL_TOOL = "kanban_terminal_fence_sentinel"
_sentinel_calls: list[dict] = []


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_call(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(*, tool_calls=None, content: str = "", finish_reason: str = "tool_calls"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _register_sentinel() -> None:
    from tools.registry import registry

    if registry.get_entry(_SENTINEL_TOOL) is not None:
        return

    def _handler(args: dict, **_kwargs) -> str:
        _sentinel_calls.append(dict(args))
        return json.dumps({"ok": True})

    registry.register(
        name=_SENTINEL_TOOL,
        toolset="kanban",
        schema=_tool_def(_SENTINEL_TOOL),
        handler=_handler,
    )


@pytest.fixture
def worker_case(monkeypatch, tmp_path):
    from gateway.session_context import reset_session_vars
    from hermes_cli import kanban_db as kb

    reset_session_vars()
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "fence-worker")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="terminal-fence", assignee="fence-worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None and claimed.current_run_id is not None
        run_id = int(claimed.current_run_id)
    finally:
        conn.close()

    session_id = "fence-session"
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_SESSION_ID", session_id)
    _register_sentinel()
    _sentinel_calls.clear()

    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_def("kanban_block"), _tool_def(_SENTINEL_TOOL)],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=8,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id=session_id,
        )

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False

    yield agent, task_id, run_id
    reset_session_vars()


def _assert_single_block_receipt(task_id: str, run_id: int) -> None:
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.current_run_id is None
        runs = kb.list_runs(conn, task_id=task_id)
        assert [run.id for run in runs] == [run_id]
        assert runs[0].outcome == "blocked"
        blocked = [event for event in kb.list_events(conn, task_id) if event.kind == "blocked"]
        assert len(blocked) == 1
    finally:
        conn.close()


def test_successful_worker_block_stops_before_second_provider_call(worker_case):
    agent, task_id, run_id = worker_case
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call("kanban_block", {"reason": "preflight failed"}, "block-1")
            ]
        ),
        _response(
            tool_calls=[_tool_call(_SENTINEL_TOOL, {"phase": "later"}, "sentinel-2")]
        ),
        _response(content="continued after terminal state", finish_reason="stop"),
    ]

    result = agent.run_conversation("perform the worker task")

    assert agent.client.chat.completions.create.call_count == 1
    assert _sentinel_calls == []
    assert result["completed"] is True
    assert result["interrupted"] is False
    assert result["failed"] is False
    assert result["turn_exit_reason"].startswith("kanban_terminal_transition(")
    assert result["terminal_transition"]["task_id"] == task_id
    assert result["terminal_transition"]["run_id"] == run_id
    assert not any(
        message.get("_empty_recovery_synthetic")
        for message in result["messages"]
        if isinstance(message, dict)
    )
    _assert_single_block_receipt(task_id, run_id)


def test_terminal_tool_skips_later_same_response_call_with_exact_pairing(worker_case):
    agent, task_id, run_id = worker_case
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call("kanban_block", {"reason": "preflight failed"}, "block-1"),
                _tool_call(_SENTINEL_TOOL, {"phase": "same-response"}, "sentinel-2"),
            ]
        ),
        _response(content="should not be requested", finish_reason="stop"),
    ]

    result = agent.run_conversation("perform the worker task")

    assert agent.client.chat.completions.create.call_count == 1
    assert _sentinel_calls == []
    tool_results = [
        message for message in result["messages"] if message.get("role") == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_results] == [
        "block-1",
        "sentinel-2",
    ]
    assert len({message["tool_call_id"] for message in tool_results}) == 2
    skipped = json.loads(tool_results[1]["content"])
    assert skipped["error_type"] == "kanban_terminal_transition"
    assert skipped["terminal"] is True
    assert result["terminal_transition"]["task_id"] == task_id
    assert result["terminal_transition"]["run_id"] == run_id
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert result["messages"][-1]["content"] == result["final_response"]
    _assert_single_block_receipt(task_id, run_id)


def test_terminal_turn_does_not_run_post_turn_hooks_or_review(worker_case):
    agent, _task_id, _run_id = worker_case
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._sync_external_memory_for_turn = MagicMock()
    agent._spawn_background_review = MagicMock()
    agent._is_user_initiated_turn = True
    agent._skill_nudge_interval = 1
    agent._iters_since_skill = 1
    agent.valid_tool_names.add("skill_manage")
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("kanban_block", {"reason": "preflight failed"}, "block-1")
        ]
    )

    with patch("hermes_cli.lifecycle.invoke_hook") as invoke_hook:
        result = agent.run_conversation("perform the worker task")

    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    agent._save_trajectory.assert_not_called()
    agent._cleanup_task_resources.assert_not_called()
    agent._sync_external_memory_for_turn.assert_not_called()
    agent._spawn_background_review.assert_not_called()
    hook_names = [call.args[0] for call in invoke_hook.call_args_list if call.args]
    assert "transform_llm_output" not in hook_names
    assert "post_llm_call" not in hook_names
    assert "on_session_end" not in hook_names


@pytest.mark.parametrize(
    ("handler_name", "arguments", "expected_tool", "expected_status"),
    [
        (
            "_handle_complete",
            {"summary": "verified implementation"},
            "kanban_complete",
            "done",
        ),
        (
            "_handle_request_review",
            {"summary": "ready for independent review"},
            "kanban_request_review",
            "review",
        ),
    ],
)
def test_sibling_worker_terminal_handlers_arm_same_exact_run_contract(
    worker_case,
    handler_name,
    arguments,
    expected_tool,
    expected_status,
):
    agent, task_id, run_id = worker_case
    from tools import kanban_tools as kt

    result = json.loads(
        getattr(kt, handler_name)(
            arguments,
            runtime_control=agent._runtime_control,
            session_id=agent.session_id,
        )
    )

    assert result["ok"] is True
    transition = agent._runtime_control.kanban_terminal_transition
    assert transition is not None
    assert transition.tool_name == expected_tool
    assert transition.task_id == task_id
    assert transition.run_id == run_id
    assert transition.status == expected_status


def test_stale_run_rejection_does_not_arm_fence(worker_case, monkeypatch):
    agent, task_id, run_id = worker_case
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id + 1))
    result = json.loads(
        kt._handle_block(
            {"reason": "stale attempt"},
            runtime_control=agent._runtime_control,
            session_id=agent.session_id,
        )
    )

    assert "error" in result
    assert agent._runtime_control.kanban_terminal_transition is None
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == run_id
    finally:
        conn.close()


def test_foreign_task_rejection_does_not_arm_fence(worker_case):
    agent, task_id, _run_id = worker_case
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        foreign_id = kb.create_task(conn, title="foreign", assignee="other")
    finally:
        conn.close()
    result = json.loads(
        kt._handle_block(
            {"task_id": foreign_id, "reason": "not mine"},
            runtime_control=agent._runtime_control,
            session_id=agent.session_id,
        )
    )

    assert "error" in result
    assert task_id in result["error"]
    assert agent._runtime_control.kanban_terminal_transition is None


def test_orchestrator_transition_does_not_arm_actor_fence(worker_case, monkeypatch):
    _agent, _task_id, _run_id = worker_case
    from agent.runtime_control import RuntimeControl
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    control = RuntimeControl.from_environment(session_id="orchestrator-session")
    conn = kb.connect()
    try:
        target = kb.create_task(conn, title="orchestrated", assignee="worker")
    finally:
        conn.close()

    result = json.loads(
        kt._handle_block(
            {"task_id": target, "reason": "operator block"},
            runtime_control=control,
            session_id="orchestrator-session",
        )
    )

    assert result["ok"] is True
    assert control.kanban_terminal_transition is None


def test_orchestrator_terminal_mutation_does_not_stop_conversation_loop(
    worker_case,
    monkeypatch,
):
    agent, _task_id, _run_id = worker_case
    from agent.runtime_control import RuntimeControl
    from hermes_cli import kanban_db as kb

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    agent._runtime_control = RuntimeControl.from_environment(
        session_id=agent.session_id,
    )
    conn = kb.connect()
    try:
        target = kb.create_task(conn, title="operator-managed", assignee="worker")
    finally:
        conn.close()

    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call(
                    "kanban_block",
                    {"task_id": target, "reason": "operator decision"},
                    "block-1",
                )
            ]
        ),
        _response(content="operator conversation continues", finish_reason="stop"),
    ]

    result = agent.run_conversation("manage another task")

    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "operator conversation continues"
    assert "terminal_transition" not in result
    assert agent._runtime_control.kanban_terminal_transition is None


def test_reviewer_request_changes_arms_same_exact_review_run_contract(
    worker_case,
    monkeypatch,
):
    _agent, task_id, implementation_run_id = worker_case
    from agent.runtime_control import RuntimeControl
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        assert kb.request_review(
            conn,
            task_id,
            summary="implementation ready",
            reviewer="fence-worker",
            expected_run_id=implementation_run_id,
        )
        review = kb.claim_review_task(
            conn,
            task_id,
            claimer="fence-worker:review",
        )
        assert review is not None and review.current_run_id is not None
        review_run_id = int(review.current_run_id)
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review_run_id))
    control = RuntimeControl.from_environment(session_id="fence-session")
    result = json.loads(
        kt._handle_request_changes(
            {"reason": "add the missing regression"},
            runtime_control=control,
            session_id="fence-session",
        )
    )

    assert result["ok"] is True
    transition = control.kanban_terminal_transition
    assert transition is not None
    assert transition.tool_name == "kanban_request_changes"
    assert transition.task_id == task_id
    assert transition.run_id == review_run_id
    assert transition.status == "ready"
