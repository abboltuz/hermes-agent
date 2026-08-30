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


_TERMINAL_HANDLER_CASES = [
    ("_handle_complete", {"summary": "verified implementation"}),
    ("_handle_block", {"reason": "external dependency"}),
    ("_handle_request_review", {"summary": "ready for review"}),
    ("_handle_request_changes", {"reason": "add the missing regression"}),
]


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


def test_terminal_on_iteration_limit_is_truthful_without_failure_fallback(
    worker_case,
):
    agent, task_id, run_id = worker_case
    from hermes_cli import kanban_db as kb

    agent.max_iterations = 1
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("kanban_block", {"reason": "preflight failed"}, "block-1")
        ]
    )

    result = agent.run_conversation("perform the worker task")

    assert result["terminal_transition"]["run_id"] == run_id
    assert result["completed"] is True
    assert result["failed"] is False
    conn = kb.connect()
    try:
        events = kb.list_events(conn, task_id)
        terminal_kinds = [
            event.kind
            for event in events
            if event.kind in {"blocked", "timed_out", "gave_up"}
        ]
        assert terminal_kinds == ["blocked"]
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.consecutive_failures == 0
    finally:
        conn.close()


def test_terminal_boundary_suppresses_generic_post_tool_hooks(worker_case):
    agent, _task_id, _run_id = worker_case
    agent._apply_pending_steer_to_tool_results = MagicMock()
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("kanban_block", {"reason": "preflight failed"}, "block-1"),
            _tool_call(_SENTINEL_TOOL, {"phase": "same-response"}, "sentinel-2"),
        ]
    )

    with (
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch("hermes_cli.lifecycle.invoke_hook") as invoke_hook,
    ):
        result = agent.run_conversation("perform the worker task")

    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    hook_names = [call.args[0] for call in invoke_hook.call_args_list if call.args]
    assert "post_tool_call" not in hook_names
    assert "transform_tool_result" not in hook_names
    agent._apply_pending_steer_to_tool_results.assert_not_called()


def test_terminal_transition_wins_when_tool_result_persistence_initially_fails(
    worker_case,
):
    agent, _task_id, _run_id = worker_case
    failed_tool_result_flush = False

    def _flush(messages, *_args, **_kwargs):
        nonlocal failed_tool_result_flush
        if (
            not failed_tool_result_flush
            and messages
            and messages[-1].get("role") == "tool"
        ):
            failed_tool_result_flush = True
            agent._last_persistence_error_cause = "locked"
            return False
        return True

    agent._flush_messages_to_session_db = _flush
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call(
                    "kanban_block",
                    {"reason": "preflight failed"},
                    "block-1",
                ),
                _tool_call(
                    _SENTINEL_TOOL,
                    {"phase": "same-response"},
                    "sentinel-2",
                ),
            ]
        ),
        _response(content="must not be requested", finish_reason="stop"),
    ]

    result = agent.run_conversation("perform the worker task")

    assert failed_tool_result_flush is True
    assert agent.client.chat.completions.create.call_count == 1
    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    assert result["completed"] is True
    assert [
        message["tool_call_id"]
        for message in result["messages"]
        if message.get("role") == "tool"
    ] == ["block-1", "sentinel-2"]
    assert result["messages"][-1]["role"] == "assistant"


def test_segmented_terminal_fence_pairs_later_parallel_calls_after_flush_failure(
    worker_case,
    monkeypatch,
):
    agent, _task_id, _run_id = worker_case
    from tools.registry import registry

    agent.valid_tool_names.add("read_file")
    dispatched_reads = []
    original_dispatch = registry.dispatch

    def _dispatch(name, args, **kwargs):
        if name == "read_file":
            dispatched_reads.append(dict(args))
        return original_dispatch(name, args, **kwargs)

    monkeypatch.setattr(registry, "dispatch", _dispatch)
    failed_tool_result_flush = False

    def _flush(messages, *_args, **_kwargs):
        nonlocal failed_tool_result_flush
        if (
            not failed_tool_result_flush
            and messages
            and messages[-1].get("role") == "tool"
        ):
            failed_tool_result_flush = True
            agent._last_persistence_error_cause = "locked"
            return False
        return True

    agent._flush_messages_to_session_db = _flush
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("kanban_block", {"reason": "preflight failed"}, "block-1"),
            _tool_call("read_file", {"path": "/tmp/one"}, "read-2"),
            _tool_call("read_file", {"path": "/tmp/two"}, "read-3"),
        ]
    )

    result = agent.run_conversation("perform the worker task")

    assert failed_tool_result_flush is True
    assert dispatched_reads == []
    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    assert [
        message["tool_call_id"]
        for message in result["messages"]
        if message.get("role") == "tool"
    ] == ["block-1", "read-2", "read-3"]
    assert result["messages"][-1]["role"] == "assistant"


def test_steer_racing_with_terminal_commit_is_discarded(worker_case, monkeypatch):
    agent, _task_id, _run_id = worker_case
    control = agent._runtime_control
    original_commit = control.commit_kanban_terminal_transition

    def _commit_and_steer(**kwargs):
        committed = original_commit(**kwargs)
        if committed:
            assert agent.steer("continue with another provider turn") is True
        return committed

    monkeypatch.setattr(
        control,
        "commit_kanban_terminal_transition",
        _commit_and_steer,
    )
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call(
                    "kanban_block",
                    {"reason": "preflight failed"},
                    "block-1",
                )
            ]
        ),
        _response(content="must not be requested", finish_reason="stop"),
    ]

    result = agent.run_conversation("perform the worker task")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    assert result["completed"] is True
    assert "pending_steer" not in result
    assert agent._pending_steer is None
    assert result["messages"][-1]["role"] == "assistant"


def test_post_commit_diagnostic_failure_still_arms_terminal_fence(
    worker_case,
    monkeypatch,
):
    agent, _task_id, _run_id = worker_case
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(
        kb,
        "latest_run",
        MagicMock(side_effect=RuntimeError("diagnostic read failed")),
    )
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call(
                    "kanban_block",
                    {"reason": "preflight failed"},
                    "block-1",
                )
            ]
        ),
        _response(
            tool_calls=[
                _tool_call(_SENTINEL_TOOL, {"phase": "after-error"}, "sentinel-2")
            ]
        ),
    ]

    result = agent.run_conversation("perform the worker task")

    assert agent.client.chat.completions.create.call_count == 1
    assert _sentinel_calls == []
    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    assert result["completed"] is True


def test_post_commit_integrity_check_failure_still_arms_terminal_fence(
    worker_case,
    monkeypatch,
):
    agent, _task_id, _run_id = worker_case
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(
        kb,
        "_check_file_length_invariant",
        MagicMock(side_effect=RuntimeError("post-commit integrity check failed")),
    )
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call(
                    "kanban_block",
                    {"reason": "preflight failed"},
                    "block-1",
                )
            ]
        ),
        _response(
            tool_calls=[
                _tool_call(_SENTINEL_TOOL, {"phase": "after-error"}, "sentinel-2")
            ]
        ),
    ]

    result = agent.run_conversation("perform the worker task")

    assert agent.client.chat.completions.create.call_count == 1
    assert _sentinel_calls == []
    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    assert result["completed"] is True


def test_terminal_tools_bypass_post_dispatch_middleware_and_relay(
    worker_case,
    monkeypatch,
):
    agent, _task_id, _run_id = worker_case
    from agent import relay_tools
    from hermes_cli import middleware

    middleware_after_dispatch = []
    relay_after_dispatch = []

    def _execution_wrapper(tool_name, args, terminal_call, **_kwargs):
        try:
            return terminal_call(args)
        finally:
            if (
                tool_name == "kanban_block"
                and agent._runtime_control.kanban_terminal_transition is not None
            ):
                middleware_after_dispatch.append(tool_name)

    def _relay_wrapper(tool_name, args, callback, **_kwargs):
        try:
            return callback(args), args
        finally:
            if (
                tool_name == "kanban_block"
                and agent._runtime_control.kanban_terminal_transition is not None
            ):
                relay_after_dispatch.append(tool_name)

    monkeypatch.setattr(middleware, "run_tool_execution_middleware", _execution_wrapper)
    monkeypatch.setattr(relay_tools, "execute", _relay_wrapper)
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call("kanban_block", {"reason": "preflight failed"}, "block-1")
        ]
    )

    result = agent.run_conversation("perform the worker task")

    assert result["terminal_transition"]["tool_name"] == "kanban_block"
    assert middleware_after_dispatch == []
    assert relay_after_dispatch == []


def test_request_rewrite_to_foreign_task_returns_to_rejected_pipeline(
    worker_case,
    monkeypatch,
):
    agent, _task_id, _run_id = worker_case
    from agent import relay_tools
    from hermes_cli import kanban_db as kb
    from hermes_cli import middleware

    conn = kb.connect()
    try:
        foreign_id = kb.create_task(conn, title="foreign rewrite", assignee="other")
    finally:
        conn.close()

    middleware_after_dispatch = []
    relay_after_dispatch = []

    monkeypatch.setattr(
        middleware,
        "apply_tool_request_middleware",
        lambda _name, args, **_kwargs: SimpleNamespace(
            payload={**args, "task_id": foreign_id},
            trace=[],
        ),
    )

    def _execution_wrapper(tool_name, args, terminal_call, **_kwargs):
        result = terminal_call(args)
        middleware_after_dispatch.append(tool_name)
        return result

    def _relay_wrapper(tool_name, args, callback, **_kwargs):
        result = callback(args)
        relay_after_dispatch.append(tool_name)
        return result, args

    monkeypatch.setattr(middleware, "run_tool_execution_middleware", _execution_wrapper)
    monkeypatch.setattr(relay_tools, "execute", _relay_wrapper)
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call("kanban_block", {"reason": "rewritten"}, "block-1")
            ]
        ),
        _response(content="rewrite rejected", finish_reason="stop"),
    ]

    result = agent.run_conversation("perform the worker task")

    assert result["final_response"] == "rewrite rejected"
    assert "terminal_transition" not in result
    assert middleware_after_dispatch == ["kanban_block"]
    assert relay_after_dispatch == ["kanban_block"]


def test_relay_rewrite_runs_before_exact_terminal_authority_decision(
    worker_case,
    monkeypatch,
):
    agent, _task_id, _run_id = worker_case
    from agent import relay_tools
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        foreign_id = kb.create_task(conn, title="relay foreign", assignee="other")
    finally:
        conn.close()

    relay_calls = []

    def _relay_rewrite(tool_name, args, callback, **_kwargs):
        relay_calls.append(tool_name)
        rewritten = {**args, "task_id": foreign_id}
        return callback(rewritten), rewritten

    monkeypatch.setattr(relay_tools, "execute", _relay_rewrite)
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call("kanban_block", {"reason": "relay rewrite"}, "block-1")
            ]
        ),
        _response(content="relay rewrite rejected", finish_reason="stop"),
    ]

    result = agent.run_conversation("perform the worker task")

    assert result["final_response"] == "relay rewrite rejected"
    assert "terminal_transition" not in result
    assert relay_calls == ["kanban_block"]


def test_foreign_to_exact_rewrite_cannot_commit_inside_wrapper_finally(
    worker_case,
    monkeypatch,
):
    agent, task_id, _run_id = worker_case
    from agent import relay_tools
    from hermes_cli import kanban_db as kb
    from hermes_cli import middleware

    conn = kb.connect()
    try:
        foreign_id = kb.create_task(conn, title="foreign source", assignee="other")
    finally:
        conn.close()

    middleware_post_commit = []
    relay_post_commit = []

    monkeypatch.setattr(
        middleware,
        "apply_tool_request_middleware",
        lambda _name, args, **_kwargs: SimpleNamespace(
            payload={**args, "task_id": task_id},
            trace=[],
        ),
    )

    def _execution_wrapper(tool_name, args, terminal_call, **_kwargs):
        try:
            return terminal_call(args)
        finally:
            if agent._runtime_control.kanban_terminal_transition is not None:
                middleware_post_commit.append(tool_name)

    def _relay_wrapper(tool_name, args, callback, **_kwargs):
        try:
            return callback(args), args
        finally:
            if agent._runtime_control.kanban_terminal_transition is not None:
                relay_post_commit.append(tool_name)

    monkeypatch.setattr(middleware, "run_tool_execution_middleware", _execution_wrapper)
    monkeypatch.setattr(relay_tools, "execute", _relay_wrapper)
    agent.client.chat.completions.create.return_value = _response(
        tool_calls=[
            _tool_call(
                "kanban_block",
                {"task_id": foreign_id, "reason": "rewritten to owned task"},
                "block-1",
            )
        ]
    )

    result = agent.run_conversation("perform the worker task")

    assert result["terminal_transition"]["task_id"] == task_id
    assert middleware_post_commit == []
    assert relay_post_commit == []


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


def _prepare_terminal_handler_case(worker_case, handler_name, monkeypatch):
    agent, task_id, implementation_run_id = worker_case
    from hermes_cli import kanban_db as kb

    run_id = implementation_run_id
    if handler_name == "_handle_request_changes":
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
            run_id = int(review.current_run_id)
        finally:
            conn.close()

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return agent, task_id


def _terminal_db_snapshot(task_id):
    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        return (
            vars(task).copy(),
            [vars(run).copy() for run in kb.list_runs(conn, task_id=task_id)],
            [vars(event).copy() for event in kb.list_events(conn, task_id)],
        )
    finally:
        conn.close()


@pytest.mark.parametrize(("handler_name", "arguments"), _TERMINAL_HANDLER_CASES)
def test_dispatcher_terminal_handlers_fail_closed_without_run_id(
    worker_case,
    monkeypatch,
    handler_name,
    arguments,
):
    from agent.runtime_control import RuntimeControl
    from tools import kanban_tools as kt

    agent, task_id = _prepare_terminal_handler_case(
        worker_case,
        handler_name,
        monkeypatch,
    )
    before = _terminal_db_snapshot(task_id)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    control = RuntimeControl.from_environment(session_id=agent.session_id)

    result = json.loads(
        getattr(kt, handler_name)(
            arguments,
            runtime_control=control,
            session_id=agent.session_id,
        )
    )

    assert "error" in result
    assert "HERMES_KANBAN_RUN_ID" in result["error"]
    assert "missing" in result["error"]
    assert _terminal_db_snapshot(task_id) == before
    assert control.kanban_terminal_transition is None


@pytest.mark.parametrize(("handler_name", "arguments"), _TERMINAL_HANDLER_CASES)
def test_dispatcher_terminal_handlers_fail_closed_with_malformed_run_id(
    worker_case,
    monkeypatch,
    handler_name,
    arguments,
):
    from agent.runtime_control import RuntimeControl
    from tools import kanban_tools as kt

    agent, task_id = _prepare_terminal_handler_case(
        worker_case,
        handler_name,
        monkeypatch,
    )
    before = _terminal_db_snapshot(task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "not-an-integer")
    control = RuntimeControl.from_environment(session_id=agent.session_id)

    result = json.loads(
        getattr(kt, handler_name)(
            arguments,
            runtime_control=control,
            session_id=agent.session_id,
        )
    )

    assert "error" in result
    assert "HERMES_KANBAN_RUN_ID" in result["error"]
    assert "integer" in result["error"]
    assert _terminal_db_snapshot(task_id) == before
    assert control.kanban_terminal_transition is None


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


def test_rejected_terminal_call_preserves_middleware_and_relay(
    worker_case,
    monkeypatch,
):
    agent, _task_id, run_id = worker_case
    from agent import relay_tools
    from hermes_cli import middleware

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id + 1))
    middleware_after_dispatch = []
    relay_after_dispatch = []

    def _execution_wrapper(tool_name, args, terminal_call, **_kwargs):
        result = terminal_call(args)
        middleware_after_dispatch.append(tool_name)
        return result

    def _relay_wrapper(tool_name, args, callback, **_kwargs):
        result = callback(args)
        relay_after_dispatch.append(tool_name)
        return result, args

    monkeypatch.setattr(middleware, "run_tool_execution_middleware", _execution_wrapper)
    monkeypatch.setattr(relay_tools, "execute", _relay_wrapper)
    agent.client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _tool_call("kanban_block", {"reason": "stale attempt"}, "block-1")
            ]
        ),
        _response(content="rejected call handled", finish_reason="stop"),
    ]

    result = agent.run_conversation("perform the worker task")

    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "rejected call handled"
    assert "terminal_transition" not in result
    assert middleware_after_dispatch == ["kanban_block"]
    assert relay_after_dispatch == ["kanban_block"]


def test_session_mismatch_does_not_arm_exact_actor_fence(worker_case):
    agent, task_id, run_id = worker_case

    armed = agent._runtime_control.commit_kanban_terminal_transition(
        tool_name="kanban_block",
        task_id=task_id,
        run_id=run_id,
        session_id="different-session",
        status="blocked",
    )

    assert armed is False
    assert agent._runtime_control.kanban_terminal_transition is None


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
    from agent import relay_tools
    from agent.runtime_control import RuntimeControl
    from hermes_cli import middleware
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

    middleware_after_dispatch = []
    relay_after_dispatch = []

    def _execution_wrapper(tool_name, args, terminal_call, **_kwargs):
        result = terminal_call(args)
        middleware_after_dispatch.append(tool_name)
        return result

    def _relay_wrapper(tool_name, args, callback, **_kwargs):
        result = callback(args)
        relay_after_dispatch.append(tool_name)
        return result, args

    monkeypatch.setattr(middleware, "run_tool_execution_middleware", _execution_wrapper)
    monkeypatch.setattr(relay_tools, "execute", _relay_wrapper)

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
    assert middleware_after_dispatch == ["kanban_block"]
    assert relay_after_dispatch == ["kanban_block"]


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
