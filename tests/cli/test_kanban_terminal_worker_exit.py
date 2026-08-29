"""One-shot CLI shutdown behavior after a Kanban actor terminal transition."""

from __future__ import annotations

from types import SimpleNamespace

import cli
import pytest

from agent.runtime_control import RuntimeControl


def _armed_control() -> RuntimeControl:
    control = RuntimeControl(
        task_id="task-1",
        run_id=7,
        session_id="session-1",
        dispatcher_owned=True,
    )
    assert control.commit_kanban_terminal_transition(
        tool_name="kanban_block",
        task_id="task-1",
        run_id=7,
        session_id="session-1",
        status="blocked",
    )
    return control


def test_goal_loop_gate_rejects_terminal_worker_result(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")

    assert cli._should_run_kanban_goal_loop({"final_response": "continue"}) is True
    assert cli._should_run_kanban_goal_loop(
        {
            "final_response": "worker ended",
            "terminal_transition": _armed_control()
            .kanban_terminal_transition
            .as_dict(),
        }
    ) is False


def test_terminal_worker_finalization_skips_linger_and_memory_postwork(monkeypatch):
    order = []
    agent = SimpleNamespace(
        session_id="session-1",
        platform="kanban",
        _runtime_control=_armed_control(),
    )
    shell = SimpleNamespace(
        agent=agent,
        session_id="session-1",
        _release_active_session=lambda: order.append("release"),
    )

    monkeypatch.setattr(
        cli,
        "_wait_for_oneshot_background_completions",
        lambda _cli: order.append("wait"),
    )
    monkeypatch.setattr(
        cli,
        "_flush_one_shot_session_store",
        lambda _cli: order.append("flush"),
    )
    monkeypatch.setattr(
        cli,
        "_notify_single_query_session_finalize",
        lambda _cli: order.append("finalize"),
    )

    def _cleanup(**kwargs):
        order.append(("cleanup", kwargs))

    monkeypatch.setattr(cli, "_run_cleanup", _cleanup)

    cli._finalize_single_query(shell)

    assert order == [
        "flush",
        (
            "cleanup",
            {
                "notify_session_finalize": False,
                "skip_memory_provider": True,
            },
        ),
        "release",
    ]


def test_quiet_goal_worker_exits_without_dispatching_goal_continuation(monkeypatch):
    calls = []
    transition = _armed_control().kanban_terminal_transition
    assert transition is not None

    def _run_conversation(*, user_message, conversation_history):
        calls.append(("run", user_message, conversation_history))
        return {
            "final_response": transition.closure_message(),
            "terminal_transition": transition.as_dict(),
            "failed": False,
        }

    class _FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "session-1"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="session-1",
                platform="kanban",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=_run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, _effective_query):
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **_kwargs):
            return True

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setattr(cli, "HermesCLI", _FakeCLI)
    monkeypatch.setattr(cli.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "_run_kanban_goal_loop_q",
        lambda *_args, **_kwargs: calls.append("goal-loop"),
    )
    monkeypatch.setattr(
        cli,
        "_finalize_single_query",
        lambda shell: calls.append(("finalize", shell.session_id)),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(query="work task-1", quiet=True, toolsets="kanban")

    assert exc_info.value.code == 0
    assert "goal-loop" not in calls
    assert ("run", "work task-1", []) in calls
    assert calls[-1] == ("finalize", "session-1")
