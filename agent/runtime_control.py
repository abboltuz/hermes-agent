"""Typed, agent-local control signals for irreversible runtime boundaries."""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass
from typing import Optional


KANBAN_TERMINAL_TOOL_NAMES = frozenset(
    {
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
    }
)


@dataclass(frozen=True)
class KanbanTerminalTransition:
    """Authoritative worker-owned lifecycle transition committed by one run."""

    tool_name: str
    task_id: str
    run_id: int
    session_id: str
    status: str

    def as_dict(self) -> dict:
        return asdict(self)

    def exit_reason(self) -> str:
        return f"kanban_terminal_transition({self.tool_name}:{self.status})"

    def closure_message(self) -> str:
        return (
            f"Kanban lifecycle transition committed for task {self.task_id} "
            f"(run {self.run_id}, status {self.status}); worker execution ended."
        )


class RuntimeControl:
    """Per-agent state for control flow that must not depend on tool prose.

    Kanban authority is captured when the dispatcher-owned agent is created.
    A producer may arm the fence only for that exact task, run, and current
    session after its guarded database mutation succeeds.
    """

    def __init__(
        self,
        *,
        task_id: Optional[str],
        run_id: Optional[int],
        session_id: str,
        dispatcher_owned: bool,
    ) -> None:
        self._task_id = task_id
        self._run_id = run_id
        self._session_id = session_id
        self._dispatcher_owned = dispatcher_owned
        self._kanban_terminal_transition: Optional[KanbanTerminalTransition] = None
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, *, session_id: str) -> "RuntimeControl":
        task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip() or None
        raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
        try:
            run_id = int(raw_run_id) if raw_run_id else None
        except ValueError:
            run_id = None
        try:
            from agent.delegation_context import is_dispatcher_owned_worker_context

            dispatcher_owned = bool(is_dispatcher_owned_worker_context())
        except Exception:
            dispatcher_owned = False
        return cls(
            task_id=task_id,
            run_id=run_id,
            session_id=session_id,
            dispatcher_owned=dispatcher_owned,
        )

    @property
    def kanban_terminal_transition(self) -> Optional[KanbanTerminalTransition]:
        with self._lock:
            return self._kanban_terminal_transition

    def bind_session(self, session_id: str) -> None:
        """Track compression-driven session rotation until the fence arms."""
        if not session_id:
            return
        with self._lock:
            if self._kanban_terminal_transition is None:
                self._session_id = session_id

    def authorizes_kanban_terminal_call(
        self,
        *,
        tool_name: str,
        task_id: Optional[str],
        run_id: Optional[int],
        session_id: str,
    ) -> bool:
        """Return whether this call can commit this actor's terminal fence."""
        if tool_name not in KANBAN_TERMINAL_TOOL_NAMES or run_id is None:
            return False
        with self._lock:
            return bool(
                self._kanban_terminal_transition is None
                and self._dispatcher_owned
                and task_id == self._task_id
                and int(run_id) == self._run_id
                and session_id
                and session_id == self._session_id
            )

    def authorizes_kanban_terminal_attempt(
        self,
        *,
        tool_name: str,
        run_id: Optional[int],
        session_id: str,
    ) -> bool:
        """Return whether rewrites could turn this attempt into an exact call."""
        if tool_name not in KANBAN_TERMINAL_TOOL_NAMES or run_id is None:
            return False
        with self._lock:
            return bool(
                self._kanban_terminal_transition is None
                and self._dispatcher_owned
                and self._task_id
                and int(run_id) == self._run_id
                and session_id
                and session_id == self._session_id
            )

    def commit_kanban_terminal_transition(
        self,
        *,
        tool_name: str,
        task_id: str,
        run_id: Optional[int],
        session_id: str,
        status: str,
    ) -> bool:
        """Arm once when the producer proves exact dispatcher-run authority."""
        if tool_name not in KANBAN_TERMINAL_TOOL_NAMES or run_id is None:
            return False
        with self._lock:
            if self._kanban_terminal_transition is not None:
                return self._kanban_terminal_transition == KanbanTerminalTransition(
                    tool_name=tool_name,
                    task_id=task_id,
                    run_id=int(run_id),
                    session_id=session_id,
                    status=status,
                )
            if not self._dispatcher_owned:
                return False
            if task_id != self._task_id or int(run_id) != self._run_id:
                return False
            if not session_id or session_id != self._session_id:
                return False
            self._kanban_terminal_transition = KanbanTerminalTransition(
                tool_name=tool_name,
                task_id=task_id,
                run_id=int(run_id),
                session_id=session_id,
                status=status,
            )
            return True


def get_kanban_terminal_transition(agent) -> Optional[KanbanTerminalTransition]:
    control = getattr(agent, "_runtime_control", None)
    if not isinstance(control, RuntimeControl):
        return None
    return control.kanban_terminal_transition


__all__ = [
    "KANBAN_TERMINAL_TOOL_NAMES",
    "KanbanTerminalTransition",
    "RuntimeControl",
    "get_kanban_terminal_transition",
]
