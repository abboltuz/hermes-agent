"""Contracts for bounded launch_app and set_window_frame actions."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pytest


class _Session:
    def __init__(
        self,
        output: Dict[str, Any],
        *,
        tools: Optional[set[str]] = None,
        properties: Optional[Dict[str, set[str]]] = None,
    ) -> None:
        self.output = output
        self.tools = tools or set()
        self.properties = properties or {}
        self.calls = []

    def _has_tool(self, name: str) -> bool:
        return name in self.tools

    def supports_input_property(self, tool: str, name: str) -> bool:
        return name in self.properties.get(tool, set())

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.calls.append((name, dict(args)))
        return self.output


def _backend(session: _Session):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "public-test-session"
    backend._active_pid = None
    backend._active_window_id = None
    backend._last_app = "existing-target"
    backend._last_target = {"pid": 5, "window_id": 6}
    backend._snapshot_tokens = {1: "token"}
    backend._snapshot_id = "snapshot"
    return backend


@pytest.fixture(autouse=True)
def _reset_callback():
    from tools.computer_use.tool import reset_backend_for_tests, set_approval_callback

    reset_backend_for_tests()
    set_approval_callback(None)
    yield
    set_approval_callback(None)
    reset_backend_for_tests()


def test_schema_exposes_only_bounded_window_management_fields():
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA

    properties = COMPUTER_USE_SCHEMA["parameters"]["properties"]
    actions = properties["action"]["enum"]
    assert "launch_app" in actions
    assert "set_window_frame" in actions
    assert properties["frame"]["required"] == ["x", "y", "width", "height"]
    assert properties["frame"]["additionalProperties"] is False
    assert properties["frame"]["properties"]["width"]["minimum"] == 1
    assert properties["frame"]["properties"]["height"]["minimum"] == 1
    for excluded in ("urls", "additional_arguments", "webkit_inspector_port"):
        assert excluded not in properties


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "launch_app"},
        {"action": "launch_app", "app": ""},
        {"action": "launch_app", "app": "Notes", "bundle_id": "com.apple.Notes"},
        {"action": "launch_app", "app": "Notes", "creates_new_application_instance": 1},
        {"action": "set_window_frame", "window_id": 2, "frame": {"x": 0, "y": 0, "width": 1, "height": 1}},
        {"action": "set_window_frame", "pid": 1, "window_id": 2, "frame": {"x": 0, "y": 0, "width": 0, "height": 1}},
        {"action": "set_window_frame", "pid": 1, "window_id": 2, "frame": {"x": 0, "y": 0, "width": 1, "height": 1, "extra": 2}},
    ],
)
def test_invalid_requests_are_refused_before_approval_or_backend(monkeypatch, arguments):
    from tools.computer_use import tool

    approvals = []
    tool.set_approval_callback(lambda *values: approvals.append(values) or "approve_once")
    monkeypatch.setattr(
        tool, "_get_backend",
        lambda *args, **kwargs: pytest.fail("backend must not be created"),
    )

    result = json.loads(tool.handle_computer_use(arguments))

    assert result["code"] == "invalid_window_management_arguments"
    assert approvals == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "launch_app", "bundle_id": "com.apple.Notes"},
        {
            "action": "set_window_frame", "pid": 1, "window_id": 2,
            "frame": {"x": -100, "y": 10, "width": 800, "height": 600},
        },
    ],
)
def test_valid_mutations_require_approval_before_dispatch(monkeypatch, arguments):
    from tools.computer_use import tool

    summaries = []
    tool.set_approval_callback(
        lambda action, args, summary: summaries.append(summary) or "deny"
    )
    monkeypatch.setattr(
        tool, "_get_backend",
        lambda *args, **kwargs: pytest.fail("denied action must not dispatch"),
    )

    result = json.loads(tool.handle_computer_use(arguments))

    assert result == {"error": "denied by user", "action": arguments["action"]}
    assert len(summaries) == 1


def test_launch_approval_summary_names_identifier_and_instance_flag():
    from tools.computer_use.tool import _summarize_action

    summary = _summarize_action("launch_app", {
        "bundle_id": "com.apple.Notes",
        "creates_new_application_instance": True,
    })
    assert "com.apple.Notes" in summary
    assert "bundle_id" in summary
    assert "new native instance" in summary


@pytest.mark.parametrize("windows", [[], [{"window_id": 1}, {"window_id": 2}]])
def test_launch_preserves_zero_or_many_windows_without_session_or_targeting(windows):
    structured = {
        "pid": 41,
        "bundle_id": "com.apple.Notes",
        "name": "Notes",
        "windows": windows,
        "launch_state": "launched",
        "self_activation_suppressed": True,
    }
    session = _Session(
        {"isError": False, "data": None, "structuredContent": structured},
        tools={"launch_app"},
        properties={"launch_app": {"bundle_id", "creates_new_application_instance"}},
    )
    backend = _backend(session)
    prior = (backend._last_app, dict(backend._last_target), dict(backend._snapshot_tokens))

    result = backend.launch_app(
        bundle_id="com.apple.Notes", creates_new_application_instance=True,
    )

    assert result == structured
    assert session.calls == [("launch_app", {
        "bundle_id": "com.apple.Notes",
        "creates_new_application_instance": True,
    })]
    assert "session" not in session.calls[0][1]
    assert (backend._last_app, backend._last_target, backend._snapshot_tokens) == prior


def test_launch_suppression_false_requires_fresh_observation():
    from tools.computer_use.tool import _launch_app_response

    result = json.loads(_launch_app_response({
        "pid": 41,
        "windows": [{"window_id": 2}],
        "launch_state": "launched",
        "self_activation_suppressed": False,
    }))

    assert result["windows"] == [{"window_id": 2}]
    assert result["task_success"] is False
    assert result["foreground_preservation"] == {
        "status": "failed",
        "condition": "foreground_changed",
        "warning": (
            "The launched target held focus despite the driver's re-demotion "
            "attempt. Hermes did not request focus or raise; inspect fresh "
            "state before continuing."
        ),
    }
    assert result["verdict"] == {
        "decision": "verify_fresh_state",
        "reason": "foreground_changed",
    }


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"message": "launched"},
        {"pid": None, "windows": None, "launch_state": ""},
    ],
)
def test_launch_missing_structured_evidence_is_not_silent_success(result):
    from tools.computer_use.tool import _launch_app_response

    payload = json.loads(_launch_app_response(result))

    assert payload["ok"] is True
    assert payload["task_success"] is False
    assert payload["launch_evidence"] == {
        "status": "missing",
        "reason": "missing_launch_evidence",
    }
    assert payload["verdict"] == {
        "decision": "verify_fresh_state",
        "reason": "missing_launch_evidence",
    }


def test_launch_logical_error_is_fail_closed_and_not_replayed():
    from tools.computer_use.cua_backend import CuaDriverCallError

    session = _Session(
        {
            "isError": True,
            "data": "launch outcome unknown",
            "structuredContent": {
                "code": "timeout_outcome_unknown",
                "message": "launch outcome unknown",
            },
        },
        tools={"launch_app"},
        properties={"launch_app": {"name"}},
    )

    with pytest.raises(CuaDriverCallError) as exc:
        _backend(session).launch_app(name="Notes")

    assert exc.value.code == "timeout_outcome_unknown"
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    ("tool", "properties", "operation", "code"),
    [
        (set(), {"launch_app": {"bundle_id"}}, "launch", "launch_app_unsupported"),
        ({"launch_app"}, {"launch_app": set()}, "launch", "launch_app_unsupported"),
        (set(), {"set_window_frame": {"pid"}}, "frame", "set_window_frame_unsupported"),
        ({"set_window_frame"}, {"set_window_frame": {"pid"}}, "frame", "set_window_frame_unsupported"),
    ],
)
def test_live_capability_property_gates_make_no_driver_call(
    tool, properties, operation, code,
):
    from tools.computer_use.backend import ComputerUseCapabilityError

    session = _Session({}, tools=tool, properties=properties)
    backend = _backend(session)
    with pytest.raises(ComputerUseCapabilityError) as exc:
        if operation == "launch":
            backend.launch_app(bundle_id="com.apple.Notes")
        else:
            backend.set_window_frame(
                pid=1, window_id=2,
                frame={"x": 0, "y": 0, "width": 1, "height": 1},
            )
    assert exc.value.code == code
    assert session.calls == []


def test_set_window_frame_forwards_negative_desktop_origin_without_sticky_target():
    from tools.computer_use.tool import _action_payload

    session = _Session(
        {
            "isError": False,
            "data": {"message": "frame applied"},
            "structuredContent": {
                "verified": True,
                "effect": "confirmed",
                "frame": {"x": -1440, "y": -20, "width": 1440, "height": 900},
            },
        },
        tools={"set_window_frame"},
        properties={
            "set_window_frame": {
                "pid", "window_id", "x", "y", "width", "height", "session",
            },
        },
    )
    backend = _backend(session)
    backend._active_pid = None
    backend._active_window_id = None

    result = backend.set_window_frame(
        pid=41,
        window_id=73,
        frame={"x": -1440, "y": -20, "width": 1440, "height": 900},
    )

    assert session.calls == [("set_window_frame", {
        "pid": 41,
        "window_id": 73,
        "x": -1440,
        "y": -20,
        "width": 1440,
        "height": 900,
        "session": "public-test-session",
    })]
    assert _action_payload(result)["verdict"] == {"decision": "verify_postcondition"}
    assert result.meta["frame"]["x"] == -1440
    assert backend._last_app == "existing-target"


def test_set_window_frame_omits_session_when_live_schema_does_not_accept_it():
    session = _Session(
        {"isError": False, "data": None, "structuredContent": {"effect": "confirmed"}},
        tools={"set_window_frame"},
        properties={
            "set_window_frame": {"pid", "window_id", "x", "y", "width", "height"},
        },
    )

    _backend(session).set_window_frame(
        pid=41, window_id=73,
        frame={"x": 0, "y": 0, "width": 1, "height": 1},
    )

    assert "session" not in session.calls[0][1]


def test_set_window_frame_unknown_transport_is_not_replayed():
    from tools.computer_use.tool import _action_payload

    session = _Session(
        {
            "isError": True,
            "data": "frame outcome unknown",
            "structuredContent": {
                "code": "transport_outcome_unknown",
                "message": "frame outcome unknown",
            },
        },
        tools={"set_window_frame"},
        properties={
            "set_window_frame": {"pid", "window_id", "x", "y", "width", "height"},
        },
    )

    result = _backend(session).set_window_frame(
        pid=41, window_id=73,
        frame={"x": 0, "y": 0, "width": 1, "height": 1},
    )

    assert len(session.calls) == 1
    assert result.code == "transport_outcome_unknown"
    assert _action_payload(result)["verdict"] == {"decision": "verify_fresh_state"}


def test_approved_dispatch_maps_public_app_to_driver_name_in_order(monkeypatch):
    from tools.computer_use import tool

    events = []
    session = _Session(
        {
            "isError": False,
            "data": None,
            "structuredContent": {
                "pid": 41,
                "name": "Notes",
                "windows": [],
                "launch_state": "launched",
                "self_activation_suppressed": True,
            },
        },
        tools={"launch_app"},
        properties={"launch_app": {"name", "creates_new_application_instance"}},
    )
    original_call = session.call_tool

    def call_tool(name, args, timeout=30.0):
        events.append(("driver", name, dict(args)))
        return original_call(name, args, timeout)

    session.call_tool = call_tool
    backend = _backend(session)
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": backend)
    tool.set_approval_callback(
        lambda action, args, summary: events.append(("approval", action)) or "approve_once"
    )

    result = json.loads(tool.handle_computer_use({
        "action": "launch_app",
        "app": " Notes ",
        "creates_new_application_instance": True,
    }))

    assert result["name"] == "Notes"
    assert events == [
        ("approval", "launch_app"),
        ("driver", "launch_app", {
            "name": "Notes",
            "creates_new_application_instance": True,
        }),
    ]


def test_approved_dispatch_flattens_nested_frame_exactly_in_order(monkeypatch):
    from tools.computer_use import tool

    events = []
    session = _Session(
        {
            "isError": False,
            "data": {"message": "frame applied"},
            "structuredContent": {"verified": True, "effect": "confirmed"},
        },
        tools={"set_window_frame"},
        properties={
            "set_window_frame": {
                "pid", "window_id", "x", "y", "width", "height", "session",
            },
        },
    )
    original_call = session.call_tool

    def call_tool(name, args, timeout=30.0):
        events.append(("driver", name, dict(args)))
        return original_call(name, args, timeout)

    session.call_tool = call_tool
    backend = _backend(session)
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": backend)
    tool.set_approval_callback(
        lambda action, args, summary: events.append(("approval", action)) or "approve_once"
    )

    result = json.loads(tool.handle_computer_use({
        "action": "set_window_frame",
        "pid": 41,
        "window_id": 73,
        "frame": {"x": -1440, "y": -20, "width": 1440, "height": 900},
    }))

    assert result["verdict"] == {"decision": "verify_postcondition"}
    assert events == [
        ("approval", "set_window_frame"),
        ("driver", "set_window_frame", {
            "pid": 41,
            "window_id": 73,
            "x": -1440,
            "y": -20,
            "width": 1440,
            "height": 900,
            "session": "public-test-session",
        }),
    ]


def test_mutations_are_not_safe_for_transport_replay():
    from tools.computer_use.cua_backend import _CuaDriverSession

    assert _CuaDriverSession._transport_replay_is_safe("launch_app") is False
    assert _CuaDriverSession._transport_replay_is_safe("set_window_frame") is False


def test_legacy_backend_remains_instantiable_with_structured_unsupported_hooks():
    from tools.computer_use.backend import ComputerUseBackend

    # The two new hooks are concrete defaults, not additions to the ABC.
    assert "launch_app" not in ComputerUseBackend.__abstractmethods__
    assert "set_window_frame" not in ComputerUseBackend.__abstractmethods__


def test_prompt_guidance_distinguishes_desktop_frame_and_no_focus_launch():
    from agent.prompt_builder import computer_use_guidance

    guidance = computer_use_guidance("darwin")
    assert "Hermes never requests focus/raise; app may self-activate" in guidance
    assert "frame={x,y,width,height}` in desktop coordinates" in guidance
    assert "verify_state` bounds predicate" in guidance
