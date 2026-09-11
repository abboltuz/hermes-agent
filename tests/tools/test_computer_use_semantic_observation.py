"""Focused contract tests for exact window discovery and verify_state."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional

import pytest


_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


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
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    def _has_tool(self, name: str) -> bool:
        return name in self.tools

    def supports_input_property(self, tool: str, name: str) -> bool:
        return name in self.properties.get(tool, set())

    def call_tool(self, name: str, args: Dict[str, Any], timeout: float = 30.0):
        self.calls.append((name, dict(args)))
        return self.output


def _cua_backend(session: _Session):
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend.__new__(CuaDriverBackend)
    backend._session = session
    backend._session_id = "public-test-session"
    return backend


def test_schema_exposes_bounded_verify_state_grammar():
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA

    properties = COMPUTER_USE_SCHEMA["parameters"]["properties"]
    actions = properties["action"]["enum"]
    assert "verify_state" in actions
    assert properties["expect"]["minItems"] == 1
    assert properties["expect"]["maxItems"] == 8
    assert properties["timeout_ms"]["minimum"] == 0
    assert properties["timeout_ms"]["maximum"] == 10000
    assert properties["stable_samples"]["minimum"] == 1
    assert properties["stable_samples"]["maximum"] == 5
    assert properties["include_screenshot"]["default"] is False
    bounds = properties["expect"]["items"]["properties"]["window"]["properties"]["bounds"]
    assert "minimum" not in bounds["properties"]["width"]
    assert "exclusiveMinimum" not in bounds["properties"]["width"]
    assert "minimum" not in bounds["properties"]["height"]
    assert "exclusiveMinimum" not in bounds["properties"]["height"]
    assert bounds["properties"]["tolerance_px"]["maximum"] == 100


@pytest.mark.parametrize(
    "arguments",
    [
        {"action": "verify_state", "window_id": 2, "expect": [{"window": {"exists": True}}]},
        {"action": "verify_state", "pid": 1, "expect": [{"window": {"exists": True}}]},
        {"action": "verify_state", "pid": 1, "window_id": 2, "expect": []},
        {
            "action": "verify_state", "pid": 1, "window_id": 2,
            "expect": [{"element": {"selector": {"role": "button"}, "exists": False}}],
        },
        {
            "action": "verify_state", "pid": 1, "window_id": 2,
            "expect": [{"window": {"bounds": {
                "x": 0, "y": 0, "width": 10, "height": 10,
                "tolerance_px": 101,
            }}}],
        },
        {
            "action": "verify_state", "pid": 1, "window_id": 2,
            "expect": [{"window": {"exists": True}}], "stable_samples": 6,
        },
    ],
)
def test_invalid_verify_state_is_refused_before_backend(monkeypatch, arguments):
    from tools.computer_use import tool

    monkeypatch.setattr(
        tool, "_get_backend",
        lambda *args, **kwargs: pytest.fail("backend must not be created"),
    )
    result = json.loads(tool.handle_computer_use(arguments))
    assert result["code"] == "invalid_observation_arguments"


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 0), (-320, 200), (640, -480)],
)
def test_verify_state_accepts_live_finite_unconstrained_bounds(width, height):
    from tools.computer_use.tool import _verify_state_validation_error

    assert _verify_state_validation_error({
        "pid": 1,
        "window_id": 2,
        "expect": [{"window": {"bounds": {
            "x": -100.5,
            "y": 20,
            "width": width,
            "height": height,
            "tolerance_px": 100,
        }}}],
    }) is None


def test_list_windows_defaults_onscreen_and_offscreen_requires_pid(monkeypatch):
    from tools.computer_use import tool

    class Backend:
        def __init__(self):
            self.calls = []

        def list_windows_exact(self, *, pid=None, on_screen_only=True):
            self.calls.append((pid, on_screen_only))
            return {"windows": [], "count": 0, "on_screen_only": on_screen_only}

    backend = Backend()
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": backend)

    result = json.loads(tool.handle_computer_use({"action": "list_windows"}))
    assert result["on_screen_only"] is True
    assert backend.calls == [(None, True)]

    refused = json.loads(tool.handle_computer_use({
        "action": "list_windows", "on_screen_only": False,
    }))
    assert refused["code"] == "invalid_observation_arguments"
    assert backend.calls == [(None, True)]


def test_exact_window_listing_preserves_native_metadata_and_nullable_z_index():
    raw = {
        "app_name": "Hermes",
        "pid": 41,
        "window_id": 73,
        "title": "Hidden document",
        "bounds": {"x": -1200, "y": 20, "width": 900, "height": 700},
        "z_index": None,
        "is_on_screen": False,
        "space_ids": [9],
        "current_space_id": 4,
        "on_current_space": False,
    }
    session = _Session(
        {
            "isError": False,
            "data": None,
            "images": [],
            "structuredContent": {"windows": [raw], "current_space_id": 4},
        },
        tools={"list_windows"},
        properties={"list_windows": {"pid", "on_screen_only", "session"}},
    )
    result = _cua_backend(session).list_windows_exact(pid=41, on_screen_only=False)

    assert result["windows"] == [raw]
    assert result["windows"][0]["z_index"] is None
    assert result["current_space_id"] == 4
    assert result["exact_metadata"] is True
    assert session.calls[0][1]["on_screen_only"] is False
    assert session.calls[0][1]["pid"] == 41


def test_exact_window_listing_filters_pid_mismatches_and_offscreen_rows():
    session = _Session(
        {
            "isError": False,
            "data": None,
            "images": [],
            "structuredContent": {
                "windows": [
                    {"pid": 99, "window_id": 1, "title": "other process", "is_on_screen": True},
                    {"pid": 41, "window_id": 2, "title": "hidden", "is_on_screen": False},
                    {"pid": 41, "window_id": 3, "title": "visible", "is_on_screen": True},
                ],
            },
        },
        tools={"list_windows"},
        properties={"list_windows": {"pid", "on_screen_only", "session"}},
    )

    result = _cua_backend(session).list_windows_exact(pid=41, on_screen_only=True)

    assert result["windows"] == [
        {"pid": 41, "window_id": 3, "title": "visible", "is_on_screen": True},
    ]


@pytest.mark.parametrize(
    ("tool", "properties", "call", "code"),
    [
        (set(), {"list_windows": {"on_screen_only"}}, "list", "exact_window_listing_unsupported"),
        ({"list_windows"}, {"list_windows": set()}, "list", "exact_window_listing_unsupported"),
        (set(), {"verify_state": {"pid", "window_id", "expect"}}, "verify", "verify_state_unsupported"),
        ({"verify_state"}, {"verify_state": {"pid", "window_id"}}, "verify", "verify_state_unsupported"),
    ],
)
def test_capability_and_property_gates_make_no_driver_call(tool, properties, call, code):
    from tools.computer_use.backend import ComputerUseCapabilityError

    session = _Session({}, tools=tool, properties=properties)
    backend = _cua_backend(session)
    with pytest.raises(ComputerUseCapabilityError) as exc:
        if call == "list":
            backend.list_windows_exact()
        else:
            backend.verify_state(
                pid=1, window_id=2,
                expect=[{"window": {"exists": True}}],
            )
    assert exc.value.code == code
    assert session.calls == []


@pytest.mark.parametrize(
    ("status", "decision", "task_success"),
    [
        ("satisfied", "postcondition_satisfied", True),
        ("unsatisfied", "postcondition_unsatisfied", False),
        ("unknown", "verify_fresh_state", False),
    ],
)
def test_verification_status_propagates_without_false_success(
    status, decision, task_success,
):
    from tools.computer_use.tool import _verification_response

    result = json.loads(_verification_response({
        "status": status, "predicates": [{"matched": status == "satisfied"}],
    }))
    assert result["status"] == status
    assert result["task_success"] is task_success
    assert result["verdict"]["decision"] == decision


def test_verify_state_forwards_exact_target_and_optional_bounds():
    session = _Session(
        {
            "isError": False,
            "data": None,
            "images": [],
            "structuredContent": {"status": "unsatisfied", "samples": 3},
        },
        tools={"verify_state"},
        properties={
            "verify_state": {
                "pid", "window_id", "expect", "timeout_ms",
                "stable_samples", "include_screenshot", "session",
            },
        },
    )
    expect = [{
        "window": {
            "bounds": {"x": 1, "y": 2, "width": 300, "height": 200, "tolerance_px": 4},
        },
    }]
    result = _cua_backend(session).verify_state(
        pid=41, window_id=73, expect=expect,
        timeout_ms=9000, stable_samples=3, include_screenshot=False,
    )

    assert result == {"status": "unsatisfied", "samples": 3}
    assert session.calls == [("verify_state", {
        "pid": 41,
        "window_id": 73,
        "expect": expect,
        "session": "public-test-session",
        "timeout_ms": 9000,
        "stable_samples": 3,
        "include_screenshot": False,
    })]


def test_verify_state_propagates_driver_error():
    from tools.computer_use.cua_backend import CuaDriverCallError

    session = _Session(
        {
            "isError": True,
            "data": "target disappeared",
            "images": [],
            "structuredContent": {
                "code": "target_unavailable",
                "message": "target disappeared",
            },
        },
        tools={"verify_state"},
        properties={"verify_state": {"pid", "window_id", "expect", "session"}},
    )

    with pytest.raises(CuaDriverCallError) as exc:
        _cua_backend(session).verify_state(
            pid=41, window_id=73,
            expect=[{"window": {"exists": True}}],
        )

    assert exc.value.code == "target_unavailable"
    assert exc.value.operation == "verify_state"
    assert len(session.calls) == 1


def test_verify_state_rejects_unrecognized_driver_status():
    from tools.computer_use.backend import ComputerUseCapabilityError

    session = _Session(
        {
            "isError": False,
            "data": None,
            "images": [],
            "structuredContent": {"status": "maybe"},
        },
        tools={"verify_state"},
        properties={"verify_state": {"pid", "window_id", "expect", "session"}},
    )

    with pytest.raises(ComputerUseCapabilityError) as exc:
        _cua_backend(session).verify_state(
            pid=41, window_id=73,
            expect=[{"window": {"exists": True}}],
        )

    assert exc.value.code == "invalid_verification_result"


def test_verify_state_screenshot_uses_multimodal_shaping_without_json_base64(monkeypatch):
    from tools.computer_use import tool

    session = _Session(
        {
            "isError": False,
            "data": None,
            "images": [_PNG_B64],
            "image_mime_types": ["image/png"],
            "structuredContent": {
                "status": "satisfied",
                "screenshot_png_b64": _PNG_B64,
                "screenshot_width": 8,
                "screenshot_height": 8,
            },
        },
        tools={"verify_state"},
        properties={
            "verify_state": {
                "pid", "window_id", "expect", "include_screenshot", "session",
            },
        },
    )
    backend = _cua_backend(session)
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    monkeypatch.setattr(tool, "_persist_capture_image", lambda capture: "/tmp/verify.png")

    response = tool._dispatch(backend, "verify_state", {
        "pid": 41,
        "window_id": 73,
        "expect": [{"window": {"exists": True}}],
        "include_screenshot": True,
    })

    assert response["_multimodal"] is True
    assert _PNG_B64 not in response["content"][0]["text"]
    assert response["content"][1]["image_url"]["url"].endswith(_PNG_B64)
    assert response["verification_result"]["task_success"] is True
    assert base64.b64decode(_PNG_B64).startswith(b"\x89PNG")


def test_unknown_verify_screenshot_never_becomes_task_success(monkeypatch):
    from tools.computer_use import tool

    session = _Session(
        {
            "isError": False,
            "data": None,
            "images": [_PNG_B64],
            "image_mime_types": ["image/png"],
            "structuredContent": {
                "status": "unknown",
                "screenshot_width": 8,
                "screenshot_height": 8,
            },
        },
        tools={"verify_state"},
        properties={
            "verify_state": {
                "pid", "window_id", "expect", "include_screenshot", "session",
            },
        },
    )
    monkeypatch.setattr(tool, "_should_route_through_aux_vision", lambda: False)
    monkeypatch.setattr(tool, "_persist_capture_image", lambda capture: "/tmp/verify.png")

    response = tool._dispatch(_cua_backend(session), "verify_state", {
        "pid": 41,
        "window_id": 73,
        "expect": [{"window": {"exists": True}}],
        "include_screenshot": True,
    })

    verification = response["verification_result"]
    assert verification["status"] == "unknown"
    assert verification["task_success"] is False
    assert verification["verdict"] == {
        "decision": "verify_fresh_state",
        "reason": "verification_unknown",
    }


def test_verify_state_is_safe_for_transport_replay():
    from tools.computer_use.cua_backend import _CuaDriverSession

    assert _CuaDriverSession._transport_replay_is_safe("verify_state") is True


def test_legacy_backend_keeps_old_list_windows_contract_and_is_instantiable():
    from tools.computer_use.backend import (
        ActionResult,
        CaptureResult,
        ComputerUseBackend,
        ComputerUseCapabilityError,
    )

    class LegacyBackend(ComputerUseBackend):
        def start(self): pass
        def stop(self): pass
        def is_available(self): return True
        def capture(self, mode="som", app=None, pid=None, window_id=None, max_elements=None):
            return CaptureResult(mode=mode, width=0, height=0)
        def click(self, **kwargs): return ActionResult(True, "click")
        def drag(self, **kwargs): return ActionResult(True, "drag")
        def scroll(self, **kwargs): return ActionResult(True, "scroll")
        def type_text(self, text, **kwargs): return ActionResult(True, "type")
        def key(self, keys, **kwargs): return ActionResult(True, "key")
        def list_apps(self): return []
        def list_windows(self): return [{"pid": 7, "window_id": 8}]
        def focus_app(self, app, raise_window=False): return ActionResult(True, "focus_app")
        def set_value(self, value, element=None): return ActionResult(True, "set_value")

    backend = LegacyBackend()
    assert backend.list_windows() == [{"pid": 7, "window_id": 8}]
    assert backend.list_windows_exact() == {
        "windows": [{"pid": 7, "window_id": 8}],
        "count": 1,
        "on_screen_only": True,
        "exact_metadata": False,
        "compatibility_fallback": True,
    }
    with pytest.raises(ComputerUseCapabilityError) as launch_exc:
        backend.launch_app(name="Notes")
    assert launch_exc.value.code == "launch_app_unsupported"
    with pytest.raises(ComputerUseCapabilityError) as frame_exc:
        backend.set_window_frame(
            pid=1, window_id=2,
            frame={"x": 0, "y": 0, "width": 1, "height": 1},
        )
    assert frame_exc.value.code == "set_window_frame_unsupported"
