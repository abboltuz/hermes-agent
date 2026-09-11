"""Schema for the generic `computer_use` tool.

Model-agnostic. Any tool-calling model can drive this. Vision-capable models
should prefer `capture(mode='som')` then `click(element=N)` — much more
reliable than pixel coordinates. Pixel coordinates remain supported for
models that were trained on them (e.g. Claude's computer-use RL).
"""

from __future__ import annotations

from typing import Any, Dict


# One consolidated tool with an `action` discriminator. Keeps the schema
# compact and the per-turn token cost low.
COMPUTER_USE_SCHEMA: Dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Drive the desktop in the background via cua-driver — screenshots, "
        "mouse, keyboard, scroll, drag — without stealing the user's cursor "
        "or keyboard focus. Supported on macOS, Windows, and Linux. "
        "Preferred workflow: call with "
        "action='capture' (mode='som' gives a screenshot plus a numbered "
        "accessibility-element list), "
        "then click by `element` index for reliability. Pixel coordinates "
        "are supported for models trained on them. Image captures include a "
        "shareable `screenshot_path`; when the user asks to receive the image "
        "and the current surface supports attachments, deliver that file using "
        "the platform's native MEDIA attachment syntax. Do not automatically "
        "send screenshots used only for computer control. Known targets can "
        "be driven in the background without claiming foreground focus; "
        "window discovery scope remains platform-dependent. Requires cua-driver to "
        "be installed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "capture",
                    "click",
                    "double_click",
                    "right_click",
                    "middle_click",
                    "drag",
                    "scroll",
                    "type",
                    "key",
                    "set_value",
                    "wait",
                    "list_apps",
                    "list_windows",
                    "verify_state",
                    "launch_app",
                    "set_window_frame",
                    "focus_app",
                ],
                "description": (
                    "Which action to perform. Capture, wait, app/window "
                    "listing, and verify_state are read-only. Mutating actions "
                    "require approval unless auto-approved. Use `set_value` for select/popup elements "
                    "and sliders — it selects the matching option directly "
                    "without opening the native menu (no focus steal)."
                ),
            },
            # ── capture ────────────────────────────────────────────
            "mode": {
                "type": "string",
                "enum": ["som", "vision", "ax"],
                "description": (
                    "Capture mode. `som` (default) is a plain screenshot plus "
                    "a separately numbered AX element list — best for vision "
                    "models and lets you click by element index. The numbers "
                    "are not drawn over the screenshot. `vision` is a plain screenshot. "
                    "`ax` is the accessibility tree only (no image; useful "
                    "for text-only models)."
                ),
            },
            "app": {
                "type": "string",
                "description": (
                    "For capture, select one unique visible window belonging "
                    "to this app (by name, e.g. 'Safari', or bundle ID, "
                    "'com.apple.Safari'); ambiguity returns candidates. For "
                    "focus_app this is the required app. On input actions app "
                    "is a safety assertion against the sticky target from the "
                    "last capture/focus_app, not a retargeting shortcut; an "
                    "unknown or different target is refused. If omitted from "
                    "capture, operates on the frontmost app's window. Pass "
                    "app='screen' to capture "
                    "everything currently displayed (a composited "
                    "full-screen grab; image only, no clickable elements). "
                    "Pass app='desktop' to target the OS desktop/shell "
                    "surface itself (wallpaper, desktop icons, taskbar) "
                    "with its clickable elements."
                ),
            },
            "pid": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Process target. Required with window_id for "
                    "action='verify_state'; optional for action='list_windows' "
                    "and action='capture'. For capture, if exactly "
                    "one visible window belongs to the process it is selected; "
                    "multiple windows fail closed with candidates. Pair with "
                    "window_id for an exact target."
                ),
            },
            "window_id": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Exact native window target for action='capture' or "
                    "action='verify_state'. Always pair with pid."
                ),
            },
            "bundle_id": {
                "type": "string",
                "description": (
                    "Only for action='launch_app'. Exact non-empty bundle "
                    "identifier. Use either bundle_id or app (as the native "
                    "application name), never both."
                ),
            },
            "creates_new_application_instance": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Only for action='launch_app'. Request a separate native "
                    "application instance. This never selects or foregrounds "
                    "a returned window."
                ),
            },
            "frame": {
                "type": "object",
                "additionalProperties": False,
                "description": (
                    "Only for action='set_window_frame'. Exact desktop "
                    "coordinates; unlike coordinate=[x,y], these are not "
                    "window-local screenshot pixels."
                ),
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number", "minimum": 1},
                    "height": {"type": "number", "minimum": 1},
                },
                "required": ["x", "y", "width", "height"],
            },
            "on_screen_only": {
                "type": "boolean",
                "default": True,
                "description": (
                    "Only for action='list_windows'. Defaults true. Set false "
                    "only together with pid to inspect that process's hidden, "
                    "minimized, or other-Space windows without leaking titles "
                    "from every process."
                ),
            },
            "expect": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "description": (
                    "Only for action='verify_state'. One to eight predicates, "
                    "ANDed, evaluated against the exact pid/window_id target."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "element": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "selector": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "label_contains": {"type": "string"},
                                        "role": {"type": "string"},
                                    },
                                    "minProperties": 1,
                                },
                                "exists": {"type": "boolean", "enum": [True]},
                                "enabled": {"type": ["boolean", "null"]},
                                "selected": {"type": ["boolean", "null"]},
                                "value_equals": {"type": ["string", "null"]},
                            },
                            "required": ["selector"],
                        },
                        "window": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "exists": {"type": ["boolean", "null"]},
                                "bounds": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                        "width": {"type": "number"},
                                        "height": {"type": "number"},
                                        "tolerance_px": {
                                            "type": "number",
                                            "minimum": 0,
                                            "maximum": 100,
                                        },
                                    },
                                    "required": ["x", "y", "width", "height"],
                                },
                            },
                            "minProperties": 1,
                        },
                    },
                    "oneOf": [
                        {"required": ["element"]},
                        {"required": ["window"]},
                    ],
                },
            },
            "timeout_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10000,
                "default": 5000,
                "description": "Only for verify_state; bounded wait in milliseconds.",
            },
            "stable_samples": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "default": 2,
                "description": "Only for verify_state; consecutive matching samples required.",
            },
            "include_screenshot": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Only for verify_state. Include fresh visual evidence; "
                    "defaults false to keep verification small."
                ),
            },
            "max_elements": {
                "type": "integer",
                "description": (
                    "Optional cap on the AX `elements` array returned by "
                    "`action='capture'`. Default 100, hard maximum 1000. "
                    "Dense UIs (Electron apps such as Obsidian or VS Code, "
                    "JetBrains IDEs) can publish 500+ AX nodes — capping "
                    "prevents a single capture from blowing session "
                    "context. When the cap trims the response, "
                    "`total_elements` and `truncated_elements` are "
                    "surfaced in the result so you can re-call with "
                    "`app=` to narrow scope or raise `max_elements` when "
                    "the full tree is required. When the live driver accepts "
                    "`max_elements`, Hermes also sends this bound to "
                    "`get_window_state` so dense trees are capped before "
                    "crossing the transport."
                ),
                "default": 100,
                "minimum": 1,
                "maximum": 1000,
            },
            # ── click / drag / scroll targeting ────────────────────
            "element": {
                "type": "integer",
                "description": (
                    "The 1-based SOM index returned by the last "
                    "`capture(mode='som')` call. Strongly preferred over "
                    "raw coordinates. For action='type', binds text to that "
                    "captured element when the live Cua schema supports it. "
                    "For action='key', targets that element when the live "
                    "press_key/hotkey schema supports element_index."
                ),
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Pixel coordinates [x, y] relative to the captured window "
                    "screenshot (top-left origin). Only use this if no element "
                    "index is available. For type/key this focuses the pixel "
                    "target before delivering input when the live driver "
                    "advertises coordinate support."
                ),
            },
            "button": {
                "type": "string",
                "enum": ["left", "right", "middle"],
                "description": "Mouse button. Defaults to left.",
            },
            "modifiers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "cmd", "shift", "option", "alt", "ctrl", "fn",
                        "win", "windows", "super", "meta",
                    ],
                },
                "description": (
                    "Modifier keys held during click, double_click, "
                    "right_click, middle_click, drag, or scroll. Live driver "
                    "schemas may refuse modifiers they do not support."
                ),
            },
            # ── drag ───────────────────────────────────────────────
            "from_element": {"type": "integer",
                              "description": (
                                  "Source element index for drag; accepted only "
                                  "when the live driver advertises element endpoints."
                              )},
            "to_element": {"type": "integer",
                            "description": (
                                "Target element index for drag; accepted only "
                                "when the live driver advertises element endpoints."
                            )},
            "from_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Source [x,y] for drag; the portable default.",
            },
            "to_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2, "maxItems": 2,
                "description": "Target [x,y] for drag; the portable default.",
            },
            # ── scroll ─────────────────────────────────────────────
            "direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction.",
            },
            "amount": {
                "type": "integer",
                "description": "Scroll wheel ticks. Default 3.",
            },
            # ── set_value ──────────────────────────────────────────
            "value": {
                "type": "string",
                "description": (
                    "For action='set_value': the value to set on the element. "
                    "For AXPopUpButton / select dropdowns, pass the option's "
                    "display label (e.g. 'Blue'). For sliders and other "
                    "AXValue-settable elements, pass the numeric or string value."
                ),
            },
            # ── type / key / wait ──────────────────────────────────
            "text": {
                "type": "string",
                "description": "Text to type (respects the current layout).",
            },
            "keys": {
                "type": "string",
                "description": (
                    "Key combo, e.g. 'cmd+s', 'ctrl+alt+t', 'return', "
                    "'escape', 'tab'. Use '+' to combine."
                ),
            },
            "seconds": {
                "type": "number",
                "description": "Seconds to wait. Max 30.",
            },
            # ── focus_app ──────────────────────────────────────────
            "raise_window": {
                "type": "boolean",
                "description": (
                    "Only for action='focus_app'. If true, brings the "
                    "window to front (DISRUPTS the user). Default false "
                    "— input is routed to the app without raising, "
                    "matching the background co-work model."
                ),
            },
            # ── delivery (verify → escalate ladder) ────────────────
            "delivery_mode": {
                "type": "string",
                "enum": ["background", "foreground"],
                "description": (
                    "How input is delivered, for the input actions (click, "
                    "double_click, right_click, drag, scroll, type, key). "
                    "`background` (DEFAULT) routes input to the target without "
                    "raising it or stealing focus — the co-work model. "
                    "`foreground` briefly fronts the window, acts, then "
                    "restores the prior frontmost app. A `confirmed` effect "
                    "still requires task-postcondition verification. For "
                    "`unverifiable`, inspect fresh state before any "
                    "retry even if escalation is recommended. Escalate only "
                    "after `suspected_noop` or a structured refusal. Do not "
                    "predict the rung from the app being Electron/Chromium. "
                    "Foreground is a visible focus change and needs its own "
                    "approval."
                ),
            },
            "bring_to_front": {
                "type": "boolean",
                "description": (
                    "Optional and only valid with delivery_mode='foreground'. "
                    "Explicitly invokes cua-driver's standalone bring_to_front "
                    "tool before the input; it is never passed as an input "
                    "property. This persistent focus change has a separate "
                    "approval scope. Default false."
                ),
            },
            # ── return shape ───────────────────────────────────────
            "capture_after": {
                "type": "boolean",
                "description": (
                    "For click, double_click, right_click, middle_click, drag, "
                    "scroll, type, key, set_value, or focus_app: if true, take "
                    "a follow-up capture after a successful action and include "
                    "it in the response."
                ),
            },
        },
        "required": ["action"],
    },
}


def get_computer_use_schema() -> Dict[str, Any]:
    """Return the generic OpenAI function-calling schema."""
    return COMPUTER_USE_SCHEMA
