#!/usr/bin/env python3
"""Consolidated desktop preview-pane tool."""

import json
from typing import Callable, Optional

from tools import desktop_ui
from tools.open_preview_tool import _normalize_target, open_preview_tool
from tools.read_preview_tool import read_preview_tool
from tools.registry import registry, tool_error


def dispatch_preview(
    args: dict,
    *,
    callback: Optional[Callable] = None,
) -> str:
    action = str(args.get("action") or "").strip()
    allowed = {
        "open": {"url", "label"},
        "close": {"url"},
        "read": {"start", "count"},
    }.get(action)
    if allowed is not None:
        unexpected = sorted(
            key
            for key, value in args.items()
            if key != "action" and value is not None and key not in allowed
        )
        if unexpected:
            return tool_error(
                f"{action} does not accept: {', '.join(unexpected)}",
                code="unexpected_action_arguments",
                unexpected=unexpected,
            )
    if action == "open":
        return open_preview_tool(
            url=args.get("url", ""),
            label=args.get("label", ""),
        )
    if action == "close":
        target = _normalize_target(args.get("url") or "")
        try:
            ok = desktop_ui.emit("preview.close", {"url": target})
        except Exception as exc:
            return tool_error(f"Failed to close the preview: {exc}")
        if not ok:
            return tool_error(
                "The preview pane is only available in the Hermes desktop app."
            )
        return json.dumps(
            {"success": True, "closed": target or "all"},
            ensure_ascii=False,
        )
    if action == "read":
        return read_preview_tool(
            start=args.get("start"),
            count=args.get("count"),
            callback=callback,
        )
    return tool_error("action must be one of: open, close, read.")


PREVIEW_SCHEMA = {
    "name": "desktop_preview",
    "description": (
        "The preview pane beside the chat in the Hermes desktop app. open: "
        "show a web URL, localhost server, or file path (HTML renders live). "
        "close: dismiss the pane or one tab via url. read: return what the "
        "pane currently shows as {kind, url, title, text, start, end, "
        "total_chars}; Browser tabs and a rendered HTML file return rendered "
        "page text, paged with start/count character offsets. Other file tabs "
        "return identity so their source can be read with read_file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["open", "close", "read"]},
            "url": {
                "type": "string",
                "description": "open: target; close: one tab (omit for whole pane).",
            },
            "label": {"type": "string", "description": "open: optional tab label."},
            "start": {"type": "integer", "description": "read: 0-indexed char offset."},
            "count": {"type": "integer", "description": "read: chars to return (capped)."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


registry.register(
    name="desktop_preview",
    toolset="desktop_ui",
    schema=PREVIEW_SCHEMA,
    handler=lambda args, **kw: dispatch_preview(args, callback=kw.get("callback")),
    emoji="🖼️",
)
