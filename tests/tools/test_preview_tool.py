import json

from tools.registry import registry


def test_desktop_preview_consolidates_open_close_and_read_contracts(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)

    from tools.preview_tool import PREVIEW_SCHEMA

    entry = registry.get_entry("desktop_preview")
    assert entry is not None
    assert entry.toolset == "desktop_ui"
    assert registry.get_entry("open_preview") is None
    assert registry.get_entry("close_preview") is None
    assert registry.get_entry("read_preview") is None
    assert PREVIEW_SCHEMA["parameters"]["properties"]["action"]["enum"] == [
        "open", "close", "read",
    ]


def test_html_file_read_contract_promises_rendered_live_text():
    from tools.preview_tool import PREVIEW_SCHEMA

    description = PREVIEW_SCHEMA["description"]
    assert "HTML file" in description
    assert "rendered page text" in description
    assert "file tab answers identity only" not in description


def test_desktop_preview_read_uses_desktop_callback():
    from tools.preview_tool import dispatch_preview

    seen = {}

    def callback(start=None, count=None):
        seen.update({"start": start, "count": count})
        return json.dumps({"kind": "file", "text": "Rendered HTML"})

    result = json.loads(dispatch_preview(
        {"action": "read", "start": 4, "count": 12},
        callback=callback,
    ))

    assert seen == {"start": 4, "count": 12}
    assert result == {"kind": "file", "text": "Rendered HTML"}


def test_desktop_preview_rejects_fields_irrelevant_to_selected_action(monkeypatch):
    from tools.preview_tool import dispatch_preview

    opened = []
    monkeypatch.setattr(
        "tools.preview_tool.open_preview_tool",
        lambda **kwargs: opened.append(kwargs) or "{}",
    )

    result = json.loads(dispatch_preview({
        "action": "open", "url": "https://example.test", "count": 10,
    }))

    assert result["code"] == "unexpected_action_arguments"
    assert result["unexpected"] == ["count"]
    assert opened == []
