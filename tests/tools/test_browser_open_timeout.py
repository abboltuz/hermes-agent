"""Tests for browser first-open timeout and timeout diagnostics."""

import json
import os
import subprocess
from unittest.mock import Mock, patch

import pytest

import tools.browser_tool as bt


_MISSING_ERROR = object()
_FORGED_RESULT_FIELDS = sorted(
    bt._BROWSER_RESULT_RESERVED_FIELDS
    | {"fallback_future_attestation", "browser_engine_future_attestation"}
)


def _run_lightpanda_process_result(
    monkeypatch,
    tmp_path,
    *,
    command,
    args,
    stdout_text,
    returncode=0,
    engine="lightpanda",
):
    task_id = f"lightpanda-protocol-{command}"
    bt._active_sessions[task_id] = {
        "session_name": f"lightpanda-{command}",
        "bb_session_id": None,
        "cdp_url": None,
    }
    process = Mock(returncode=returncode)
    process.wait.return_value = returncode
    chrome_fallback = Mock(return_value={"success": True, "data": {}})
    screenshot_fallback = Mock(return_value={"success": True, "data": {}})

    def popen(*_args, **kwargs):
        output = stdout_text if isinstance(stdout_text, bytes) else stdout_text.encode("utf-8")
        os.write(kwargs["stdout"], output)
        return process

    monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
    monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
    monkeypatch.setattr(bt, "_get_browser_engine", lambda: engine)
    monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
    monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
    monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
    monkeypatch.setattr(bt, "_run_chrome_fallback_command", chrome_fallback)
    monkeypatch.setattr(bt, "_chrome_fallback_screenshot", screenshot_fallback)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    result = bt._run_browser_command(task_id, command, args, timeout=1)
    return result, chrome_fallback, screenshot_fallback


def _run_image_extraction_result(monkeypatch, src, *, task_key="task::local"):
    images = [{
        "src": src,
        "alt": "avatar",
        "width": 640,
        "height": 480,
    }]
    monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(bt, "_last_session_key", lambda _task: task_key)
    monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda _task: False)
    monkeypatch.setattr(
        bt,
        "_run_browser_command",
        lambda *_args, **_kwargs: {
            "success": True,
            "data": {"result": images},
        },
    )
    return json.loads(bt.browser_get_images(task_id="task"))


@pytest.fixture(autouse=True)
def _reset_browser_caches():
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    bt._active_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()
    bt._invalidated_session_bindings.clear()
    yield
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    bt._active_sessions.clear()
    bt._session_last_activity.clear()
    bt._last_active_session_key.clear()
    bt._invalidated_session_bindings.clear()


class TestOpenCommandTimeout:
    def test_first_open_uses_longer_floor(self, monkeypatch):
        monkeypatch.setattr(bt, "_get_command_timeout", lambda: 30)
        assert bt._get_open_command_timeout(first_open=True) == bt.MIN_FIRST_OPEN_TIMEOUT
        assert bt._get_open_command_timeout(first_open=False) == bt.MIN_OPEN_TIMEOUT

    def test_respects_config_above_floor(self, monkeypatch):
        monkeypatch.setattr(bt, "_get_command_timeout", lambda: 180)
        assert bt._get_open_command_timeout(first_open=True) == 180
        assert bt._get_open_command_timeout(first_open=False) == 180


class TestSandboxBypass:
    def test_docker_triggers_bypass(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: True)
        assert bt._needs_chromium_sandbox_bypass() is True

    def test_apparmor_userns_triggers_bypass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        sysctl = tmp_path / "apparmor_restrict_unprivileged_userns"
        sysctl.write_text("1\n", encoding="utf-8")

        import builtins

        real_open = builtins.open

        def _open(path, *args, **kwargs):
            if "apparmor_restrict_unprivileged_userns" in str(path):
                return real_open(sysctl, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _open)
        assert bt._needs_chromium_sandbox_bypass() is True


class TestTimeoutErrorFormatting:
    def test_includes_stderr_detail(self):
        err = bt._format_browser_timeout_error(
            "open",
            120,
            "",
            "Daemon process exited during startup",
        )
        assert "120 seconds" in err
        assert "Daemon process exited" in err


    def test_local_install_hint(self, monkeypatch):
        monkeypatch.setattr(bt, "_is_local_mode", lambda: True)
        monkeypatch.setattr(bt, "_running_in_docker", lambda: False)
        err = bt._format_browser_timeout_error("open", 60, "", "")
        assert "agent-browser install --with-deps" in err

    @pytest.mark.parametrize("stream", ["stdout", "stderr"])
    @pytest.mark.parametrize(
        "private_text",
        [
            "Authorization: Bearer sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
            "Cookie: session_id=private-cookie-value",
            "password=private-password-value",
            "private form text: patient record 123-45-6789",
        ],
    )
    def test_captured_output_is_never_copied_to_timeout_error(
        self, stream, private_text
    ):
        output = f"Daemon process exited during startup\n{private_text}"
        err = bt._format_browser_timeout_error(
            "open",
            60,
            output if stream == "stdout" else "",
            output if stream == "stderr" else "",
        )

        assert "Daemon process exited during startup." in err
        assert private_text not in err


class TestReadCommandOutputFiles:
    def test_reads_stdout_and_stderr(self, tmp_path):
        stdout_path = tmp_path / "out"
        stderr_path = tmp_path / "err"
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("warn", encoding="utf-8")
        stdout, stderr = bt._read_command_output_files(str(stdout_path), str(stderr_path))
        assert stdout == "ok"
        assert stderr == "warn"


class TestCommandTimeoutRecovery:
    @pytest.mark.parametrize("cloud", [False, True])
    def test_timeout_replaces_only_stuck_client(self, monkeypatch, tmp_path, cloud):
        task_id = "stuck-command"
        session_info = {
            "session_name": "stuck-session",
            "bb_session_id": "cloud-session-1" if cloud else None,
            "cdp_url": "ws://cloud.invalid/devtools/browser/1" if cloud else None,
        }
        bt._active_sessions[task_id] = session_info
        bt._session_last_activity[task_id] = 1.0
        bt._last_active_session_key[task_id] = task_id

        process = Mock()
        process.returncode = 0
        process.wait.side_effect = [subprocess.TimeoutExpired("agent-browser", 1), -9, 0]
        supervisor_events = []

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_ensure_cdp_supervisor", lambda _: supervisor_events.append("ensure"))
        monkeypatch.setattr(bt, "_stop_cdp_supervisor", lambda _: supervisor_events.append("stop"))
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        bt._run_browser_command(task_id, "click", ["@e1"], timeout=1)

        assert task_id not in bt._last_active_session_key
        assert task_id not in bt._invalidated_session_bindings
        assert not (tmp_path / "agent-browser-stuck-session").exists()
        if not cloud:
            assert task_id not in bt._active_sessions and task_id not in bt._session_last_activity
            return

        replacement = bt._active_sessions[task_id]
        assert replacement is not session_info
        assert replacement["session_name"] != "stuck-session"
        assert replacement["bb_session_id"] == "cloud-session-1"
        assert bt._get_session_info(task_id) is replacement

        provider = Mock()
        monkeypatch.setattr(bt, "_get_cloud_provider", lambda: provider)
        bt.cleanup_browser(task_id)
        provider.close_session.assert_called_once_with("cloud-session-1")
        assert supervisor_events == ["ensure", "stop", "stop"]

    def test_stale_timeout_cannot_remove_concurrent_replacement(self, tmp_path):
        stale, replacement = {"session_name": "stale"}, {"session_name": "replacement"}
        bt._active_sessions["race"] = replacement

        bt._discard_timed_out_browser_session("race", stale, str(tmp_path))

        assert bt._active_sessions["race"] is replacement
        assert tmp_path.exists()

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("open", ["https://example.com"]),
            ("click", ["@e1"]),
            ("fill", ["@e1", "value"]),
            ("scroll", ["down", "500"]),
            ("back", []),
            ("press", ["Enter"]),
            ("eval", ["document.title"]),
            ("snapshot", ["-c"]),
            ("screenshot", []),
            ("console", []),
            ("errors", []),
        ],
    )
    def test_lightpanda_timeout_unknown_is_never_replayed(
        self, monkeypatch, tmp_path, command, args
    ):
        task_id = "lightpanda-timeout"
        bt._active_sessions[task_id] = {
            "session_name": "lightpanda-session",
            "bb_session_id": None,
            "cdp_url": None,
        }
        process = Mock(returncode=0)
        process.wait.side_effect = [subprocess.TimeoutExpired("agent-browser", 1), -9]
        chrome_fallback = Mock(side_effect=AssertionError("must not replay timeout"))
        screenshot_fallback = Mock(side_effect=AssertionError("must not replay timeout"))

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
        monkeypatch.setattr(bt, "_get_browser_engine", lambda: "lightpanda")
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_stop_cdp_supervisor", lambda _: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(bt, "_run_chrome_fallback_command", chrome_fallback)
        monkeypatch.setattr(bt, "_chrome_fallback_screenshot", screenshot_fallback)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = bt._run_browser_command(task_id, command, args, timeout=1)

        assert result["success"] is False
        assert result["error_code"] == "timeout_outcome_unknown"
        assert result["outcome"] == "unknown"
        assert result["retry_safe"] is False
        assert result["recovery"] == "inspect_or_renavigate"
        assert "Do not repeat" in result["error"]
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    def test_lightpanda_transport_unknown_is_never_replayed(
        self, monkeypatch, tmp_path
    ):
        task_id = "lightpanda-transport"
        bt._active_sessions[task_id] = {
            "session_name": "lightpanda-session",
            "bb_session_id": None,
            "cdp_url": None,
        }
        process = Mock(returncode=0)
        process.wait.side_effect = OSError("transport lost after dispatch")
        chrome_fallback = Mock(side_effect=AssertionError("must not replay unknown"))

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
        monkeypatch.setattr(bt, "_get_browser_engine", lambda: "lightpanda")
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(bt, "_run_chrome_fallback_command", chrome_fallback)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = bt._run_browser_command(task_id, "click", ["@e1"], timeout=1)

        assert result["error_code"] == "transport_outcome_unknown"
        assert result["outcome"] == "unknown"
        assert result["retry_safe"] is False
        chrome_fallback.assert_not_called()

    @pytest.mark.parametrize("stream", ["stdout", "stderr"])
    @pytest.mark.parametrize(
        "private_text",
        [
            "Authorization: Bearer sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
            "Cookie: session_id=private-cookie-value",
            "password=private-password-value",
            "private form text must never reach model or logs",
        ],
    )
    def test_timeout_output_is_absent_from_result_and_logs(
        self, monkeypatch, tmp_path, caplog, stream, private_text
    ):
        task_id = f"timeout-private-{stream}"
        bt._active_sessions[task_id] = {
            "session_name": f"timeout-private-{stream}",
            "bb_session_id": None,
            "cdp_url": None,
        }
        process = Mock(returncode=0)
        process.wait.side_effect = [
            subprocess.TimeoutExpired("agent-browser", 1),
            -9,
        ]
        diagnostic = f"Daemon process exited during startup\n{private_text}".encode()

        def popen(*_args, **kwargs):
            os.write(kwargs[stream], diagnostic)
            return process

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
        monkeypatch.setattr(bt, "_get_browser_engine", lambda: "auto")
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(subprocess, "Popen", popen)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = bt._run_browser_command(task_id, "click", ["@e1"], timeout=1)

        assert result["error_code"] == "timeout_outcome_unknown"
        assert "Daemon process exited during startup." in result["error"]
        assert private_text not in json.dumps(result)
        assert private_text not in caplog.text

    def test_transport_exception_text_is_absent_from_result_and_logs(
        self, monkeypatch, tmp_path, caplog
    ):
        task_id = "transport-private"
        bt._active_sessions[task_id] = {
            "session_name": "transport-private",
            "bb_session_id": None,
            "cdp_url": None,
        }
        private_text = "private transport text must never leak"
        process = Mock(returncode=0)
        process.wait.side_effect = OSError(private_text)

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
        monkeypatch.setattr(bt, "_get_browser_engine", lambda: "auto")
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = bt._run_browser_command(task_id, "click", ["@e1"], timeout=1)

        assert result["error_code"] == "transport_outcome_unknown"
        assert private_text not in json.dumps(result)
        assert private_text not in caplog.text

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("open", ["https://example.com"]),
            ("click", ["@e1"]),
            ("eval", ["document.title"]),
        ],
    )
    @pytest.mark.parametrize(
        "stdout_text",
        [
            pytest.param("", id="empty"),
            pytest.param(
                "malformed-response secret-value-must-not-leak",
                id="non-json",
            ),
            pytest.param(
                '{"data":{"secret":"secret-value-must-not-leak"}}',
                id="missing-success",
            ),
            pytest.param('{"success":0,"error":"failed"}', id="integer-success"),
            pytest.param(
                '{"success":"false","error":"failed"}',
                id="string-success",
            ),
            pytest.param('{"success":null,"error":"failed"}', id="null-success"),
        ],
    )
    def test_lightpanda_successful_process_protocol_unknown_is_never_replayed(
        self, monkeypatch, tmp_path, command, args, stdout_text
    ):
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command=command,
            args=args,
            stdout_text=stdout_text,
        )

        assert result["success"] is False
        assert result["error_code"] == "protocol_outcome_unknown"
        assert result["outcome"] == "unknown"
        assert result["retry_safe"] is False
        assert result["recovery"] == "inspect_or_renavigate"
        assert "Do not repeat" in result["error"]
        assert "secret-value-must-not-leak" not in json.dumps(result)
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("open", ["https://example.com"]),
            ("snapshot", ["-c"]),
            ("screenshot", []),
            ("eval", ["document.title"]),
            ("console", []),
            ("errors", []),
            ("back", []),
            ("record", ["stop"]),
        ],
    )
    @pytest.mark.parametrize(
        "data_value",
        [
            ["secret-value-must-not-leak"],
            "secret-value-must-not-leak",
            42,
            True,
        ],
        ids=["list", "string", "number", "bool"],
    )
    def test_success_envelope_rejects_non_object_data_without_replay(
        self, monkeypatch, tmp_path, caplog, command, args, data_value
    ):
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command=command,
            args=args,
            stdout_text=json.dumps({"success": True, "data": data_value}),
        )

        assert result["success"] is False
        assert result["error_code"] == "protocol_outcome_unknown"
        assert result["outcome"] == "unknown"
        assert result["retry_safe"] is False
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize(
        ("command", "data"),
        [
            ("open", {"title": ["secret-value-must-not-leak"]}),
            ("open", {"url": 42}),
            ("snapshot", {"snapshot": ["secret-value-must-not-leak"]}),
            ("snapshot", {"refs": ["secret-value-must-not-leak"]}),
            ("screenshot", {"path": ["secret-value-must-not-leak"]}),
            ("screenshot", {"path": "bad\x00secret-value-must-not-leak"}),
            ("screenshot", {"annotations": {"secret": "secret-value-must-not-leak"}}),
            ("back", {"url": ["secret-value-must-not-leak"]}),
            ("record", {"path": ["secret-value-must-not-leak"]}),
            ("console", {"messages": {"secret": "secret-value-must-not-leak"}}),
            ("console", {"messages": [["secret-value-must-not-leak"]]}),
            ("console", {"messages": [{"text": {"secret": "secret-value-must-not-leak"}}]}),
            ("errors", {"errors": {"secret": "secret-value-must-not-leak"}}),
            ("errors", {"errors": [["secret-value-must-not-leak"]]}),
            ("errors", {"errors": [{"message": {"secret": "secret-value-must-not-leak"}}]}),
        ],
    )
    def test_success_envelope_rejects_invalid_consumed_member_types(
        self, monkeypatch, tmp_path, caplog, command, data
    ):
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command=command,
            args=[],
            stdout_text=json.dumps({"success": True, "data": data}),
        )

        assert result["success"] is False
        assert result["error_code"] == "protocol_outcome_unknown"
        assert result["retry_safe"] is False
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize(
        "data_value",
        [_MISSING_ERROR, None],
        ids=["missing", "null"],
    )
    def test_success_envelope_normalizes_absent_or_null_data(
        self, monkeypatch, tmp_path, data_value
    ):
        envelope = {"success": True}
        if data_value is not _MISSING_ERROR:
            envelope["data"] = data_value

        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command="click",
            args=["@e1"],
            stdout_text=json.dumps(envelope),
            engine="auto",
        )

        assert result == {"success": True, "data": {}}
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize(
        ("command", "data"),
        [
            ("open", {"title": "Example", "url": "https://example.com"}),
            ("snapshot", {"snapshot": "button Example", "refs": {"e1": {}}}),
            ("screenshot", {"path": "/tmp/example.png", "annotations": []}),
            ("eval", {"result": {"arbitrary": [1, True, None]}}),
            ("console", {"messages": [{"type": "log", "text": "ok"}]}),
            ("errors", {"errors": [{"message": "ok"}]}),
        ],
    )
    def test_success_envelope_preserves_valid_command_shapes(
        self, command, data
    ):
        result = bt._validated_browser_command_result(
            {"success": True, "data": data},
            command,
            0,
        )

        assert result == {"success": True, "data": data}

    def test_eval_result_remains_arbitrary_but_url_accessor_is_typed(self):
        result = bt._validated_browser_command_result(
            {"success": True, "data": {"result": {"arbitrary": [1, 2]}}},
            "eval",
            0,
        )

        assert result["data"]["result"] == {"arbitrary": [1, 2]}
        assert bt._browser_result_string(result, "result") == ""

    def test_image_extraction_rejects_non_list_eval_result(self, monkeypatch):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_last_session_key", lambda _task: "task")
        monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda _task: False)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *_args, **_kwargs: {
                "success": True,
                "data": {"result": {"not": "an image list"}},
            },
        )

        response = json.loads(bt.browser_get_images(task_id="task"))

        assert response == {
            "success": False,
            "error": "Browser image extraction returned an invalid result.",
        }

    @pytest.mark.parametrize(
        "annotations",
        [
            ["secret-value-must-not-leak"],
            [42],
            [["secret-value-must-not-leak"]],
            [{"id": "secret-value-must-not-leak", "label": "button"}],
            [{"id": 1, "label": {"secret": "secret-value-must-not-leak"}}],
            [{"id": True, "label": "secret-value-must-not-leak"}],
            [{"id": -1, "label": "secret-value-must-not-leak"}],
        ],
    )
    def test_screenshot_rejects_malformed_annotation_items_without_replay(
        self, monkeypatch, tmp_path, caplog, annotations
    ):
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command="screenshot",
            args=["--annotate"],
            stdout_text=json.dumps({
                "success": True,
                "data": {"path": "/tmp/example.png", "annotations": annotations},
            }),
        )

        assert result["success"] is False
        assert result["error_code"] == "protocol_outcome_unknown"
        assert result["retry_safe"] is False
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    def test_screenshot_annotation_is_minimized_redacted_and_bounded(self):
        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
        result = bt._validated_browser_command_result(
            {
                "success": True,
                "data": {
                    "path": "/tmp/example.png",
                    "annotations": [{
                        "id": 1,
                        "label": f"Submit {secret}" + "x" * 2_000,
                        "hostile": {"secret": "nested-secret-must-not-leak"},
                    }],
                },
            },
            "screenshot",
            0,
        )

        annotation = result["data"]["annotations"][0]
        assert set(annotation) == {"id", "label"}
        assert annotation["id"] == 1
        assert secret not in annotation["label"]
        assert len(annotation["label"]) <= bt._MAX_BROWSER_ANNOTATION_LABEL
        assert "nested-secret-must-not-leak" not in json.dumps(result)

    @pytest.mark.parametrize(
        "command,data",
        [
            ("console", {"messages": [{"type": "log", "text": {"secret": "secret-value-must-not-leak"}}]}),
            ("console", {"messages": [42]}),
            ("errors", {"errors": [{"message": ["secret-value-must-not-leak"]}]}),
            ("errors", {"errors": ["secret-value-must-not-leak"]}),
        ],
    )
    def test_console_collections_reject_malformed_items_without_raw_output(
        self, monkeypatch, tmp_path, caplog, command, data
    ):
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command=command,
            args=[],
            stdout_text=json.dumps({"success": True, "data": data}),
        )

        assert result["error_code"] == "protocol_outcome_unknown"
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    def test_console_collection_is_minimized_and_force_redacted(self):
        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
        result = bt._validated_browser_command_result(
            {
                "success": True,
                "data": {
                    "messages": [{
                        "type": "log",
                        "text": f"token {secret}",
                        "nested": {"secret": "nested-secret-must-not-leak"},
                    }],
                },
            },
            "console",
            0,
        )

        message = result["data"]["messages"][0]
        assert set(message) == {"type", "text"}
        assert secret not in message["text"]
        assert "nested-secret-must-not-leak" not in json.dumps(result)

    @pytest.mark.parametrize(
        "images",
        [
            ["secret-value-must-not-leak"],
            [42],
            [["secret-value-must-not-leak"]],
            [{"src": {"secret": "secret-value-must-not-leak"}, "alt": "", "width": 1, "height": 1}],
            [{"src": "https://example.com/a.png", "alt": {"secret": "secret-value-must-not-leak"}, "width": 1, "height": 1}],
            [{"src": "https://example.com/a.png", "alt": "", "width": True, "height": 1}],
            [{"src": "https://example.com/a.png", "alt": "", "width": 1, "height": -1}],
        ],
    )
    def test_image_extraction_rejects_malformed_items_without_leak(
        self, monkeypatch, caplog, images
    ):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_last_session_key", lambda _task: "task::local")
        monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda _task: False)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *_args, **_kwargs: {
                "success": True,
                "data": {"result": images},
            },
        )

        response = json.loads(bt.browser_get_images(task_id="task"))

        assert response == {
            "success": False,
            "error": "Browser image extraction returned an invalid result.",
        }
        assert "secret-value-must-not-leak" not in json.dumps(response)
        assert "secret-value-must-not-leak" not in caplog.text

    def test_image_extraction_minimizes_and_redacts_valid_items(self, monkeypatch):
        secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
        images = [{
            "src": "https://example.com/a.png",
            "alt": f"profile {secret}",
            "width": 640,
            "height": 480,
            "nested": {"secret": "nested-secret-must-not-leak"},
        }]
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_last_session_key", lambda _task: "task::local")
        monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda _task: False)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *_args, **_kwargs: {
                "success": True,
                "data": {"result": images},
            },
        )

        response = json.loads(bt.browser_get_images(task_id="task"))

        assert response["success"] is True
        assert set(response["images"][0]) == {"src", "alt", "width", "height"}
        assert secret not in response["images"][0]["alt"]
        assert "nested-secret-must-not-leak" not in json.dumps(response)

    def test_image_extraction_rejects_credential_url(self, monkeypatch):
        images = [{
            "src": "https://example.com/a.png?access_token=private-token",
            "alt": "avatar",
            "width": 1,
            "height": 1,
        }]
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_last_session_key", lambda _task: "task::local")
        monkeypatch.setattr(bt, "_eval_ssrf_guard_active", lambda _task: False)
        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *_args, **_kwargs: {
                "success": True,
                "data": {"result": images},
            },
        )

        response = json.loads(bt.browser_get_images(task_id="task"))

        assert response["success"] is False
        assert "private-token" not in json.dumps(response)

    @pytest.mark.parametrize(
        "src",
        [
            "https://user:private-password@example.com/a.png",
            "https://user@example.com/a.png",
            "https://%75ser:private-password@example.com/a.png",
        ],
    )
    def test_image_url_rejects_userinfo_before_safety_checks(
        self, monkeypatch, caplog, src
    ):
        always_blocked = Mock(side_effect=AssertionError("must reject before safety"))
        network_safe = Mock(side_effect=AssertionError("must reject before safety"))
        monkeypatch.setattr(bt, "_is_always_blocked_url", always_blocked)
        monkeypatch.setattr(bt, "_is_safe_url", network_safe)

        response = _run_image_extraction_result(monkeypatch, src)

        assert response["success"] is False
        assert "private-password" not in json.dumps(response)
        assert "private-password" not in caplog.text
        always_blocked.assert_not_called()
        network_safe.assert_not_called()

    @pytest.mark.parametrize(
        "query",
        [
            "ACCESS_TOKEN=private-query-value",
            "%61ccess_%74oken=private-query-value",
            "%2561ccess_token=private-query-value",
            "ApiKey=private-query-value",
        ],
    )
    def test_image_url_rejects_encoded_or_case_varied_credential_query(
        self, monkeypatch, caplog, query
    ):
        network_safe = Mock(side_effect=AssertionError("must reject before DNS safety"))
        monkeypatch.setattr(bt, "_is_safe_url", network_safe)
        src = f"https://example.com/a.png?{query}"

        response = _run_image_extraction_result(monkeypatch, src)

        assert response["success"] is False
        assert "private-query-value" not in json.dumps(response)
        assert "private-query-value" not in caplog.text
        network_safe.assert_not_called()

    @pytest.mark.parametrize(
        "fragment",
        [
            "access_token=private-fragment-value",
            "%61ccess_%74oken=private-fragment-value",
            "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
        ],
    )
    def test_image_url_strips_fragment_before_publication(
        self, monkeypatch, caplog, fragment
    ):
        src = f"https://example.com/a.png?size=large#{fragment}"

        response = _run_image_extraction_result(monkeypatch, src)

        assert response["success"] is True
        assert response["images"][0]["src"] == (
            "https://example.com/a.png?size=large"
        )
        assert "private-fragment-value" not in json.dumps(response)
        assert "private-fragment-value" not in caplog.text

    @pytest.mark.parametrize(
        "src",
        [
            "https://[broken/private-secret?access_token=private-query-value",
            "https://example.com:invalid/private-secret?token=private-query-value",
            "https://example.com/a.png\x00private-secret",
        ],
    )
    def test_malformed_secret_image_url_fails_without_logs_or_safety_call(
        self, monkeypatch, caplog, src
    ):
        network_safe = Mock(side_effect=AssertionError("must reject before safety"))
        monkeypatch.setattr(bt, "_is_safe_url", network_safe)

        response = _run_image_extraction_result(monkeypatch, src)

        assert response["success"] is False
        assert "private-secret" not in json.dumps(response)
        assert "private-query-value" not in json.dumps(response)
        assert "private-secret" not in caplog.text
        assert "private-query-value" not in caplog.text
        network_safe.assert_not_called()

    def test_network_safety_receives_origin_only_and_exception_is_sanitized(
        self, monkeypatch, caplog
    ):
        seen_urls = []
        private_value = "private-benign-query-value"

        def fail_safety(url):
            seen_urls.append(url)
            raise RuntimeError(f"checker failed near {private_value}")

        monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
        monkeypatch.setattr(bt, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(bt, "_is_always_blocked_url", lambda _url: False)
        monkeypatch.setattr(bt, "_is_safe_url", fail_safety)
        src = f"https://example.com/private-path.png?theme={private_value}#private-fragment"

        response = _run_image_extraction_result(
            monkeypatch,
            src,
            task_key="task",
        )

        assert response["success"] is False
        assert seen_urls == ["https://example.com/"]
        assert private_value not in json.dumps(response)
        assert private_value not in caplog.text
        assert "private-path" not in caplog.text

    def test_valid_image_url_is_preserved_except_fragment(self, monkeypatch):
        src = "https://cdn.example.com/images/a%20b.png?size=large&theme=dark#section"

        response = _run_image_extraction_result(monkeypatch, src)

        assert response["success"] is True
        assert response["images"][0]["src"] == (
            "https://cdn.example.com/images/a%20b.png?size=large&theme=dark"
        )

    @pytest.mark.parametrize(
        "entrypoint,confirmed_results,expected_calls",
        [
            (
                lambda: bt.browser_snapshot(task_id="task"),
                [{"success": True, "data": {"snapshot": "private", "refs": {}}}],
                2,
            ),
            (
                lambda: bt.browser_get_images(task_id="task"),
                [{"success": True, "data": {"result": "[]"}}],
                2,
            ),
            (
                lambda: bt.browser_vision("inspect", task_id="task"),
                [],
                1,
            ),
            (
                lambda: bt.browser_console(task_id="task"),
                [],
                1,
            ),
            (
                lambda: bt.browser_back(task_id="task"),
                [{"success": True, "data": {"url": "https://example.com"}}],
                2,
            ),
            (
                lambda: bt.browser_click("e1", task_id="task"),
                [],
                1,
            ),
        ],
        ids=["snapshot", "images", "vision", "console", "back", "click"],
    )
    def test_content_and_action_siblings_fail_closed_on_unknown_safety_probe(
        self, monkeypatch, entrypoint, confirmed_results, expected_calls
    ):
        unknown = bt._unknown_browser_command_result(
            "private probe output",
            "timeout_outcome_unknown",
        )
        run_command = Mock(side_effect=[*confirmed_results, unknown])
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_last_session_key", lambda _task: "task")
        monkeypatch.setattr(bt, "_is_local_backend", lambda: False)
        monkeypatch.setattr(bt, "_is_local_sidecar_key", lambda _task: False)
        monkeypatch.setattr(bt, "_allow_private_urls", lambda: False)
        monkeypatch.setattr(bt, "_run_browser_command", run_command)

        raw_response = entrypoint()
        response = (
            raw_response
            if isinstance(raw_response, dict)
            else json.loads(raw_response)
        )

        assert response["success"] is False
        assert response["error_code"] == "timeout_outcome_unknown"
        assert response["outcome"] == "unknown"
        assert response["retry_safe"] is False
        assert "private probe output" not in json.dumps(response)
        assert run_command.call_count == expected_calls

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("open", ["https://example.com"]),
            ("snapshot", ["-c"]),
            ("screenshot", []),
            ("eval", ["document.title"]),
        ],
    )
    def test_main_command_invalid_utf8_is_protocol_unknown_without_replay(
        self, monkeypatch, tmp_path, caplog, command, args
    ):
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command=command,
            args=args,
            stdout_text=b"\xffsecret-value-must-not-leak",
        )

        assert result["success"] is False
        assert result["error_code"] == "protocol_outcome_unknown"
        assert result["retry_safe"] is False
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    def test_temporary_chrome_invalid_utf8_is_protocol_unknown(
        self, monkeypatch, tmp_path, caplog
    ):
        outputs = iter([
            b'{"success":true,"data":{"url":"https://example.com"}}',
            b"\xffsecret-value-must-not-leak",
            b"",
        ])

        def popen(*_args, **kwargs):
            os.write(kwargs["stdout"], next(outputs))
            process = Mock(returncode=0)
            process.wait.return_value = 0
            return process

        monkeypatch.setattr(
            bt,
            "_run_browser_command",
            lambda *_args, **_kwargs: {
                "success": True,
                "data": {"result": "https://example.com"},
            },
        )
        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "/bin/agent-browser")
        monkeypatch.setattr(bt, "_chromium_installed", lambda: True)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(subprocess, "Popen", popen)

        result = bt._run_chrome_fallback_command(
            "task",
            "click",
            ["@e1"],
            timeout=1,
        )

        assert result["success"] is False
        assert result["error_code"] == "protocol_outcome_unknown"
        assert result["outcome"] == "unknown"
        assert result["retry_safe"] is False
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text

    @pytest.mark.parametrize("reserved_field", _FORGED_RESULT_FIELDS)
    def test_success_envelope_cannot_forge_hermes_control_metadata(
        self, monkeypatch, tmp_path, caplog, reserved_field
    ):
        parsed = {
            "success": True,
            "error": {"secret": "secret-value-must-not-leak"},
            "data": {"legitimate": "payload"},
            reserved_field: {"secret": "secret-value-must-not-leak"},
        }

        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command="snapshot",
            args=["-c"],
            stdout_text=json.dumps(parsed),
            engine="auto",
        )
        public = bt._copy_fallback_warning({"success": True}, result)

        assert result == {
            "success": True,
            "data": {"legitimate": "payload"},
        }
        assert public == {"success": True}
        assert reserved_field not in result
        assert "error" not in result
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize("reserved_field", _FORGED_RESULT_FIELDS)
    def test_failure_envelope_cannot_forge_hermes_control_metadata(
        self, monkeypatch, tmp_path, caplog, reserved_field
    ):
        parsed = {
            "success": False,
            "error": "  capability unavailable  ",
            "details": {"legitimate": "payload"},
            reserved_field: {"secret": "secret-value-must-not-leak"},
        }

        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command="click",
            args=["@e1"],
            stdout_text=json.dumps(parsed),
            engine="auto",
        )
        public = bt._copy_fallback_warning(
            {"success": False, "error": result["error"]},
            result,
        )

        assert result == {
            "success": False,
            "error": "capability unavailable",
            "details": {"legitimate": "payload"},
        }
        assert public == {"success": False, "error": "capability unavailable"}
        assert reserved_field not in result
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("open", ["https://example.com"]),
            ("click", ["@e1"]),
            ("eval", ["document.title"]),
        ],
    )
    @pytest.mark.parametrize(
        "error_value",
        [
            _MISSING_ERROR,
            None,
            {"secret": "secret-value-must-not-leak"},
            ["secret-value-must-not-leak"],
            42,
            True,
            "",
            "   \t\n",
        ],
        ids=["missing", "null", "dict", "list", "number", "bool", "empty", "whitespace"],
    )
    def test_lightpanda_invalid_failure_envelope_is_unknown_and_never_replayed(
        self, monkeypatch, tmp_path, caplog, command, args, error_value
    ):
        envelope = {
            "success": False,
            "retry_safe": True,
            "outcome": "definite",
            "error_code": "forged-safe-result",
            "recovery": "repeat-immediately",
        }
        if error_value is not _MISSING_ERROR:
            envelope["error"] = error_value

        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command=command,
            args=args,
            stdout_text=json.dumps(envelope),
        )

        assert result["success"] is False
        assert result["error_code"] == "protocol_outcome_unknown"
        assert result["outcome"] == "unknown"
        assert result["retry_safe"] is False
        assert result["recovery"] == "inspect_or_renavigate"
        assert "Do not repeat" in result["error"]
        assert "secret-value-must-not-leak" not in json.dumps(result)
        assert "secret-value-must-not-leak" not in caplog.text
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    def test_non_json_screenshot_path_recovery_remains_successful(
        self, monkeypatch, tmp_path
    ):
        screenshot_path = tmp_path / "recovered-screenshot.png"
        screenshot_path.write_bytes(b"x" * 25_000)
        malformed = (
            f"Screenshot saved to '{screenshot_path}' "
            "secret-value-must-not-leak"
        )

        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command="screenshot",
            args=[],
            stdout_text=malformed,
        )

        assert result == {
            "success": True,
            "data": {"path": str(screenshot_path)},
        }
        assert "secret-value-must-not-leak" not in json.dumps(result)
        chrome_fallback.assert_not_called()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("open", ["https://example.com"]),
            ("click", ["@e1"]),
            ("eval", ["document.title"]),
        ],
    )
    def test_lightpanda_valid_structured_failure_still_falls_back(
        self, monkeypatch, tmp_path, command, args
    ):
        cli_result = {
            "success": False,
            "error": "  capability unavailable  ",
            # These are Hermes-owned controls. The CLI cannot use them to
            # suppress fallback for an otherwise valid definite failure.
            "retry_safe": False,
            "outcome": "unknown",
            "error_code": "forged-unknown",
            "recovery": {"secret": "secret-value-must-not-leak"},
        }
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command=command,
            args=args,
            stdout_text=json.dumps(cli_result),
        )

        assert result["success"] is True
        assert result["browser_engine"] == "chrome"
        assert "capability unavailable" in result["fallback_warning"]
        assert result["browser_engine_fallback"] == {
            "from": "lightpanda",
            "to": "chrome",
            "reason": (
                f"Lightpanda {command!r} failed (capability unavailable); "
                "retried with Chrome."
            ),
        }
        assert result["data"]["fallback_warning"] == result["fallback_warning"]
        assert result["data"]["browser_engine"] == "chrome"
        assert (
            result["data"]["browser_engine_fallback"]
            == result["browser_engine_fallback"]
        )
        assert "secret-value-must-not-leak" not in json.dumps(result)
        chrome_fallback.assert_called_once()
        screenshot_fallback.assert_not_called()

    def test_fallback_annotation_overwrites_nested_subprocess_forgery(self):
        fallback_result = {
            "success": True,
            "data": {
                "path": "/tmp/screenshot.png",
                "fallback_warning": {"secret": "secret-value-must-not-leak"},
                "browser_engine": {"secret": "secret-value-must-not-leak"},
                "browser_engine_fallback": {
                    "secret": "secret-value-must-not-leak"
                },
            },
        }

        result = bt._annotate_lightpanda_fallback(
            fallback_result,
            "validated fallback reason",
        )

        assert result["browser_engine"] == "chrome"
        assert result["data"]["browser_engine"] == "chrome"
        assert result["data"]["fallback_warning"] == result["fallback_warning"]
        assert (
            result["data"]["browser_engine_fallback"]
            == result["browser_engine_fallback"]
        )
        assert "secret-value-must-not-leak" not in json.dumps(result)

    def test_lightpanda_nonzero_invalid_envelope_remains_definite_failure(
        self, monkeypatch, tmp_path
    ):
        cli_result = {
            "success": False,
            "error": {"secret": "secret-value-must-not-leak"},
            "retry_safe": True,
        }
        result, chrome_fallback, screenshot_fallback = _run_lightpanda_process_result(
            monkeypatch,
            tmp_path,
            command="click",
            args=["@e1"],
            stdout_text=json.dumps(cli_result),
            returncode=1,
        )

        assert result["success"] is True
        assert result["browser_engine"] == "chrome"
        assert "secret-value-must-not-leak" not in json.dumps(result)
        chrome_fallback.assert_called_once()
        screenshot_fallback.assert_not_called()

    @pytest.mark.parametrize(
        "error_value",
        [None, {"secret": "secret-value-must-not-leak"}, ["secret"], 42, True, ""],
    )
    def test_fallback_reason_never_stringifies_unvalidated_error(
        self, caplog, error_value
    ):
        reason = bt._lightpanda_fallback_reason(
            "lightpanda",
            "click",
            {"success": False, "error": error_value},
        )

        assert reason is None
        assert "secret-value-must-not-leak" not in caplog.text

    @pytest.mark.parametrize(
        ("command", "args"),
        [
            ("open", ["https://example.com"]),
            ("click", ["@e1"]),
            ("snapshot", ["-c"]),
        ],
    )
    def test_lightpanda_definite_eligible_failure_still_falls_back(
        self, monkeypatch, tmp_path, command, args
    ):
        task_id = "lightpanda-definite-failure"
        bt._active_sessions[task_id] = {
            "session_name": "lightpanda-session",
            "bb_session_id": None,
            "cdp_url": None,
        }
        process = Mock(returncode=1)
        process.wait.return_value = 1
        chrome_fallback = Mock(return_value={"success": True, "data": {}})

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
        monkeypatch.setattr(bt, "_get_browser_engine", lambda: "lightpanda")
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(bt, "_run_chrome_fallback_command", chrome_fallback)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = bt._run_browser_command(task_id, command, args, timeout=1)

        assert result["success"] is True
        assert result["browser_engine"] == "chrome"
        chrome_fallback.assert_called_once_with(task_id, command, args, 1)

    def test_local_sidecar_timeout_blocks_non_navigation_until_rebind(
        self, monkeypatch, tmp_path
    ):
        task_id = "hybrid-task"
        sidecar_key = f"{task_id}::local"
        cloud_session = {
            "session_name": "cloud-session",
            "bb_session_id": "cloud-id",
            "cdp_url": "ws://cloud.invalid/devtools/browser/1",
        }
        local_session = {
            "session_name": "local-session",
            "bb_session_id": None,
            "cdp_url": None,
        }
        bt._active_sessions[task_id] = cloud_session
        bt._active_sessions[sidecar_key] = local_session
        bt._session_last_activity[sidecar_key] = 1.0
        bt._last_active_session_key[task_id] = sidecar_key

        process = Mock()
        process.returncode = 0
        process.wait.side_effect = [subprocess.TimeoutExpired("agent-browser", 1), -9]

        monkeypatch.setattr(bt, "_find_agent_browser", lambda: "agent-browser")
        monkeypatch.setattr(bt, "_requires_real_termux_browser_install", lambda _cmd: False)
        monkeypatch.setattr(bt, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(bt, "_stop_cdp_supervisor", lambda _: None)
        monkeypatch.setattr(bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
        monkeypatch.setattr(bt, "_write_owner_pid", lambda *_args: None)
        monkeypatch.setattr(bt, "_build_browser_env", lambda: {})
        monkeypatch.setattr(bt, "_merge_browser_path", lambda value: value)
        monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = bt._run_browser_command(sidecar_key, "click", ["@e1"], timeout=1)

        assert result["success"] is False
        assert bt._active_sessions[task_id] is cloud_session
        assert sidecar_key not in bt._active_sessions
        assert bt._last_active_session_key[task_id] == sidecar_key
        assert bt._invalidated_session_bindings[task_id] == sidecar_key

        forbidden_run = Mock(side_effect=AssertionError("must not target cloud session"))
        monkeypatch.setattr(bt, "_run_browser_command", forbidden_run)
        actions = [
            lambda: bt.browser_click("e1", task_id=task_id),
            lambda: bt.browser_type("e1", "value", task_id=task_id),
            lambda: bt.browser_snapshot(task_id=task_id),
            lambda: bt.browser_scroll("down", task_id=task_id),
            lambda: bt.browser_back(task_id=task_id),
            lambda: bt.browser_press("Enter", task_id=task_id),
            lambda: bt.browser_console(task_id=task_id),
            lambda: bt.browser_console(expression="document.title", task_id=task_id),
            lambda: bt.browser_get_images(task_id=task_id),
            lambda: bt.browser_vision("What is visible?", task_id=task_id),
        ]
        for action in actions:
            response = action()
            payload = response if isinstance(response, dict) else json.loads(response)
            assert payload["success"] is False
            assert "Navigate successfully" in payload["error"]
        forbidden_run.assert_not_called()

        replacement_sidecar = {
            "session_name": "replacement-local",
            "bb_session_id": None,
            "cdp_url": None,
            "_first_nav": False,
            "features": {"local": True},
        }
        calls = []

        def successful_run(session_key, command, args=None, timeout=None, **_kwargs):
            calls.append((session_key, command))
            if command == "open":
                return {
                    "success": True,
                    "data": {"title": "Local", "url": args[0]},
                }
            if command == "snapshot":
                return {"success": True, "data": {"snapshot": "", "refs": {}}}
            return {"success": True, "data": {}}

        def get_replacement(session_key):
            bt._active_sessions[session_key] = replacement_sidecar
            return replacement_sidecar

        monkeypatch.setattr(bt, "_navigation_session_key", lambda _task, _url: sidecar_key)
        monkeypatch.setattr(bt, "_get_session_info", get_replacement)
        monkeypatch.setattr(bt, "_run_browser_command", successful_run)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: True)
        monkeypatch.setattr(bt, "check_website_access", lambda _url: None)

        navigation = json.loads(bt.browser_navigate("http://localhost:3000", task_id=task_id))
        assert navigation["success"] is True
        assert task_id not in bt._invalidated_session_bindings
        assert bt._last_active_session_key[task_id] == sidecar_key

        click = json.loads(bt.browser_click("e2", task_id=task_id))
        assert click["success"] is True
        assert calls[-1] == (sidecar_key, "click")
        assert all(session_key != task_id for session_key, _command in calls)


class TestBrowserNavigateOpenTimeout:
    def test_first_navigation_uses_first_open_timeout(self, monkeypatch):
        captured: dict = {}

        def fake_run(task_id, command, args, timeout=None):
            if command == "open":
                captured["timeout"] = timeout
            return {"success": True, "data": {"title": "t", "url": args[0] if args else ""}}

        monkeypatch.setattr(bt, "_get_open_command_timeout", lambda first_open=False: 120 if first_open else 60)
        monkeypatch.setattr(bt, "_run_browser_command", fake_run)
        monkeypatch.setattr(bt, "_get_session_info", lambda key: {"_first_nav": True, "features": {}})
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt, "_is_local_backend", lambda: True)
        monkeypatch.setattr(bt, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(bt, "_navigation_session_key", lambda task_id, url: task_id)
        monkeypatch.setattr(bt, "_maybe_start_recording", lambda *a, **kw: None)
        monkeypatch.setattr(bt, "check_website_access", lambda url: None)

        bt.browser_navigate("https://example.com", task_id="task-1")
        assert captured["timeout"] == 120
