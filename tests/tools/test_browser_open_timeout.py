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
        os.write(kwargs["stdout"], stdout_text.encode("utf-8"))
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
