"""Regression tests for sudo detection and terminal execution contracts."""

import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_background_spawn_checkpoints_api_routing_before_return(
    monkeypatch, tmp_path
):
    """A real terminal spawn is immediately recoverable with its API route."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import async_delegation
    from tools import process_registry as process_registry_module

    registry = process_registry_module.process_registry
    checkpoint = tmp_path / "processes.json"
    task_id = "checkpoint-api-route"
    config = {
        "env_type": "local",
        "cwd": str(tmp_path),
        "timeout": 60,
        "lifetime_seconds": 3600,
    }
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {task_id: SimpleNamespace(env={}, cwd=str(tmp_path))},
    )
    monkeypatch.setattr(terminal_tool, "_last_activity", {task_id: 0.0})
    monkeypatch.setattr(
        process_registry_module,
        "CHECKPOINT_PATH",
        checkpoint,
    )
    tokens = set_session_vars(
        platform="api_server",
        chat_id="raw-writer-session",
        chat_type="dm",
        session_key="memory-scope",
        session_id="parent-writer-session",
        profile="writer",
        api_route_profile="writer",
        async_delivery=False,
    )
    process_id = ""
    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                command="sleep 30",
                background=True,
                notify_on_complete=True,
                task_id=task_id,
                force=True,
            )
        )
        process_id = result["session_id"]
        rows = json.loads(checkpoint.read_text(encoding="utf-8"))
        row = next(item for item in rows if item["session_id"] == process_id)
        assert row["origin_session_id"] == "raw-writer-session"
        assert row["origin_profile"] == "writer"
        assert row["origin_api_route_profile"] == "writer"
        assert row["parent_session_id"] == "parent-writer-session"
        assert row["notify_on_complete"] is True

        monkeypatch.setattr(
            async_delegation,
            "restore_undelivered_completions",
            lambda _queue: 0,
        )
        recovered_registry = process_registry_module.ProcessRegistry()
        assert recovered_registry.recover_from_checkpoint() == 1
        recovered = recovered_registry.get(process_id)
        assert recovered is not None
        assert recovered.origin_session_id == "raw-writer-session"
        assert recovered.origin_profile == "writer"
        assert recovered.origin_api_route_profile == "writer"
        assert recovered.notify_on_complete is True
    finally:
        clear_session_vars(tokens)
        if process_id:
            registry.kill_process(process_id)


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_terminal_schema_advertises_persistent_env_state():
    description = terminal_tool.TERMINAL_TOOL_DESCRIPTION

    assert "exported environment variables persist between calls" in description
    assert "activate a virtualenv" in description
    assert "once per session" in description


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_validate_workdir_allows_unicode_filesystem_paths():
    assert terminal_tool._validate_workdir(
        "/Users/alice/Documents/Obs_Hermes_Data/项目-projects/客户拜访"
    ) is None
    assert terminal_tool._validate_workdir("/tmp/テスト") is None
    assert terminal_tool._validate_workdir("/home/jürgen/über projekt") is None


def test_validate_workdir_still_blocks_metachars_in_unicode_paths():
    # Widening to Unicode letters must not open the injection boundary:
    # shell metacharacters and control chars stay rejected even when mixed
    # with non-ASCII path segments.
    assert terminal_tool._validate_workdir("/tmp/テスト; rm -rf /")
    assert terminal_tool._validate_workdir("/tmp/项目$(whoami)")
    assert terminal_tool._validate_workdir("/tmp/über`id`")
    assert terminal_tool._validate_workdir("/tmp/テスト\nwhoami")
    assert terminal_tool._validate_workdir("/tmp/项目|cat /etc/passwd")
    assert terminal_tool._validate_workdir("/tmp/ü\x00ber")


def test_count_real_sudo_invocations_ignores_mentions(monkeypatch):
    assert terminal_tool._count_real_sudo_invocations("grep sudo README.md") == 0
    assert terminal_tool._count_real_sudo_invocations("sudo a; sudo b") == 2
