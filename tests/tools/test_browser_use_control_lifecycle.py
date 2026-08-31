"""Browser Use dialog, artifact, and daemon lifecycle contracts.

These tests exercise the supported Browser Harness process boundary: Hermes
supplies isolated ``BH_RUNTIME_DIR`` / ``BH_TMP_DIR`` values and asks the CLI
that created a daemon to reload that exact runtime.  A small Python CLI fixture
stands in for the external executable so the tests can spawn and reap a real
child process without depending on a user's browser installation.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import tools.browser_use_cli as bu_cli


@pytest.fixture(autouse=True)
def _isolated_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("BH_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("BH_TMP_DIR", raising=False)
    reset = getattr(bu_cli, "_reset_browser_use_lifecycle_for_tests", None)
    if reset is not None:
        reset()
    yield
    if reset is not None:
        reset()


def _python_cli(tmp_path: Path, source: str) -> list[str]:
    script = tmp_path / "fake_browser_use.py"
    script.write_text(textwrap.dedent(source), encoding="utf-8")
    return [sys.executable, str(script)]


def _wait_for_process_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


def _terminate_exact_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if not _wait_for_process_exit(pid):
        os.kill(pid, signal.SIGKILL)
        assert _wait_for_process_exit(pid)


class TestDialogSafeClickCompatibility:
    def _namespace(self, monkeypatch, *, dialog_after_release: bool):
        release_started = threading.Event()
        release_gate = threading.Event()
        dialog = {"type": "prompt", "message": "name?"}

        helpers = types.ModuleType("browser_harness.helpers")

        def send(req):
            assert req == {"meta": "pending_dialog"}
            return {
                "dialog": dialog
                if dialog_after_release and release_started.is_set()
                else None
            }

        helpers._send = send
        package = types.ModuleType("browser_harness")
        package.helpers = helpers
        monkeypatch.setitem(sys.modules, "browser_harness", package)
        monkeypatch.setitem(sys.modules, "browser_harness.helpers", helpers)

        calls = []

        def cdp(method, **params):
            calls.append((method, params))
            if params.get("type") == "mouseReleased":
                release_started.set()
                release_gate.wait(2)
            return {}

        namespace = {"cdp": cdp}
        exec(bu_cli._dialog_safe_click_preamble(timeout_s=0.25), namespace)
        return namespace, calls, release_started, release_gate

    def test_release_triggered_prompt_is_exposed_without_deadlock(
        self, monkeypatch, capsys
    ):
        namespace, calls, release_started, release_gate = self._namespace(
            monkeypatch, dialog_after_release=True
        )

        started = time.monotonic()
        result = namespace["click_at_xy"](10, 20)
        elapsed = time.monotonic() - started
        release_gate.set()

        assert elapsed < 0.25
        assert release_started.is_set()
        assert result["status"] == "dialog_pending"
        assert result["dialog"] == {"type": "prompt", "message": "name?"}
        assert "Page.handleJavaScriptDialog" in result["next_action"]
        assert "dialog_pending" in capsys.readouterr().out
        assert [params["type"] for _, params in calls] == [
            "mousePressed",
            "mouseReleased",
        ]

    def test_input_wedge_is_a_precise_error(self, monkeypatch):
        namespace, _, release_started, release_gate = self._namespace(
            monkeypatch, dialog_after_release=False
        )

        with pytest.raises(RuntimeError, match="mouseReleased.*no JavaScript dialog"):
            namespace["click_at_xy"](10, 20)

        assert release_started.is_set()
        release_gate.set()

    def test_dialog_metadata_ipc_failure_is_a_precise_error(self, monkeypatch):
        helpers = types.ModuleType("browser_harness.helpers")
        helpers._send = lambda _request: (_ for _ in ()).throw(
            TimeoutError("fixture IPC timed out")
        )
        package = types.ModuleType("browser_harness")
        package.helpers = helpers
        monkeypatch.setitem(sys.modules, "browser_harness", package)
        monkeypatch.setitem(sys.modules, "browser_harness.helpers", helpers)
        namespace = {"cdp": lambda *_args, **_kwargs: {}}
        exec(bu_cli._dialog_safe_click_preamble(timeout_s=0.1), namespace)

        with pytest.raises(
            RuntimeError,
            match="HERMES_BROWSER_CONTROL_WEDGED.*fixture IPC timed out",
        ):
            namespace["click_at_xy"](10, 20)

    def test_control_wedge_exit_stops_exact_managed_runtime(
        self, tmp_path, monkeypatch
    ):
        cli = _python_cli(
            tmp_path,
            """
            import sys
            if '--reload' in sys.argv:
                raise SystemExit(0)
            sys.stdin.read()
            print('HERMES_BROWSER_CONTROL_WEDGED: fixture', file=sys.stderr)
            raise SystemExit(3)
            """,
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)

        result = json.loads(
            bu_cli.browser_exec("click_at_xy(10, 20)", task_id="task-a")
        )

        assert result["success"] is False
        assert result["exit_code"] == 3
        assert result["daemon_cleanup"] == {"attempted": True, "success": True}
        assert bu_cli._browser_use_sessions == {}


class TestConcurrentScreenshotArtifacts:
    def test_named_calls_never_share_or_overwrite_default_screenshot(
        self, tmp_path, monkeypatch
    ):
        shared = tmp_path / "external-default-tmp"
        cli = _python_cli(
            tmp_path,
            f"""
            import os, pathlib, sys, time
            if '--reload' in sys.argv:
                raise SystemExit(0)
            sys.stdin.read()
            root = pathlib.Path(os.environ.get('BH_TMP_DIR', {str(shared)!r}))
            root.mkdir(parents=True, exist_ok=True)
            shot = root / 'shot.png'
            name = os.environ.get('BU_NAME', 'default')
            shot.write_bytes(name.encode('utf-8'))
            time.sleep(0.20 if name == 'alpha' else 0.05)
            print(shot)
            """,
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)
        monkeypatch.setattr(
            "tools.vision_tools._should_use_native_vision_fast_path", lambda: False
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            alpha_future = pool.submit(
                bu_cli.browser_exec,
                "print(capture_screenshot())",
                "alpha",
                5,
                "task-alpha",
            )
            beta_future = pool.submit(
                bu_cli.browser_exec,
                "print(capture_screenshot())",
                "beta",
                5,
                "task-beta",
            )
            alpha = json.loads(alpha_future.result())
            beta = json.loads(beta_future.result())

        alpha_path = Path(alpha["screenshot_path"])
        beta_path = Path(beta["screenshot_path"])
        assert alpha_path != beta_path
        assert alpha_path.is_file() and alpha_path.read_bytes() == b"alpha"
        assert beta_path.is_file() and beta_path.read_bytes() == b"beta"
        assert alpha_path.parent != beta_path.parent


class TestPersistedLifecycleOwnership:
    def _persisted_instance(self, *, live_owner: bool) -> tuple[Path, Path]:
        profile_root = bu_cli._browser_use_profile_root()
        profile_root.mkdir(parents=True, exist_ok=True)
        instance_root = profile_root / (
            "live-instance" if live_owner else "dead-instance"
        )
        runtime_dir = instance_root / "s-persisted"
        runtime_dir.mkdir(parents=True)
        owner = {
            "managed_by": "hermes-browser-use",
            "instance_id": instance_root.name,
            "pid": os.getpid() if live_owner else 2_000_000_000,
            "process_started_at": (
                bu_cli._browser_use_process_start(os.getpid())
                if live_owner
                else 1.0
            ),
        }
        (instance_root / bu_cli._OWNER_FILE).write_text(json.dumps(owner))
        (runtime_dir / bu_cli._SESSION_FILE).write_text(
            json.dumps(
                {
                    "managed_by": "hermes-browser-use",
                    "instance_id": instance_root.name,
                    "last_activity": 0.0,
                }
            )
        )
        return instance_root, runtime_dir

    def test_live_owner_is_never_reaped(self, monkeypatch):
        instance_root, runtime_dir = self._persisted_instance(live_owner=True)
        reload_called = False

        def unexpected_reload(_state):
            nonlocal reload_called
            reload_called = True
            return True, ""

        monkeypatch.setattr(bu_cli, "_reload_browser_use_session", unexpected_reload)

        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert not reload_called
        assert instance_root.is_dir()
        assert runtime_dir.is_dir()

    def test_unverifiable_owner_is_never_reaped(self, monkeypatch):
        instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        owner_path = instance_root / bu_cli._OWNER_FILE
        owner = json.loads(owner_path.read_text())
        owner.pop("process_started_at")
        owner_path.write_text(json.dumps(owner))
        reload_called = False

        def unexpected_reload(_state):
            nonlocal reload_called
            reload_called = True
            return True, ""

        monkeypatch.setattr(bu_cli, "_reload_browser_use_session", unexpected_reload)

        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert not reload_called
        assert instance_root.is_dir()
        assert runtime_dir.is_dir()

    def test_dead_owner_reloads_only_its_persisted_runtime(
        self, tmp_path, monkeypatch
    ):
        instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        observed = tmp_path / "reload-observed.json"
        cli = _python_cli(
            tmp_path,
            f"""
            import json, os, pathlib, sys
            if '--reload' not in sys.argv:
                raise SystemExit(9)
            pathlib.Path({str(observed)!r}).write_text(json.dumps({{
                'runtime': os.environ.get('BH_RUNTIME_DIR'),
                'tmp': os.environ.get('BH_TMP_DIR'),
            }}))
            """,
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)

        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert json.loads(observed.read_text()) == {
            "runtime": str(runtime_dir),
            "tmp": str(runtime_dir / "cleanup"),
        }
        assert not instance_root.exists()


class TestLogicalSessionIdentity:
    def test_explicit_name_is_stable_across_tasks(self):
        assert bu_cli._browser_use_session_key(
            "task-a", "research"
        ) == bu_cli._browser_use_session_key("task-b", "research")

    def test_unnamed_default_is_isolated_per_task(self):
        assert bu_cli._browser_use_session_key(
            "task-a", ""
        ) != bu_cli._browser_use_session_key("task-b", "")

    def test_fork_child_reset_discards_inherited_state_without_reload(
        self, monkeypatch
    ):
        reload_called = False

        def unexpected_reload(_state):
            nonlocal reload_called
            reload_called = True
            return True, ""

        monkeypatch.setattr(bu_cli, "_reload_browser_use_session", unexpected_reload)
        bu_cli._browser_use_sessions["parent-runtime"] = {
            "runtime_dir": Path("/parent-owned-runtime"),
        }
        bu_cli._browser_use_instance = {"pid": 123, "instance_id": "parent"}

        bu_cli._reset_browser_use_lifecycle_after_fork()
        bu_cli._shutdown_browser_use_lifecycle()

        assert not reload_called
        assert bu_cli._browser_use_sessions == {}
        assert bu_cli._browser_use_instance is None
        assert bu_cli._browser_use_owner_pid == os.getpid()


@pytest.mark.live_system_guard_bypass
class TestManagedDaemonLifecycle:
    _DAEMON_CLI = """
        import os, pathlib, signal, subprocess, sys, time
        runtime = pathlib.Path(os.environ['BH_RUNTIME_DIR'])
        runtime.mkdir(parents=True, exist_ok=True)
        pid_path = runtime / 'fake-daemon.pid'
        if '--reload' in sys.argv:
            if os.environ.get('FAKE_RELOAD_FAIL') == '1':
                print('reload refused by fixture', file=sys.stderr)
                raise SystemExit(7)
            try:
                pid = int(pid_path.read_text())
            except (FileNotFoundError, ValueError):
                raise SystemExit(0)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            pid_path.unlink(missing_ok=True)
            raise SystemExit(0)
        if not pid_path.exists():
            daemon = subprocess.Popen(
                [sys.executable, '-c', 'import time; time.sleep(60)'],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            pid_path.write_text(str(daemon.pid))
        sys.stdin.read()
        print('runtime=' + str(runtime))
    """

    def test_inactivity_reloads_only_the_exact_owned_runtime(
        self, tmp_path, monkeypatch
    ):
        cli = _python_cli(tmp_path, self._DAEMON_CLI)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)
        monkeypatch.setattr(bu_cli, "_browser_use_inactivity_timeout", lambda: 30)

        result = json.loads(
            bu_cli.browser_exec("print(page_info())", session="owned", task_id="task-a")
        )
        assert result["success"] is True
        assert len(bu_cli._browser_use_sessions) == 1
        state = next(iter(bu_cli._browser_use_sessions.values()))
        pid = int((state["runtime_dir"] / "fake-daemon.pid").read_text())
        assert not _wait_for_process_exit(pid, timeout=0.05)

        state["last_activity"] = 0.0
        try:
            failures = bu_cli._cleanup_browser_use_sessions(now=1000.0)
            assert failures == []
            assert _wait_for_process_exit(pid)
            assert bu_cli._browser_use_sessions == {}
        finally:
            _terminate_exact_pid(pid)

    def test_operator_runtime_opts_out_and_is_never_reaped(
        self, tmp_path, monkeypatch
    ):
        cli = _python_cli(tmp_path, self._DAEMON_CLI)
        external = tmp_path / "operator-owned-runtime"
        monkeypatch.setenv("BH_RUNTIME_DIR", str(external))
        monkeypatch.setenv("BH_TMP_DIR", str(external / "tmp"))
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)

        result = json.loads(
            bu_cli.browser_exec("print(page_info())", session="manual", task_id="task-a")
        )
        pid = int((external / "fake-daemon.pid").read_text())
        try:
            assert result["success"] is True
            assert bu_cli._browser_use_sessions == {}
            assert bu_cli._cleanup_browser_use_sessions(now=time.time() + 3600) == []
            assert not _wait_for_process_exit(pid, timeout=0.05)
        finally:
            _terminate_exact_pid(pid)

    def test_timeout_surfaces_exact_reload_failure(self, tmp_path, monkeypatch):
        cli = _python_cli(
            tmp_path,
            """
            import os, sys, time
            if '--reload' in sys.argv:
                print('fixture reload failed', file=sys.stderr)
                raise SystemExit(7)
            sys.stdin.read()
            time.sleep(30)
            """,
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)
        monkeypatch.setattr(bu_cli, "_MIN_TIMEOUT_S", 1)

        parsed = json.loads(
            bu_cli.browser_exec("print(1)", session="wedged", timeout_s=1)
        )

        assert "timed out" in parsed["error"]
        assert "cleanup failed" in parsed["error"]
        assert "fixture reload failed" in parsed["error"]

    def test_reload_zero_with_live_exact_daemon_is_reported_as_failure(
        self, tmp_path, monkeypatch
    ):
        cli = _python_cli(
            tmp_path,
            """
            import os, pathlib, subprocess, sys
            runtime = pathlib.Path(os.environ['BH_RUNTIME_DIR'])
            runtime.mkdir(parents=True, exist_ok=True)
            pid_path = runtime / 'bu.pid'
            if '--reload' in sys.argv:
                raise SystemExit(0)
            if not pid_path.exists():
                daemon = subprocess.Popen(
                    [sys.executable, '-c', 'import time; time.sleep(60)'],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                pid_path.write_text(str(daemon.pid))
            sys.stdin.read()
            print('started')
            """,
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)
        monkeypatch.setattr(bu_cli, "_LIFECYCLE_VERIFY_TIMEOUT_S", 0.1)

        result = json.loads(
            bu_cli.browser_exec("print(page_info())", session="owned", task_id="task-a")
        )
        assert result["success"] is True
        state = next(iter(bu_cli._browser_use_sessions.values()))
        pid = int((state["runtime_dir"] / "bu.pid").read_text())
        try:
            state["last_activity"] = 0.0
            failures = bu_cli._cleanup_browser_use_sessions(now=1000.0)
            assert len(failures) == 1
            assert "exited 0" in failures[0]
            assert "is still alive" in failures[0]
            assert str(state["runtime_dir"]) in bu_cli._browser_use_sessions
            assert not _wait_for_process_exit(pid, timeout=0.05)
        finally:
            _terminate_exact_pid(pid)
