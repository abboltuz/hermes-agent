"""Browser Use dialog, artifact, and daemon lifecycle contracts.

These tests exercise the supported Browser Harness process boundary: Hermes
supplies isolated ``BH_RUNTIME_DIR`` / ``BH_TMP_DIR`` values and asks the CLI
environment that created a daemon to stop that exact runtime. A small fixture
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


def _write_owned_session_marker(
    runtime_dir: Path, instance_id: str = "fixture-instance"
) -> None:
    (runtime_dir / bu_cli._SESSION_FILE).write_text(
        json.dumps(
            {
                "managed_by": "hermes-browser-use",
                "instance_id": instance_id,
                "session": "",
                "last_activity": 0.0,
                "cleanup_state": "active",
            }
        )
    )


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

        helpers.cdp = cdp

        def stock_click(x, y, button="left", clicks=1):
            helpers.cdp(
                "Input.dispatchMouseEvent",
                type="mousePressed",
                x=x,
                y=y,
                button=button,
                clickCount=clicks,
            )
            helpers.cdp(
                "Input.dispatchMouseEvent",
                type="mouseReleased",
                x=x,
                y=y,
                button=button,
                clickCount=clicks,
            )

        stock_click.__bh_traced__ = True
        namespace = {"cdp": cdp, "click_at_xy": stock_click}
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
        helpers.cdp = lambda *_args, **_kwargs: {}
        namespace = {
            "cdp": helpers.cdp,
            "click_at_xy": lambda *_args, **_kwargs: None,
        }
        exec(bu_cli._dialog_safe_click_preamble(timeout_s=0.1), namespace)

        with pytest.raises(
            RuntimeError,
            match="HERMES_BROWSER_CONTROL_WEDGED.*fixture IPC timed out",
        ):
            namespace["click_at_xy"](10, 20)

    def test_compatibility_layer_preserves_harness_trace_and_debug_behavior(
        self, monkeypatch
    ):
        monkeypatch.setenv("BH_DEBUG_CLICKS", "1")
        helpers = types.ModuleType("browser_harness.helpers")
        helpers._send = lambda _request: {"dialog": None}
        package = types.ModuleType("browser_harness")
        package.helpers = helpers
        monkeypatch.setitem(sys.modules, "browser_harness", package)
        monkeypatch.setitem(sys.modules, "browser_harness.helpers", helpers)

        dispatches = []
        debug_overlays = []
        observations = []

        def helper_cdp(method, **params):
            dispatches.append((method, params))
            return {}

        helpers.cdp = helper_cdp

        def stock_click(x, y, button="left", clicks=1):
            if os.environ.get("BH_DEBUG_CLICKS"):
                debug_overlays.append((x, y, button, clicks))
            helpers.cdp(
                "Input.dispatchMouseEvent",
                type="mousePressed",
                x=x,
                y=y,
                button=button,
                clickCount=clicks,
            )
            helpers.cdp(
                "Input.dispatchMouseEvent",
                type="mouseReleased",
                x=x,
                y=y,
                button=button,
                clickCount=clicks,
            )

        def traced_click(*args, **kwargs):
            result = stock_click(*args, **kwargs)
            observations.append(("click_at_xy", args, kwargs))
            return result

        traced_click.__bh_traced__ = True
        namespace = {"cdp": helper_cdp, "click_at_xy": traced_click}

        exec(bu_cli._dialog_safe_click_preamble(timeout_s=0.25), namespace)
        result = namespace["click_at_xy"](12, 34, button="left", clicks=1)

        assert result is None
        assert namespace["click_at_xy"].__bh_traced__ is True
        assert debug_overlays == [(12, 34, "left", 1)]
        assert observations == [("click_at_xy", (12, 34), {"button": "left", "clicks": 1})]
        assert [params["type"] for _, params in dispatches] == [
            "mousePressed",
            "mouseReleased",
        ]

    def test_control_wedge_exit_stops_exact_managed_runtime(
        self, tmp_path, monkeypatch
    ):
        cli = _python_cli(
            tmp_path,
            """
            import os, pathlib, sys
            if os.environ.get('BH_TMP_DIR', '').endswith('cleanup'):
                sys.stdin.read()
                print('[hermes-browser-cleanup] {"ok": true, "version": "0.1.10", "browser_kind": "cdp"}')
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
            if os.environ.get('BH_TMP_DIR', '').endswith('cleanup'):
                sys.stdin.read()
                print('[hermes-browser-cleanup] {{"ok": true, "version": "0.1.10", "browser_kind": "cdp"}}')
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

        failures = bu_cli._reap_orphaned_browser_use_instances()
        assert len(failures) == 1
        assert "owner is unverifiable" in failures[0]
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
            if not os.environ.get('BH_TMP_DIR', '').endswith('cleanup'):
                raise SystemExit(9)
            sys.stdin.read()
            pathlib.Path({str(observed)!r}).write_text(json.dumps({{
                'runtime': os.environ.get('BH_RUNTIME_DIR'),
                'tmp': os.environ.get('BH_TMP_DIR'),
            }}))
            print('[hermes-browser-cleanup] {{"ok": true, "version": "0.1.10", "browser_kind": "cdp"}}')
            """,
        )
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: cli)

        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert json.loads(observed.read_text()) == {
            "runtime": str(runtime_dir),
            "tmp": str(runtime_dir / "cleanup"),
        }
        assert not instance_root.exists()

    def test_stale_reaper_claim_does_not_block_verified_dead_owner(
        self, monkeypatch
    ):
        instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        (instance_root / bu_cli._REAP_LOCK_FILE).write_text(
            json.dumps(
                {
                    "pid": 2_000_000_000,
                    "process_started_at": 1.0,
                    "claim_id": "crashed-reaper",
                }
            )
        )
        reloads = []

        def successful_reload(state):
            reloads.append(Path(state["runtime_dir"]))
            return bu_cli._mark_browser_use_cleanup_confirmed(state)

        monkeypatch.setattr(bu_cli, "_reload_browser_use_session", successful_reload)

        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert reloads == [runtime_dir]
        assert not instance_root.exists()

    def test_active_reaper_claim_is_never_stolen(self, monkeypatch):
        instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        claim, busy, error = bu_cli._acquire_browser_use_reap_claim(instance_root)
        assert claim is not None
        assert busy is False
        assert error == ""
        reload_called = False

        def unexpected_reload(_state):
            nonlocal reload_called
            reload_called = True
            return True, ""

        monkeypatch.setattr(bu_cli, "_reload_browser_use_session", unexpected_reload)
        try:
            assert bu_cli._reap_orphaned_browser_use_instances() == []
            assert not reload_called
            assert instance_root.is_dir()
            assert runtime_dir.is_dir()
        finally:
            bu_cli._release_browser_use_reap_claim(claim)

    def test_cleanup_tmp_directory_is_fully_retired_without_losing_authority(
        self, tmp_path, monkeypatch
    ):
        instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        cli = _python_cli(
            tmp_path,
            """
            import json, os, pathlib, sys
            cleanup = pathlib.Path(os.environ['BH_TMP_DIR'])
            cleanup.mkdir(parents=True, exist_ok=True)
            sys.stdin.read()
            print('[hermes-browser-cleanup] ' + json.dumps({
                'ok': True,
                'version': '0.1.10',
                'browser_kind': 'cdp',
            }))
            """,
        )
        state = {
            "runtime_dir": runtime_dir,
            "instance_id": instance_root.name,
            "command": cli,
            "session": "",
            "last_activity": 0.0,
            "operation_lock": threading.Lock(),
        }
        bu_cli._browser_use_sessions[str(runtime_dir)] = state

        assert bu_cli._cleanup_browser_use_sessions(force=True) == []
        assert not runtime_dir.exists()
        assert str(runtime_dir) not in bu_cli._browser_use_sessions

        reload_calls = []
        monkeypatch.setattr(
            bu_cli,
            "_reload_browser_use_session",
            lambda followup: reload_calls.append(followup) or (True, ""),
        )
        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert reload_calls == []
        assert not instance_root.exists()

    def test_dead_owner_retires_empty_markerless_crash_window(self, monkeypatch):
        instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        (runtime_dir / bu_cli._SESSION_FILE).unlink()
        reload_calls = []
        monkeypatch.setattr(
            bu_cli,
            "_reload_browser_use_session",
            lambda state: reload_calls.append(state) or (True, ""),
        )

        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert reload_calls == []
        assert not runtime_dir.exists()
        assert not instance_root.exists()

    @pytest.mark.windows_only
    def test_windows_reaper_retires_instance_with_native_lock_handle(
        self, monkeypatch
    ):
        instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        session_path = runtime_dir / bu_cli._SESSION_FILE
        persisted = json.loads(session_path.read_text())
        persisted["cleanup_state"] = "confirmed"
        session_path.write_text(json.dumps(persisted))

        monkeypatch.setattr(
            bu_cli,
            "_reload_browser_use_session",
            lambda _state: (_ for _ in ()).throw(
                AssertionError("confirmed cleanup must not be repeated")
            ),
        )

        assert bu_cli._reap_orphaned_browser_use_instances() == []
        assert not runtime_dir.exists()
        assert not instance_root.exists()

    def test_retirement_rejects_unexpected_cleanup_content_and_keeps_marker(self):
        _instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        session_path = runtime_dir / bu_cli._SESSION_FILE
        persisted = json.loads(session_path.read_text())
        persisted["cleanup_state"] = "confirmed"
        session_path.write_text(json.dumps(persisted))
        cleanup_dir = runtime_dir / "cleanup"
        cleanup_dir.mkdir()
        (cleanup_dir / "unexpected.txt").write_text("retain me")

        retired, detail = bu_cli._retire_browser_use_runtime(
            {
                "runtime_dir": runtime_dir,
                "instance_id": runtime_dir.parent.name,
                "cleanup_state": "confirmed",
            }
        )

        assert retired is False
        assert "unexpected entries" in detail
        assert session_path.exists()
        assert (cleanup_dir / "unexpected.txt").read_text() == "retain me"

    @pytest.mark.require_symlinks
    def test_retirement_rejects_symlinked_cleanup_and_keeps_marker(self, tmp_path):
        _instance_root, runtime_dir = self._persisted_instance(live_owner=False)
        session_path = runtime_dir / bu_cli._SESSION_FILE
        persisted = json.loads(session_path.read_text())
        persisted["cleanup_state"] = "confirmed"
        session_path.write_text(json.dumps(persisted))
        outside = tmp_path / "outside-cleanup"
        outside.mkdir()
        (runtime_dir / "cleanup").symlink_to(outside, target_is_directory=True)

        retired, detail = bu_cli._retire_browser_use_runtime(
            {
                "runtime_dir": runtime_dir,
                "instance_id": runtime_dir.parent.name,
                "cleanup_state": "confirmed",
            }
        )

        assert retired is False
        assert "cleanup path is unsafe" in detail
        assert session_path.exists()
        assert outside.is_dir()


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


class TestOwnedTabLifecycle:
    @pytest.mark.parametrize(
        ("mode", "expected_target", "expected_origin", "create_calls"),
        [
            ("claim", "harness-target", "harness-dedicated", 0),
            ("create", "hermes-target", "hermes-created", 1),
        ],
    )
    def test_shared_session_persists_one_exact_owned_target(
        self,
        tmp_path,
        monkeypatch,
        mode,
        expected_target,
        expected_origin,
        create_calls,
    ):
        target_file = tmp_path / bu_cli._TARGET_FILE
        pid_file = tmp_path / "bu.pid"
        pid_file.write_text("4242")
        monkeypatch.setenv("HERMES_BH_TARGET_FILE", str(target_file))
        monkeypatch.setenv("HERMES_BH_INSTANCE_ID", "fixture-instance")
        monkeypatch.setenv("HERMES_BH_TARGET_MODE", mode)

        targets = ["user-target"]
        if mode == "claim":
            targets.insert(0, "harness-target")
        created = []
        switched = []

        helpers = types.ModuleType("browser_harness.helpers")
        helpers._send = lambda payload: (
            {"targetId": "harness-target"}
            if payload == {"meta": "current_tab"}
            else (_ for _ in ()).throw(AssertionError(payload))
        )
        ipc = types.ModuleType("browser_harness._ipc")
        ipc.pid_path = lambda _name: pid_file
        package = types.ModuleType("browser_harness")
        package.helpers = helpers
        package._ipc = ipc
        monkeypatch.setitem(sys.modules, "browser_harness", package)
        monkeypatch.setitem(sys.modules, "browser_harness.helpers", helpers)
        monkeypatch.setitem(sys.modules, "browser_harness._ipc", ipc)

        def cdp(method, **params):
            if method == "Target.getTargets":
                return {
                    "targetInfos": [
                        {"targetId": target, "type": "page"} for target in targets
                    ]
                }
            if method == "Target.createTarget":
                created.append(params)
                targets.append("hermes-target")
                return {"targetId": "hermes-target"}
            if method == "Target.closeTarget":
                targets.remove(params["targetId"])
                return {"success": True}
            raise AssertionError(method)

        namespace = {"cdp": cdp, "switch_tab": switched.append}
        exec(bu_cli._OWN_TAB_PREAMBLE, namespace)
        # A second call for the same daemon must reuse the persisted ownership
        # rather than create or claim another target.
        exec(bu_cli._OWN_TAB_PREAMBLE, namespace)

        marker = json.loads(target_file.read_text())
        assert marker == {
            "daemon_pid": "4242",
            "instance_id": "fixture-instance",
            "managed_by": "hermes-browser-use",
            "origin": expected_origin,
            "target_id": expected_target,
        }
        assert len(created) == create_calls
        assert switched == (["hermes-target"] if mode == "create" else [])
        assert "user-target" in targets


class TestDaemonContractLifecycle:
    def test_stale_contract_without_endpoint_is_never_overwritten(self, tmp_path):
        runtime = tmp_path / "instance" / "s-runtime"
        runtime.mkdir(parents=True)
        (runtime / bu_cli._DAEMON_FILE).write_text(
            json.dumps(
                {
                    "managed_by": "hermes-browser-use",
                    "instance_id": "instance",
                    "daemon_pid": "4242",
                    "daemon_started": "fixture-start",
                    "harness_version": "0.1.9",
                    "strict_cloud_shutdown": False,
                }
            )
        )

        assert (
            bu_cli._browser_use_should_record_daemon_contract(
                {"runtime_dir": runtime, "cleanup_state": "active"}
            )
            is False
        )
        assert (
            bu_cli._browser_use_should_record_daemon_contract(
                {"runtime_dir": runtime, "cleanup_state": "confirmed"}
            )
            is True
        )

    @pytest.mark.parametrize(
        ("version", "strict"), [("0.1.9", False), ("0.1.10", True)]
    )
    def test_new_daemon_records_exact_pid_and_spawning_package_contract(
        self, tmp_path, monkeypatch, version, strict
    ):
        contract_file = tmp_path / bu_cli._DAEMON_FILE
        pid_file = tmp_path / "bu.pid"
        pid_file.write_text("4242")
        monkeypatch.setenv("HERMES_BH_RECORD_DAEMON_CONTRACT", "1")
        monkeypatch.setenv("HERMES_BH_DAEMON_CONTRACT_FILE", str(contract_file))
        monkeypatch.setenv("HERMES_BH_INSTANCE_ID", "fixture-instance")

        admin = types.ModuleType("browser_harness.admin")
        admin._version = lambda: version
        admin._process_start_time = lambda _pid: "fixture-start"
        ipc = types.ModuleType("browser_harness._ipc")
        ipc.pid_path = lambda _name: pid_file
        package = types.ModuleType("browser_harness")
        package.admin = admin
        package._ipc = ipc
        monkeypatch.setitem(sys.modules, "browser_harness", package)
        monkeypatch.setitem(sys.modules, "browser_harness.admin", admin)
        monkeypatch.setitem(sys.modules, "browser_harness._ipc", ipc)

        exec(bu_cli._DAEMON_CONTRACT_PREAMBLE, {})

        assert json.loads(contract_file.read_text()) == {
            "daemon_pid": "4242",
            "daemon_started": "fixture-start",
            "harness_version": version,
            "instance_id": "fixture-instance",
            "managed_by": "hermes-browser-use",
            "strict_cloud_shutdown": strict,
        }

    def test_existing_daemon_contract_is_not_rewritten_by_newer_cli(
        self, tmp_path, monkeypatch
    ):
        contract_file = tmp_path / bu_cli._DAEMON_FILE
        old_contract = {
            "daemon_pid": "4242",
            "daemon_started": "fixture-start",
            "harness_version": "0.1.9",
            "instance_id": "fixture-instance",
            "managed_by": "hermes-browser-use",
            "strict_cloud_shutdown": False,
        }
        contract_file.write_text(json.dumps(old_contract))
        monkeypatch.setenv("HERMES_BH_RECORD_DAEMON_CONTRACT", "0")
        monkeypatch.setenv("HERMES_BH_DAEMON_CONTRACT_FILE", str(contract_file))
        monkeypatch.setenv("HERMES_BH_INSTANCE_ID", "fixture-instance")

        exec(bu_cli._DAEMON_CONTRACT_PREAMBLE, {})

        assert json.loads(contract_file.read_text()) == old_contract


@pytest.mark.live_system_guard_bypass
class TestManagedDaemonLifecycle:
    _DAEMON_CLI = """
        import os, pathlib, signal, subprocess, sys, time
        runtime = pathlib.Path(os.environ['BH_RUNTIME_DIR'])
        runtime.mkdir(parents=True, exist_ok=True)
        pid_path = runtime / 'fake-daemon.pid'
        if os.environ.get('BH_TMP_DIR', '').endswith('cleanup'):
            sys.stdin.read()
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
            print('[hermes-browser-cleanup] {"ok": true, "version": "0.1.10", "browser_kind": "cdp"}')
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
            if os.environ.get('BH_TMP_DIR', '').endswith('cleanup'):
                sys.stdin.read()
                print('[hermes-browser-cleanup] {"ok": false, "version": "0.1.10", "error": "fixture reload failed"}')
                raise SystemExit(86)
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
            if os.environ.get('BH_TMP_DIR', '').endswith('cleanup'):
                sys.stdin.read()
                print('[hermes-browser-cleanup] {"ok": true, "version": "0.1.10", "browser_kind": "cdp"}')
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
            assert "strict cleanup returned" in failures[0]
            assert "is still alive" in failures[0]
            assert str(state["runtime_dir"]) in bu_cli._browser_use_sessions
            assert not _wait_for_process_exit(pid, timeout=0.05)
        finally:
            _terminate_exact_pid(pid)

    def test_reload_zero_cannot_hide_daemon_clean_shutdown_failure(
        self, tmp_path
    ):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        hidden_failure = tmp_path / "hidden-failure.txt"
        cli = _python_cli(
            tmp_path,
            f"""
            import pathlib, sys
            runtime = pathlib.Path({str(runtime)!r})
            if '--reload' in sys.argv:
                pathlib.Path({str(hidden_failure)!r}).write_text('billing cleanup failed')
                for name in ('bu.sock', 'bu.port', 'bu.pid'):
                    (runtime / name).unlink(missing_ok=True)
                raise SystemExit(0)
            # The corrected path executes a strict cleanup program. Model the
            # daemon reporting an error even though it also disappears.
            pathlib.Path({str(hidden_failure)!r}).write_text('billing cleanup failed')
            for name in ('bu.sock', 'bu.port', 'bu.pid'):
                (runtime / name).unlink(missing_ok=True)
            print('[hermes-browser-cleanup] {{"ok": false, "error": "billing cleanup failed", "version": "0.1.10"}}')
            raise SystemExit(86)
            """,
        )
        for name in ("bu.sock", "bu.port"):
            (runtime / name).write_text("fixture")

        ok, detail = bu_cli._reload_browser_use_session(
            {"runtime_dir": runtime, "command": cli, "session": "owned"}
        )

        assert hidden_failure.read_text() == "billing cleanup failed"
        assert ok is False
        assert "billing cleanup failed" in detail

    def test_cleanup_never_autospawns_an_absent_daemon(self, tmp_path):
        runtime = tmp_path / "absent-runtime"
        runtime.mkdir()
        spawned = tmp_path / "spawned.txt"
        cli = _python_cli(
            tmp_path,
            f"""
            import pathlib, sys
            code = sys.stdin.read()
            # Model the Harness run.py dispatch boundary: ordinary code calls
            # ensure_daemon(), while an admin-shaped stop program does not.
            if not code.lstrip().startswith('stop_remote_daemon('):
                pathlib.Path({str(spawned)!r}).write_text('spawned')
            print('[hermes-browser-cleanup] {{"ok": false, "error": "owned daemon did not answer an authenticated ping", "version": "0.1.10"}}')
            raise SystemExit(86)
            """,
        )

        ok, detail = bu_cli._reload_browser_use_session(
            {
                "runtime_dir": runtime,
                "command": cli,
                "session": "owned",
                "instance_id": "fixture-instance",
            }
        )

        assert ok is False
        assert "did not answer" in detail
        assert not spawned.exists()

    @pytest.mark.parametrize(
        ("browser_kind", "expected_ok", "shutdown_expected"),
        [("cdp", True, True), ("cloud", False, False)],
    )
    def test_harness_019_cleanup_contract_is_explicit(
        self,
        tmp_path,
        monkeypatch,
        browser_kind,
        expected_ok,
        shutdown_expected,
    ):
        runtime = tmp_path / f"runtime-{browser_kind}"
        runtime.mkdir()
        _write_owned_session_marker(runtime)
        shutdown_marker = tmp_path / f"shutdown-{browser_kind}.txt"
        for name in ("bu.sock", "bu.port"):
            (runtime / name).write_text("fixture")
        monkeypatch.setenv("FIXTURE_BROWSER_KIND", browser_kind)
        monkeypatch.setenv("FIXTURE_SHUTDOWN_MARKER", str(shutdown_marker))
        cli = _python_cli(
            tmp_path,
            """
            import os, pathlib, sys, types
            runtime = pathlib.Path(os.environ['BH_RUNTIME_DIR'])
            marker = pathlib.Path(os.environ['FIXTURE_SHUTDOWN_MARKER'])
            kind = os.environ['FIXTURE_BROWSER_KIND']

            admin = types.ModuleType('browser_harness.admin')
            admin.NAME = os.environ.get('BU_NAME', 'default')
            admin._version = lambda: '0.1.9'
            def restart_daemon(name=None):
                raise AssertionError('0.1.9 compatibility must use explicit IPC')
            admin.restart_daemon = restart_daemon

            ipc = types.ModuleType('browser_harness._ipc')
            class Connection:
                def close(self):
                    pass
            ipc.connect = lambda _name, timeout=5.0: (Connection(), None)
            def request(_conn, _token, payload):
                if payload.get('meta') == 'ping':
                    return {'pong': True, 'pid': os.getpid(), 'browser_kind': kind}
                if payload.get('meta') == 'shutdown':
                    marker.write_text('shutdown-confirmed')
                    for endpoint in ('bu.sock', 'bu.port', 'bu.pid'):
                        (runtime / endpoint).unlink(missing_ok=True)
                    return {'ok': True}
                raise AssertionError(payload)
            ipc.request = request

            package = types.ModuleType('browser_harness')
            package.admin = admin
            package._ipc = ipc
            sys.modules['browser_harness'] = package
            sys.modules['browser_harness.admin'] = admin
            sys.modules['browser_harness._ipc'] = ipc
            exec(sys.stdin.read(), {'cdp': lambda *_args, **_kwargs: {}})
            """,
        )

        ok, detail = bu_cli._reload_browser_use_session(
            {
                "runtime_dir": runtime,
                "command": cli,
                "session": "owned",
                "instance_id": "fixture-instance",
            }
        )

        assert ok is expected_ok
        assert shutdown_marker.exists() is shutdown_expected
        if browser_kind == "cloud":
            assert "cannot confirm Browser Use Cloud" in detail
            assert (runtime / "bu.sock").exists()
        else:
            assert detail == ""
            assert not (runtime / "bu.sock").exists()

    def test_harness_010_cli_rejects_unproven_019_cloud_daemon(
        self, tmp_path, monkeypatch
    ):
        runtime = tmp_path / "mixed-runtime"
        runtime.mkdir()
        (runtime / "bu.sock").write_text("fixture")
        (runtime / ".hermes-daemon.json").write_text(
            json.dumps(
                {
                    "managed_by": "hermes-browser-use",
                    "instance_id": "fixture-instance",
                    "daemon_pid": "4242",
                    "daemon_started": "old-daemon-start",
                    "harness_version": "0.1.9",
                    "strict_cloud_shutdown": False,
                }
            )
        )
        hidden_failure = tmp_path / "hidden-cloud-failure.txt"
        monkeypatch.setenv("FIXTURE_HIDDEN_FAILURE", str(hidden_failure))
        cli = _python_cli(
            tmp_path,
            """
            import os, pathlib, sys, types
            runtime = pathlib.Path(os.environ['BH_RUNTIME_DIR'])
            hidden = pathlib.Path(os.environ['FIXTURE_HIDDEN_FAILURE'])
            pathlib.Path(os.environ['BH_TMP_DIR']).mkdir(parents=True, exist_ok=True)

            ipc = types.ModuleType('browser_harness._ipc')
            class Connection:
                def close(self):
                    pass
            ipc.connect = lambda _name, timeout=5.0: (Connection(), None)
            def request(_conn, _token, payload):
                if payload.get('meta') == 'ping':
                    return {
                        'pong': True,
                        'pid': 4242,
                        'browser_kind': 'cloud',
                    }
                if payload.get('meta') == 'shutdown':
                    # Model a v0.1.9 daemon: acknowledge first, then fail its
                    # best-effort Cloud stop during finalization.
                    hidden.write_text('v0.1.9 daemon cloud stop/billing cleanup failed')
                    return {'ok': True}
                raise AssertionError(payload)
            ipc.request = request

            admin = types.ModuleType('browser_harness.admin')
            admin._version = lambda: '0.1.10'
            def restart_daemon(name=None, require_clean=False):
                assert require_clean is True
                response = request(None, None, {'meta': 'shutdown'})
                assert response == {'ok': True}
                for endpoint in ('bu.sock', 'bu.port', 'bu.pid'):
                    (runtime / endpoint).unlink(missing_ok=True)
            admin.restart_daemon = restart_daemon

            package = types.ModuleType('browser_harness')
            package.admin = admin
            package._ipc = ipc
            sys.modules['browser_harness'] = package
            sys.modules['browser_harness.admin'] = admin
            sys.modules['browser_harness._ipc'] = ipc
            exec(sys.stdin.read(), {'cdp': lambda *_args, **_kwargs: {}})
            """,
        )

        ok, detail = bu_cli._reload_browser_use_session(
            {
                "runtime_dir": runtime,
                "command": cli,
                "session": "owned",
                "instance_id": "fixture-instance",
            }
        )

        assert ok is False
        assert "does not prove strict cloud shutdown" in detail
        assert not hidden_failure.exists()
        assert (runtime / "bu.sock").exists()

    @pytest.mark.parametrize("cli_version", ["0.1.9", "0.1.10"])
    @pytest.mark.parametrize(
        ("live_started", "expected_ok"),
        [("strict-daemon-start", True), ("reused-daemon-start", False)],
    )
    def test_proven_010_cloud_daemon_cleans_with_either_supported_cli(
        self, tmp_path, monkeypatch, cli_version, live_started, expected_ok
    ):
        runtime = tmp_path / f"proven-runtime-{cli_version}-{live_started}"
        runtime.mkdir()
        _write_owned_session_marker(runtime)
        (runtime / "bu.sock").write_text("fixture")
        (runtime / bu_cli._DAEMON_FILE).write_text(
            json.dumps(
                {
                    "managed_by": "hermes-browser-use",
                    "instance_id": "fixture-instance",
                    "daemon_pid": "4242",
                    "daemon_started": "strict-daemon-start",
                    "harness_version": "0.1.10",
                    "strict_cloud_shutdown": True,
                }
            )
        )
        shutdown_marker = tmp_path / f"shutdown-{cli_version}.txt"
        monkeypatch.setenv("FIXTURE_CLI_VERSION", cli_version)
        monkeypatch.setenv("FIXTURE_LIVE_STARTED", live_started)
        monkeypatch.setenv("FIXTURE_SHUTDOWN_MARKER", str(shutdown_marker))
        cli = _python_cli(
            tmp_path,
            """
            import os, pathlib, sys, types
            runtime = pathlib.Path(os.environ['BH_RUNTIME_DIR'])
            shutdown_marker = pathlib.Path(os.environ['FIXTURE_SHUTDOWN_MARKER'])
            pathlib.Path(os.environ['BH_TMP_DIR']).mkdir(parents=True, exist_ok=True)

            admin = types.ModuleType('browser_harness.admin')
            admin._version = lambda: os.environ['FIXTURE_CLI_VERSION']
            admin._process_start_time = lambda _pid: os.environ['FIXTURE_LIVE_STARTED']

            ipc = types.ModuleType('browser_harness._ipc')
            class Connection:
                def close(self):
                    pass
            ipc.connect = lambda _name, timeout=5.0: (Connection(), None)
            def request(_conn, _token, payload):
                if payload.get('meta') == 'ping':
                    return {
                        'pong': True,
                        'pid': 4242,
                        'browser_kind': 'cloud',
                    }
                if payload.get('meta') == 'shutdown':
                    shutdown_marker.write_text('strict-cloud-stop-confirmed')
                    for endpoint in ('bu.sock', 'bu.port', 'bu.pid'):
                        (runtime / endpoint).unlink(missing_ok=True)
                    return {'ok': True}
                raise AssertionError(payload)
            ipc.request = request

            package = types.ModuleType('browser_harness')
            package.admin = admin
            package._ipc = ipc
            sys.modules['browser_harness'] = package
            sys.modules['browser_harness.admin'] = admin
            sys.modules['browser_harness._ipc'] = ipc
            exec(sys.stdin.read(), {'cdp': lambda *_args, **_kwargs: {}})
            """,
        )

        ok, detail = bu_cli._reload_browser_use_session(
            {
                "runtime_dir": runtime,
                "command": cli,
                "session": "owned",
                "instance_id": "fixture-instance",
            }
        )

        assert ok is expected_ok, detail
        if expected_ok:
            assert shutdown_marker.read_text() == "strict-cloud-stop-confirmed"
            persisted = json.loads((runtime / bu_cli._SESSION_FILE).read_text())
            assert persisted["cleanup_state"] == "confirmed"
        else:
            assert "does not prove strict cloud shutdown" in detail
            assert not shutdown_marker.exists()
            assert (runtime / "bu.sock").exists()

    @pytest.mark.parametrize("session", ["owned", ""])
    def test_cleanup_closes_exact_persisted_target_and_preserves_unrelated(
        self, tmp_path, monkeypatch, session
    ):
        runtime = tmp_path / ("named-runtime" if session else "unnamed-runtime")
        runtime.mkdir()
        _write_owned_session_marker(runtime)
        targets_path = tmp_path / ("named-targets.json" if session else "unnamed-targets.json")
        targets_path.write_text(json.dumps(["owned-target", "user-target"]))
        target_file = runtime / ".hermes-target.json"
        target_file.write_text(
            json.dumps(
                {
                    "managed_by": "hermes-browser-use",
                    "instance_id": "fixture-instance",
                    "daemon_pid": "4242",
                    "origin": (
                        "harness-dedicated" if session else "hermes-created"
                    ),
                    "target_id": "owned-target",
                }
            )
        )
        monkeypatch.setenv("FIXTURE_TARGETS_PATH", str(targets_path))
        cli = _python_cli(
            tmp_path,
            """
            import inspect, json, os, pathlib, sys, types
            runtime = pathlib.Path(os.environ['BH_RUNTIME_DIR'])
            targets_path = pathlib.Path(os.environ['FIXTURE_TARGETS_PATH'])
            if '--reload' in sys.argv:
                for name in ('bu.sock', 'bu.port', 'bu.pid'):
                    (runtime / name).unlink(missing_ok=True)
                raise SystemExit(0)

            admin = types.ModuleType('browser_harness.admin')
            admin.NAME = os.environ.get('BU_NAME', 'default')
            admin._version = lambda: '0.1.10'
            def restart_daemon(name=None, require_clean=False):
                assert require_clean is True
                for endpoint in ('bu.sock', 'bu.port', 'bu.pid'):
                    (runtime / endpoint).unlink(missing_ok=True)
            admin.restart_daemon = restart_daemon

            ipc = types.ModuleType('browser_harness._ipc')
            class Connection:
                def close(self):
                    pass
            ipc.connect = lambda _name, timeout=5.0: (Connection(), None)
            ipc.request = lambda _conn, _token, payload: {
                'pong': True,
                'pid': os.getpid(),
                'browser_kind': 'cdp',
            } if payload.get('meta') == 'ping' else {'ok': True}

            package = types.ModuleType('browser_harness')
            package.admin = admin
            package._ipc = ipc
            sys.modules['browser_harness'] = package
            sys.modules['browser_harness.admin'] = admin
            sys.modules['browser_harness._ipc'] = ipc

            def cdp(method, **params):
                targets = json.loads(targets_path.read_text())
                if method == 'Target.getTargets':
                    return {'targetInfos': [
                        {'targetId': target, 'type': 'page'} for target in targets
                    ]}
                if method == 'Target.closeTarget':
                    targets.remove(params['targetId'])
                    targets_path.write_text(json.dumps(targets))
                    return {'success': True}
                raise AssertionError(method)

            code = sys.stdin.read()
            exec(code, {'cdp': cdp})
            """,
        )
        monkeypatch.setattr(bu_cli, "_TARGET_FILE", target_file.name, raising=False)

        ok, detail = bu_cli._reload_browser_use_session(
            {
                "runtime_dir": runtime,
                "command": cli,
                "session": session,
                "instance_id": "fixture-instance",
            }
        )

        assert ok is True, detail
        assert json.loads(targets_path.read_text()) == ["user-target"]
