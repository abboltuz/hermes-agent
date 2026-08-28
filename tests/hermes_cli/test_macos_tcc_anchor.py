"""Tests for the macOS TCC anchor (issue #85345).

The anchor makes the TCC client path stable by replacing the venv's
``bin/python`` symlink (which resolves into uv's versioned store) with a
real-file copy of the interpreter.  All tests run on Linux against fake
checkout/uv-store layouts; ``platform.system`` is monkeypatched to simulate
macOS.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_cli.doctor as doctor
import hermes_cli.macos_tcc_anchor as tcc
from hermes_constants import venv_python_path

_STORE_ROOT = "cpython-3.11.15-macos-aarch64-none"


def _darwin(monkeypatch):
    monkeypatch.setattr(tcc.platform, "system", lambda: "Darwin")


def _linux(monkeypatch):
    monkeypatch.setattr(tcc.platform, "system", lambda: "Linux")


@pytest.fixture
def fake_anchor_postconditions(monkeypatch):
    """Let fake interpreter bytes exercise layout logic without executing."""
    monkeypatch.setattr(tcc, "_macos_sign_managed_python", lambda _path: True)
    monkeypatch.setattr(
        tcc, "_macos_verify_managed_python_identity", lambda _path: True
    )
    monkeypatch.setattr(tcc, "_passes_boot_gate", lambda _path, _venv: True)


def _build_store(
    tmp_path, version: str = "3.11.15", *, with_libpython: bool = False
) -> Path:
    store = (
        tmp_path
        / "uv-store"
        / "uv"
        / "python"
        / f"cpython-{version}-macos-aarch64-none"
    )
    store_bin = store / "bin"
    store_bin.mkdir(parents=True)
    store_py = store_bin / "python3.11"
    store_py.write_bytes(f"#!fake interpreter {version}".encode())
    store_py.chmod(0o755)
    if with_libpython:
        store_lib = store / "lib"
        store_lib.mkdir()
        (store_lib / "libpython3.11.dylib").write_bytes(b"fake dylib")
    return store_bin


def _build_checkout(
    tmp_path,
    *,
    store_bin: Path | None = None,
    version: str = "3.11.15",
    anchored: bool = False,
    homebrew: bool = False,
) -> Path:
    root = tmp_path / "checkout"
    venv = root / ".venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    if homebrew:
        brew = tmp_path / "opt" / "homebrew" / "bin"
        brew.mkdir(parents=True)
        brew_py = brew / "python3.14"
        brew_py.write_bytes(b"#!homebrew")
        brew_py.chmod(0o755)
        (venv / "pyvenv.cfg").write_text(f"home = {brew}\n")
        os.symlink(brew_py, venv_bin / "python")
        os.symlink(brew_py, venv_bin / "python3")
        return root
    if store_bin is None:
        store_bin = _build_store(tmp_path, version)
    (venv / "pyvenv.cfg").write_text(f"home = {store_bin}\n")
    store_py = store_bin / "python3.11"
    if anchored:
        venv_py = venv_bin / "python"
        venv_py.write_bytes(store_py.read_bytes())
        venv_py.chmod(0o755)
        (venv_bin / ".tcc-anchor-source").write_text(str(store_py), encoding="utf-8")
        os.symlink(venv_py, venv_bin / "python3")
    else:
        os.symlink(store_py, venv_bin / "python")
        os.symlink(store_py, venv_bin / "python3")
    return root


class TestUvStoreDetection:
    def test_matches_uv_macos_store_path(self):
        path = (
            "/Users/u/.local/share/uv/python/"
            "cpython-3.11.15-macos-aarch64-none/bin/python3.11"
        )
        assert tcc._is_uv_macos_store(path)

    def test_matches_managed_runtime_repair_generation(self):
        path = (
            "/Users/u/hermes-agent/.hermes-runtime/python/"
            "generation-a1b2c3/cpython-3.11.15-macos-aarch64-none/bin/python3.11"
        )
        assert tcc._is_uv_macos_store(path)

    def test_rejects_homebrew_interpreter(self):
        path = (
            "/opt/homebrew/Cellar/python@3.14/3.14.6/Frameworks/"
            "Python.framework/Versions/3.14/bin/python3.14"
        )
        assert not tcc._is_uv_macos_store(path)

    def test_rejects_linux_interpreter(self):
        assert not tcc._is_uv_macos_store("/usr/bin/python3")

    def test_rejects_uv_store_on_linux(self):
        path = (
            "/home/u/.local/share/uv/python/"
            "cpython-3.11.15-x86_64-unknown-linux-gnu/bin/python3.11"
        )
        assert not tcc._is_uv_macos_store(path)


class TestEnsureTccAnchor:
    def test_noop_on_non_macos(self, tmp_path, monkeypatch):
        _linux(monkeypatch)
        root = _build_checkout(tmp_path, store_bin=_build_store(tmp_path))
        venv_py = venv_python_path(root / ".venv")

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()  # untouched

    def test_anchors_uv_managed_interpreter(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        venv_py = venv_python_path(root / ".venv")
        assert venv_py.is_symlink()  # preconditions: uv layout

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        # The venv interpreter is now a real file, not a symlink into the
        # versioned store — the TCC client path is stable.
        assert venv_py.is_file() and not venv_py.is_symlink()
        assert venv_py.read_bytes() == (store_bin / "python3.11").read_bytes()
        assert os.access(venv_py, os.X_OK)
        # Marker records the store binary the copy came from.
        marker = venv_py.parent / ".tcc-anchor-source"
        assert marker.read_text(encoding="utf-8").strip() == str(
            store_bin / "python3.11"
        )
        # Aliases are real files; symlinks reproduce the /install-prefix crash.
        alias = venv_py.parent / "python3"
        assert alias.is_file() and not alias.is_symlink()
        assert alias.read_bytes() == venv_py.read_bytes()

    def test_install_signs_canonical_and_validates_every_entrypoint(
        self, tmp_path, monkeypatch
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        signed = []
        verified = []
        booted = []
        monkeypatch.setattr(
            tcc,
            "_macos_sign_managed_python",
            lambda path: signed.append(Path(path)) or True,
        )
        monkeypatch.setattr(
            tcc,
            "_macos_verify_managed_python_identity",
            lambda path: verified.append(Path(path)) or True,
        )
        monkeypatch.setattr(
            tcc,
            "_passes_boot_gate",
            lambda path, _venv: booted.append(Path(path)) or True,
        )

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored is not None
        assert len(signed) == 1
        assert signed[0].parent == anchored.parent
        assert {path.name for path in verified} == {"python", "python3", "python3.11"}
        assert {path.name for path in booted} >= {"python", "python3", "python3.11"}

    def test_provisions_libpython_as_hardlink_when_present(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path, with_libpython=True)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        source = store_bin.parent / "lib" / "libpython3.11.dylib"

        assert tcc.ensure_tcc_anchor(root) is not None

        provisioned = root / ".venv" / "lib" / source.name
        assert provisioned.is_file()
        assert provisioned.read_bytes() == source.read_bytes()
        assert provisioned.stat().st_ino == source.stat().st_ino

    def test_boot_gate_refusal_leaves_original_interpreter_unmarked(
        self, tmp_path, monkeypatch
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        venv_py = venv_python_path(root / ".venv")
        monkeypatch.setattr(tcc, "_macos_sign_managed_python", lambda _path: True)
        monkeypatch.setattr(tcc, "_passes_boot_gate", lambda *_args: False)

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()
        assert not (venv_py.parent / ".tcc-anchor-source").exists()

    def test_signing_failure_is_fail_closed(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        root = _build_checkout(tmp_path, store_bin=_build_store(tmp_path))
        venv_py = venv_python_path(root / ".venv")
        monkeypatch.setattr(tcc, "_macos_sign_managed_python", lambda _path: False)

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()
        assert not (venv_py.parent / ".tcc-anchor-source").exists()

    def test_alias_failure_leaves_predecessor_unmarked(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin, anchored=True)
        venv_py = venv_python_path(root / ".venv")
        marker = venv_py.parent / ".tcc-anchor-source"
        assert marker.is_file()
        monkeypatch.setattr(tcc, "_copy_alias", lambda *_args, **_kwargs: False)

        assert tcc.ensure_tcc_anchor(root) is None
        assert not marker.exists()
        assert tcc.tcc_anchor_state(root)[0] != "active"

    def test_signing_failure_leaves_predecessor_unmarked(
        self, tmp_path, monkeypatch
    ):
        _darwin(monkeypatch)
        root = _build_checkout(
            tmp_path, store_bin=_build_store(tmp_path), anchored=True
        )
        venv_py = venv_python_path(root / ".venv")
        marker = venv_py.parent / ".tcc-anchor-source"
        monkeypatch.setattr(tcc, "_macos_sign_managed_python", lambda _path: False)
        monkeypatch.setattr(
            tcc, "_macos_verify_managed_python_identity", lambda _path: False
        )
        monkeypatch.setattr(tcc, "_passes_boot_gate", lambda *_args: True)

        assert tcc.ensure_tcc_anchor(root) is None
        assert not marker.exists()
        assert tcc.tcc_anchor_state(root)[0] != "active"

    def test_repairs_exact_predecessor_layout_and_false_active_state(
        self, tmp_path, monkeypatch
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin, anchored=True)
        venv_py = venv_python_path(root / ".venv")
        sign_started = False

        def sign(_path):
            nonlocal sign_started
            sign_started = True
            return True

        monkeypatch.setattr(tcc, "_macos_sign_managed_python", sign)
        monkeypatch.setattr(
            tcc,
            "_macos_verify_managed_python_identity",
            lambda _path: sign_started,
        )
        monkeypatch.setattr(tcc, "_passes_boot_gate", lambda *_args: True)

        assert (venv_py.parent / "python3").is_symlink()
        assert tcc.tcc_anchor_state(root)[0] != "active"

        assert tcc.ensure_tcc_anchor(root) == venv_py
        for name in ("python", "python3", "python3.11"):
            entrypoint = venv_py.parent / name
            assert entrypoint.is_file() and not entrypoint.is_symlink()
        assert tcc.tcc_anchor_state(root)[0] == "active"

    def test_idempotent(self, tmp_path, monkeypatch, fake_anchor_postconditions):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin, anchored=True)
        venv_py = venv_python_path(root / ".venv")
        marker = venv_py.parent / ".tcc-anchor-source"
        before = marker.read_text(encoding="utf-8")

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        assert venv_py.is_file() and not venv_py.is_symlink()
        assert marker.read_text(encoding="utf-8") == before

    def test_reanchors_after_patch_bump(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        old_bin = _build_store(tmp_path, version="3.11.15")
        root = _build_checkout(tmp_path, store_bin=old_bin, anchored=True)
        venv_py = venv_python_path(root / ".venv")

        # Simulate `uv sync` bumping 3.11.15 -> 3.11.16: uv re-links the venv
        # interpreter to the new store and rewrites pyvenv.cfg home.
        new_bin = _build_store(tmp_path, version="3.11.16")
        new_py = new_bin / "python3.11"
        venv_py.unlink()
        os.symlink(new_py, venv_py)
        (root / ".venv" / "pyvenv.cfg").write_text(f"home = {new_bin}\n")

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        assert not venv_py.is_symlink()
        assert venv_py.read_bytes() == new_py.read_bytes()
        marker = venv_py.parent / ".tcc-anchor-source"
        assert marker.read_text(encoding="utf-8").strip() == str(new_py)

    def test_skips_homebrew_interpreter(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        root = _build_checkout(tmp_path, homebrew=True)
        venv_py = venv_python_path(root / ".venv")

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()  # untouched: stable identity already

    def test_no_venv_returns_none(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        assert tcc.ensure_tcc_anchor(tmp_path / "missing") is None

    def test_preserves_stdlib_source_home(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        cfg = root / ".venv" / "pyvenv.cfg"

        tcc.ensure_tcc_anchor(root)

        # pyvenv.cfg still points stdlib at the uv store — the anchor only
        # changes the executable identity, not where the stdlib loads from.
        assert f"home = {store_bin}" in cfg.read_text(encoding="utf-8")


class TestBootGate:
    @staticmethod
    def _proc(argv, returncode, stdout="", stderr=""):
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    def test_accepts_encodings_and_matching_prefix(self, tmp_path, monkeypatch):
        venv = tmp_path / "venv"
        venv.mkdir()
        monkeypatch.setattr(
            tcc.subprocess,
            "run",
            lambda *args, **_kwargs: self._proc(args[0], 0, f"{venv}\n", ""),
        )

        assert tcc._passes_boot_gate(tmp_path / "staged", venv)

    @pytest.mark.parametrize(
        ("returncode", "stdout", "stderr"),
        [
            (1, "", "ModuleNotFoundError: encodings"),
            (0, "/install\n", ""),
            (0, "", ""),
        ],
    )
    def test_refuses_failed_or_wrong_prefix_process(
        self, tmp_path, monkeypatch, returncode, stdout, stderr
    ):
        monkeypatch.setattr(
            tcc.subprocess,
            "run",
            lambda *args, **_kwargs: self._proc(
                args[0], returncode, stdout, stderr
            ),
        )

        assert not tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

    def test_refuses_timeout_and_unexecutable_staging(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            tcc.subprocess,
            "run",
            lambda *args, **_kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(args[0], 30)
            ),
        )
        assert not tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

        monkeypatch.setattr(
            tcc.subprocess,
            "run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("not executable")
            ),
        )
        assert not tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

    def test_scrubs_python_environment(self, tmp_path, monkeypatch):
        captured = {}
        venv = tmp_path / "venv"
        venv.mkdir()

        def run(argv, **kwargs):
            captured.update(kwargs)
            return self._proc(argv, 0, f"{venv}\n", "")

        monkeypatch.setattr(tcc.subprocess, "run", run)
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__"):
            monkeypatch.setenv(key, "/poison")

        assert tcc._passes_boot_gate(tmp_path / "staged", venv)
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__"):
            assert key not in captured["env"]


class TestMarker:
    def test_normalizes_source_and_writes_atomically(self, tmp_path):
        resolved = tmp_path / "resolved" / "python3.11"
        resolved.parent.mkdir()
        resolved.write_bytes(b"python")
        spelling = tmp_path / "versionless" / "python3.11"
        spelling.parent.mkdir()
        os.symlink(resolved, spelling)
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)

        tcc._write_marker(venv_bin, spelling)

        marker = venv_bin / ".tcc-anchor-source"
        assert marker.read_text(encoding="utf-8") == str(resolved)
        assert not list(venv_bin.glob(".tcc-anchor-source.*"))


class TestAliasMaterialization:
    def test_copy_failure_warns_and_cleans_unique_staging(
        self, tmp_path, monkeypatch, caplog
    ):
        anchor = tmp_path / "python"
        anchor.write_bytes(b"python")
        anchor.chmod(0o755)

        def fail_copy(*_args, **_kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(tcc.shutil, "copy2", fail_copy)

        with caplog.at_level("WARNING", logger=tcc.__name__):
            assert not tcc._copy_alias(tmp_path, "python3", anchor)

        assert any("python3" in record.message for record in caplog.records)
        assert not list(tmp_path.glob(".python3.tcc-*"))


class TestTccAnchorState:
    def test_active_through_versionless_store_symlink(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        patched = _build_store(tmp_path, version="3.11.15")
        versionless = patched.parent.parent / "cpython-3.11-macos-aarch64-none"
        os.symlink(patched.parent, versionless)
        home = versionless / "bin"
        root = tmp_path / "checkout"
        venv = root / ".venv"
        venv_bin = venv / "bin"
        venv_bin.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(f"home = {home}\n", encoding="utf-8")
        os.symlink(home / "python3.11", venv_bin / "python")
        os.symlink("python", venv_bin / "python3")
        os.symlink("python", venv_bin / "python3.11")

        assert tcc.ensure_tcc_anchor(root) is not None
        assert tcc.tcc_anchor_state(root)[0] == "active"

    def test_state_missing_then_active(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)

        status, detail = tcc.tcc_anchor_state(root)
        assert status == "missing"
        assert str(venv_python_path(root / ".venv")) in detail

        tcc.ensure_tcc_anchor(root)

        status, detail = tcc.tcc_anchor_state(root)
        assert status == "active"

    def test_state_skip_on_linux(self, tmp_path, monkeypatch):
        _linux(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        status, detail = tcc.tcc_anchor_state(root)
        assert status == "skip"
        assert detail == "not macOS"

    def test_state_skip_for_homebrew(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        root = _build_checkout(tmp_path, homebrew=True)
        status, detail = tcc.tcc_anchor_state(root)
        assert status == "skip"
        assert "not uv-managed" in detail

    def test_state_stale_after_patch_bump(
        self, tmp_path, monkeypatch, fake_anchor_postconditions
    ):
        _darwin(monkeypatch)
        old_bin = _build_store(tmp_path, version="3.11.15")
        root = _build_checkout(tmp_path, store_bin=old_bin, anchored=True)
        # Simulate a patch bump where pyvenv.cfg now points at a new store
        # while the venv still holds the previous anchor copy.
        new_bin = _build_store(tmp_path, version="3.11.16")
        (root / ".venv" / "pyvenv.cfg").write_text(f"home = {new_bin}\n")
        status, _ = tcc.tcc_anchor_state(root)
        assert status == "stale"
        # ensure_tcc_anchor() refreshes the copy from the new interpreter.
        anchored = tcc.ensure_tcc_anchor(root)
        assert anchored == venv_python_path(root / ".venv")
        assert (root / ".venv" / "bin" / "python").read_bytes() == (
            new_bin / "python3.11"
        ).read_bytes()
        status, _ = tcc.tcc_anchor_state(root)
        assert status == "active"


class TestDoctorCheck:
    def test_missing_warns_without_fix(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("missing", "/x/.venv/bin/python")
        )
        doctor.check_macos_tcc_anchor(should_fix=False)
        out = capsys.readouterr().out
        assert "macOS TCC anchor missing" in out

    def test_fix_installs_anchor(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("missing", "/x/.venv/bin/python")
        )
        monkeypatch.setattr(
            tcc, "ensure_tcc_anchor", lambda *a, **k: Path("/x/.venv/bin/python")
        )
        doctor.check_macos_tcc_anchor(should_fix=True)
        out = capsys.readouterr().out
        assert "macOS TCC anchor installed" in out

    def test_active_reports_ok(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("active", "/x/.venv/bin/python")
        )
        doctor.check_macos_tcc_anchor(should_fix=False)
        out = capsys.readouterr().out
        assert "macOS TCC anchor active" in out

    def test_skip_is_silent_on_non_macos(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("skip", "not macOS")
        )
        doctor.check_macos_tcc_anchor(should_fix=False)
        assert capsys.readouterr().out == ""

    def test_never_crashes_on_exception(self, monkeypatch, capsys):
        def boom(*a, **k):
            raise RuntimeError("tccd down")

        monkeypatch.setattr(tcc, "tcc_anchor_state", boom)
        doctor.check_macos_tcc_anchor(should_fix=False)  # must not raise
        out = capsys.readouterr().out
        assert "macOS TCC anchor check failed" in out


@pytest.mark.macos_only
class TestAnchoredAliasesBootE2E:
    """Prove every venv interpreter entrypoint boots in a fresh process."""

    def test_python_aliases_and_console_script_boot_after_anchor(self, tmp_path):
        minor = f"python3.{sys.version_info.minor}"
        base = Path(sys.base_prefix)
        real_py = base / "bin" / minor
        if not real_py.is_file() or real_py.is_symlink():
            resolved = real_py.resolve() if real_py.exists() else None
            if resolved is None or not resolved.is_file():
                pytest.fail(f"no real base interpreter binary at {real_py}")
            real_py = resolved
        if not (base / "lib" / minor / "os.py").is_file():
            pytest.fail("base stdlib is not in the expected lib layout")

        store = (
            tmp_path
            / "uv"
            / "python"
            / f"cpython-{platform.python_version()}-macos-aarch64-none"
        )
        store_bin = store / "bin"
        store_bin.mkdir(parents=True)
        shutil.copy2(real_py, store_bin / minor)
        os.symlink(base / "lib", store / "lib")

        root = tmp_path / "checkout"
        venv = root / ".venv"
        venv_bin = venv / "bin"
        venv_bin.mkdir(parents=True)
        (venv / "lib" / minor / "site-packages").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            f"home = {store_bin}\nversion = {platform.python_version()}\n",
            encoding="utf-8",
        )
        os.symlink(store_bin / minor, venv_bin / "python")
        os.symlink("python", venv_bin / "python3")
        os.symlink("python", venv_bin / minor)

        console_script = venv_bin / "anchor-console-probe"
        console_script.write_text(
            f"#!{venv_bin / 'python3'}\n"
            "import encodings, sys\n"
            "print(sys.prefix)\n",
            encoding="utf-8",
        )
        console_script.chmod(0o755)

        anchored = tcc.ensure_tcc_anchor(root)
        assert anchored is not None
        for name in ("python", "python3", minor):
            entrypoint = venv_bin / name
            assert entrypoint.is_file() and not entrypoint.is_symlink()
            assert tcc._macos_verify_managed_python_identity(entrypoint)
        assert tcc.tcc_anchor_state(root)[0] == "active"

        env = dict(os.environ)
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__"):
            env.pop(key, None)
        entrypoints = [
            venv_bin / "python",
            venv_bin / "python3",
            venv_bin / minor,
            console_script,
        ]
        for entrypoint in entrypoints:
            argv = [str(entrypoint)]
            if entrypoint != console_script:
                argv.extend(["-c", "import encodings, sys; print(sys.prefix)"])
            probe = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            assert probe.returncode == 0, (
                f"{entrypoint.name} failed to boot after anchoring:\n{probe.stderr}"
            )
            assert Path(probe.stdout.strip()).resolve() == venv.resolve()
