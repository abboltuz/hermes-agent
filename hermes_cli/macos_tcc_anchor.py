"""Stable macOS TCC anchor for the uv-managed Python interpreter (#95596).

Re-land of the interpreter anchor reverted in #95563.  macOS keys TCC grants
to the resolved absolute path of the client binary.  Hermes' interpreter is
managed by uv and lives at a versioned store path; every patch bump orphans
every prior grant (#85345).

The first landing copied the interpreter into ``venv/bin/python`` but left
two holes that bricked real Macs:

* Dynamically-linked builds look up ``libpython`` via
  ``@executable_path/../lib``.  That resolved into ``venv/lib/``, which had
  no dylib — every hermes command, including update/doctor, died in dyld
  (#95425).
* Alias names (``python3``, ``python3.N``) were re-pointed at the copy as
  *symlinks*.  Invoking the copied interpreter through a symlink makes
  CPython getpath lose the venv prefix on affected python-build-standalone
  builds — startup dies with ``ModuleNotFoundError: encodings`` and the
  stdlib resolves to the build-time ``/install`` prefix (#95541).  Console
  scripts exec ``python3``, so the entire CLI surface died.

This re-land keeps the copy + identifier-pinned signature (TCC attribution
stays on the stable venv path) and closes both holes:

1. Aliases are materialized as real-file copies of the anchor, never
   symlinks.
2. If the store ships ``libpython*``, it is hardlinked into ``venv/lib/``
   (copy if the store is on another device).  Existing ``LC_RPATH`` already
   points at ``@executable_path/../lib`` — no rewrite.
3. A pre-install boot gate actually launches the staged copy and demands
   ``import encodings`` plus ``sys.prefix == <venv>``.  Every live executable
   and dylib is snapshotted before mutation, aliases are promoted while the
   original canonical route is still live, and the canonical executable is
   promoted last.  Any failed postcondition restores the complete predecessor
   layout, so a bad anchor can never brick update/doctor again.

All functions are no-ops on non-macOS and for interpreters that are not
uv-managed.  Best-effort: never raises to callers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterator

from hermes_constants import venv_python_path
from hermes_cli.managed_uv import (
    _RUNTIME_DIR_NAME,
    _macos_sign_managed_python,
    _macos_verify_managed_python_identity,
)
from utils import atomic_write_text

logger = logging.getLogger(__name__)

_MARKER_NAME = ".tcc-anchor-source"
_LOCK_NAME = ".tcc-anchor.lock"

_STORE_COMMON_MARKERS = ("cpython-", "-macos-")
# The runtime-store marker is derived from managed_uv so a rename of the
# repair-generation directory cannot silently stop the anchor from matching.
_STORE_ROOT_MARKERS = ("/uv/python/", f"/{_RUNTIME_DIR_NAME}/python/")
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class _BootGateFailed(Exception):
    """Staged copy refused to boot; the live venv must stay untouched."""


class _AnchorInstallFailed(Exception):
    """A fail-closed anchor postcondition was not satisfied."""


def _marker_value(source_file: Path) -> str:
    """Canonical marker value: fully resolved so symlinked spellings of the
    same store binary (``cpython-3.11-macos-*`` → ``cpython-3.11.15-macos-*``)
    compare equal."""
    return os.path.realpath(str(source_file))


def is_macos() -> bool:
    return platform.system() == "Darwin"


def _store_bin_names() -> tuple[str, ...]:
    """Unversioned interpreter aliases inside a store ``bin`` dir."""
    return ("python3", "python")


def _is_uv_macos_store(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if not all(marker in normalized for marker in _STORE_COMMON_MARKERS):
        return False
    return any(marker in normalized for marker in _STORE_ROOT_MARKERS)


def _venv_dir(project_root: Path | None = None) -> Path | None:
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    for name in ("venv", ".venv"):
        candidate = root / name
        venv_py = venv_python_path(candidate)
        if venv_py.is_file() or venv_py.is_symlink():
            return candidate
    return None


def _interpreter_file(src: str | Path) -> Path | None:
    """Return the interpreter binary file at/inside *src*."""
    p = Path(src)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    for name in _store_bin_names():
        candidate = p / name
        if candidate.is_file():
            return candidate
    try:
        for candidate in sorted(p.glob("python3.*")):
            if candidate.is_file() and not candidate.name.endswith((".dSYM", ".txt")):
                return candidate
    except OSError:
        return None
    return None


def _interpreter_source(venv_dir: Path) -> str | None:
    """Return the interpreter file the venv currently resolves to."""
    venv_py = venv_python_path(venv_dir)
    if venv_py.is_symlink():
        try:
            resolved = venv_py.resolve(strict=False)
        except OSError:
            return None
        return str(resolved)
    cfg = venv_dir / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    home = ""
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("home"):
                _, _, home = line.partition("=")
                home = home.strip()
                break
    except OSError:
        return None
    if not home:
        return None
    interp = _interpreter_file(home)
    return str(interp) if interp is not None else None


def _anchor_marker(venv_bin: Path) -> Path:
    return venv_bin / _MARKER_NAME


@contextmanager
def _anchor_transaction_lock(venv_dir: Path) -> Iterator[None]:
    """Serialize anchor inspection and mutation across threads and processes."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - tests can simulate Darwin on Windows
        fcntl = None

    lock_path = venv_dir / "bin" / _LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.path.realpath(str(lock_path))
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _write_marker(venv_bin: Path, source_file: Path) -> None:
    """Write the anchor marker atomically via the shared helper.

    A concurrent ensure (update + doctor --fix) must never observe a
    partially-written marker: a torn read would compare unequal and trigger
    a spurious reinstall, and ``write_text`` alone is not atomic.
    """
    atomic_write_text(
        _anchor_marker(venv_bin),
        _marker_value(source_file),
        tmp_prefix=f"{_MARKER_NAME}.",
    )


def _store_root(source_file: Path) -> Path:
    # .../cpython-<ver>-macos-*/bin/python3.N → store root
    return source_file.resolve(strict=False).parent.parent


def _target_minor_alias(source_file: Path) -> str | None:
    """Return the versioned alias named by the target interpreter."""
    try:
        name = source_file.resolve(strict=False).name
    except OSError:
        name = source_file.name
    return name if re.fullmatch(r"python3\.\d+", name) else None


def _alias_names(venv_bin: Path, source_file: Path) -> list[str]:
    """Return required and already-present Python aliases."""
    names = {"python3"}
    target_minor = _target_minor_alias(source_file)
    if target_minor is not None:
        names.add(target_minor)
    try:
        names.update(
            path.name
            for path in venv_bin.glob("python3*")
            if re.fullmatch(r"python3(\.\d+)?", path.name)
        )
    except OSError:
        pass
    return sorted(names)


def _libpython_targets(venv_dir: Path, source_file: Path) -> list[tuple[Path, Path]]:
    """Return source/destination pairs required by the target interpreter."""
    src_lib = _store_root(source_file) / "lib"
    if not src_lib.is_dir():
        return []
    try:
        return [
            (src, venv_dir / "lib" / src.name)
            for src in sorted(src_lib.glob("libpython*"))
            if src.is_file()
        ]
    except OSError:
        return []


@dataclass
class _PathSnapshot:
    path: Path
    kind: str
    backup: Path | None = None
    link_target: str | None = None


def _snapshot_path(path: Path) -> _PathSnapshot:
    """Capture a live entry without mutating it."""
    if path.is_symlink():
        return _PathSnapshot(path, "symlink", link_target=os.readlink(path))
    if not path.exists():
        return _PathSnapshot(path, "missing")
    fd, backup_name = tempfile.mkstemp(
        prefix=f".{path.name}.tcc-backup-", dir=str(path.parent)
    )
    os.close(fd)
    backup = Path(backup_name)
    backup.unlink()
    try:
        try:
            os.link(path, backup)
        except OSError:
            shutil.copy2(path, backup)
        return _PathSnapshot(path, "file", backup=backup)
    except Exception:
        backup.unlink(missing_ok=True)
        raise


def _snapshot_paths(paths: list[Path]) -> list[_PathSnapshot]:
    """Capture each distinct live path, cleaning partial backups on failure."""
    snapshots: list[_PathSnapshot] = []
    seen: set[Path] = set()
    try:
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            snapshots.append(_snapshot_path(path))
        return snapshots
    except Exception:
        _discard_snapshots(snapshots)
        raise


def _restore_snapshot(snapshot: _PathSnapshot) -> None:
    if snapshot.kind == "file":
        if snapshot.backup is None:
            raise OSError(f"missing backup for {snapshot.path}")
        os.replace(snapshot.backup, snapshot.path)
        snapshot.backup = None
        return
    if snapshot.kind == "symlink":
        fd, staging_name = tempfile.mkstemp(
            prefix=f".{snapshot.path.name}.tcc-restore-",
            dir=str(snapshot.path.parent),
        )
        os.close(fd)
        staging = Path(staging_name)
        staging.unlink()
        try:
            os.symlink(snapshot.link_target or "", staging)
            os.replace(staging, snapshot.path)
        finally:
            staging.unlink(missing_ok=True)
        return
    snapshot.path.unlink(missing_ok=True)


def _restore_snapshots(snapshots: list[_PathSnapshot]) -> bool:
    ok = True
    for snapshot in reversed(snapshots):
        try:
            _restore_snapshot(snapshot)
        except OSError:
            ok = False
            logger.error("TCC anchor rollback failed for %s", snapshot.path, exc_info=True)
    _discard_snapshots(snapshots)
    return ok


def _discard_snapshots(snapshots: list[_PathSnapshot]) -> None:
    for snapshot in snapshots:
        if snapshot.backup is not None:
            try:
                snapshot.backup.unlink(missing_ok=True)
            except OSError:
                logger.debug("could not remove TCC backup %s", snapshot.backup)
            snapshot.backup = None


def _provision_libpython(
    venv_dir: Path, source_file: Path, *, refresh: bool = False
) -> bool:
    """Atomically hardlink (else copy) store ``libpython*`` into the venv.

    Every replacement is fully staged before any live dylib is touched.  A
    promotion failure restores the complete predecessor set.
    """
    pairs = [
        (src, dst)
        for src, dst in _libpython_targets(venv_dir, source_file)
        if refresh or not (dst.exists() or dst.is_symlink())
    ]
    if not pairs:
        return True
    dst_lib = venv_dir / "lib"
    staged: list[tuple[Path, Path]] = []
    snapshots: list[_PathSnapshot] = []
    try:
        dst_lib.mkdir(parents=True, exist_ok=True)
        snapshots = _snapshot_paths([dst for _src, dst in pairs])
        for src, dst in pairs:
            fd, staging_name = tempfile.mkstemp(
                prefix=f".{dst.name}.tcc-", dir=str(dst_lib)
            )
            os.close(fd)
            staging = Path(staging_name)
            staging.unlink()
            staged.append((staging, dst))
            try:
                os.link(src, staging)
            except OSError:
                shutil.copy2(src, staging)
        for staging, dst in staged:
            os.replace(staging, dst)
        _discard_snapshots(snapshots)
        return True
    except OSError as exc:
        logger.warning("libpython provision failed", exc_info=True)
        if not _restore_snapshots(snapshots):
            raise _AnchorInstallFailed(
                "libpython provisioning failed and rollback was incomplete"
            ) from exc
        return False
    except BaseException as exc:
        if not _restore_snapshots(snapshots):
            raise _AnchorInstallFailed(
                "libpython provisioning was interrupted and rollback was incomplete"
            ) from exc
        raise
    finally:
        for staging, _dst in staged:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass


def _copy_alias(venv_bin: Path, name: str, anchor: Path) -> bool:
    """Materialize *name* as a real-file copy of *anchor* (atomic rename).

    Returns False (and warns) on failure: a leftover alias *symlink* to the
    anchor is the exact #95541 crash shape, so callers must know when the
    alias set is incomplete.  The staging name is unique (mkstemp) so a
    concurrent ensure (update + doctor --fix) cannot promote a truncated
    interim copy.
    """
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.tcc-", dir=str(venv_bin))
        os.close(fd)
        tmp_path = Path(tmp_name)
        shutil.copy2(anchor, tmp_path)
        os.chmod(tmp_path, anchor.stat().st_mode | 0o111)
        os.replace(tmp_path, venv_bin / name)
        return True
    except OSError as exc:
        logger.warning("TCC anchor alias %s not materialized: %s", name, exc)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _passes_boot_gate(staged: Path, venv_dir: Path) -> bool:
    """Launch *staged* and demand encodings + the venv prefix.

    The probe runs with Python path overrides scrubbed: an inherited value can
    paper over exactly the prefix-resolution failure the gate exists to catch.
    Any launch error fails closed because an unexecuted probe proves nothing.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "__PYVENV_LAUNCHER__",
        )
    }
    try:
        proc = subprocess.run(
            [str(staged), "-c", "import encodings, sys; print(sys.prefix)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except OSError as exc:
        logger.warning("boot gate: staged copy not executable: %s", exc)
        return False
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    printed = (proc.stdout or "").strip().splitlines()
    if not printed:
        return False
    try:
        return Path(printed[-1]).resolve() == venv_dir.resolve()
    except OSError:
        return False


def _alias_paths(venv_bin: Path, source_file: Path) -> list[Path]:
    """Return every required/current Python alias in deterministic order."""
    return [venv_bin / name for name in _alias_names(venv_bin, source_file)]


def _libpython_layout_complete(venv_dir: Path, source_file: Path) -> bool:
    """Require every store libpython artifact in the venv dylib search path."""
    src_lib = _store_root(source_file) / "lib"
    if not src_lib.is_dir():
        return True
    try:
        required = [path for path in src_lib.glob("libpython*") if path.is_file()]
        return all((venv_dir / "lib" / path.name).is_file() for path in required)
    except OSError:
        return False


def _anchor_layout_complete(
    venv_dir: Path,
    source_file: Path,
    *,
    require_marker: bool,
) -> bool:
    """Validate structure, fresh startup, stable DR, dylibs, and marker."""
    anchor = venv_python_path(venv_dir)
    try:
        if anchor.is_symlink() or not anchor.is_file() or not os.access(anchor, os.X_OK):
            return False
        if not _libpython_layout_complete(venv_dir, source_file):
            return False
        if not _macos_verify_managed_python_identity(anchor):
            return False
        if not _passes_boot_gate(anchor, venv_dir):
            return False
        for alias in _alias_paths(anchor.parent, source_file):
            if alias.is_symlink() or not alias.is_file() or not os.access(alias, os.X_OK):
                return False
            if not _macos_verify_managed_python_identity(alias):
                return False
            if not _passes_boot_gate(alias, venv_dir):
                return False
        if require_marker:
            marker = _anchor_marker(anchor.parent)
            if not marker.is_file():
                return False
            if marker.read_text(encoding="utf-8").strip() != _marker_value(source_file):
                return False
        return True
    except OSError:
        return False


def _install_anchor(venv_dir: Path, source_file: Path) -> None:
    """Install a complete anchor with bounded predecessor rollback."""
    venv_py = venv_python_path(venv_dir)
    venv_bin = venv_py.parent
    venv_bin.mkdir(parents=True, exist_ok=True)
    marker = _anchor_marker(venv_bin)
    aliases = _alias_paths(venv_bin, source_file)
    dylibs = [dst for _src, dst in _libpython_targets(venv_dir, source_file)]
    snapshots = _snapshot_paths([venv_py, *aliases, *dylibs])
    tmp_path: Path | None = None
    alias_stage_dir: Path | None = None
    try:
        marker.unlink(missing_ok=True)
        if not _provision_libpython(venv_dir, source_file, refresh=True):
            raise _AnchorInstallFailed("libpython provisioning was incomplete")

        fd, tmp_name = tempfile.mkstemp(prefix=".python-tcc-", dir=str(venv_bin))
        os.close(fd)
        tmp_path = Path(tmp_name)
        shutil.copy2(source_file, tmp_path)
        os.chmod(tmp_path, source_file.stat().st_mode | 0o111)
        if not _macos_sign_managed_python(tmp_path):
            raise _AnchorInstallFailed("stable managed-Python signing failed")
        if not _passes_boot_gate(tmp_path, venv_dir):
            raise _BootGateFailed(
                f"staged copy at {tmp_path} failed encodings/prefix probe"
            )
        # Stage each alias under its final basename one directory below the
        # venv root.  That preserves both CPython's pyvenv.cfg discovery and
        # @executable_path/../lib while keeping codesign away from live paths.
        alias_stage_dir = Path(
            tempfile.mkdtemp(prefix=".python-aliases-tcc-", dir=str(venv_dir))
        )
        for alias in aliases:
            staged_alias = alias_stage_dir / alias.name
            if not _copy_alias(alias_stage_dir, alias.name, tmp_path):
                raise _AnchorInstallFailed(
                    f"alias materialization failed: {alias.name}"
                )
            if not _macos_sign_managed_python(staged_alias):
                raise _AnchorInstallFailed(f"alias signing failed: {alias.name}")
            if not _macos_verify_managed_python_identity(staged_alias):
                raise _AnchorInstallFailed(
                    f"alias identity verification failed: {alias.name}"
                )
            if not _passes_boot_gate(staged_alias, venv_dir):
                raise _AnchorInstallFailed(f"alias boot gate failed: {alias.name}")

        # Promote only complete, signed, verified, boot-tested aliases while
        # the original store-backed canonical route is still live.
        for alias in aliases:
            os.replace(alias_stage_dir / alias.name, alias)

        # Canonical promotion is the last executable cutover.
        os.replace(tmp_path, venv_py)
        tmp_path = None
        if not _anchor_layout_complete(
            venv_dir, source_file, require_marker=False
        ):
            raise _AnchorInstallFailed("installed anchor failed postcondition checks")
        # Marker is the activation record for the complete canonical + alias
        # layout.  It is written atomically and strictly last.
        _write_marker(venv_bin, source_file)
        _discard_snapshots(snapshots)
    except BaseException as exc:
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
        rollback_ok = _restore_snapshots(snapshots)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        if not rollback_ok:
            raise _AnchorInstallFailed(
                "anchor installation failed and predecessor rollback was incomplete"
            ) from exc
        raise
    finally:
        if alias_stage_dir is not None:
            shutil.rmtree(alias_stage_dir, ignore_errors=True)


def ensure_tcc_anchor(project_root: Path | None = None) -> Path | None:
    """Pin a dylib-complete interpreter anchor for macOS TCC (#95596).

    No-op (returns None) on non-macOS, when no venv interpreter exists, or
    when the interpreter is not uv-managed.  Idempotent.  Best-effort —
    returns None (and logs) if the copy or boot-gate fails; callers must
    never depend on success.
    """
    if not is_macos():
        return None
    venv_dir = _venv_dir(project_root)
    if venv_dir is None:
        return None
    try:
        with _anchor_transaction_lock(venv_dir):
            # Re-read every mutable input after lock acquisition.  Another
            # updater may have completed while this caller was waiting.
            venv_py = venv_python_path(venv_dir)
            if not (venv_py.is_file() or venv_py.is_symlink()):
                return None
            source = _interpreter_source(venv_dir)
            if source is None or not _is_uv_macos_store(source):
                return None
            source_file = _interpreter_file(source)
            if source_file is None:
                return None
            if not venv_py.is_symlink() and _anchor_layout_complete(
                venv_dir, source_file, require_marker=True
            ):
                return venv_py
            _install_anchor(venv_dir, source_file)
            return venv_py
    except _BootGateFailed as exc:
        logger.warning("macOS TCC anchor boot-gate refused install: %s", exc)
        return None
    except _AnchorInstallFailed as exc:
        logger.warning("macOS TCC anchor refused activation: %s", exc)
        return None
    except Exception as exc:  # best-effort: never break update/doctor
        logger.warning("macOS TCC anchor install failed: %s", exc)
        return None


def tcc_anchor_state(project_root: Path | None = None) -> tuple[str, str]:
    """Report the anchor state for ``hermes doctor``.

    Returns ``(status, detail)`` with status one of:

    - ``"skip"``    — not applicable (non-macOS, no venv, or not uv-managed)
    - ``"active"``  — venv interpreter is pinned at a stable real-file anchor
    - ``"stale"``   — pinned but the interpreter changed since the last copy
    - ``"missing"`` — uv-managed interpreter with no stable anchor installed
    """
    if not is_macos():
        return "skip", "not macOS"
    venv_dir = _venv_dir(project_root)
    if venv_dir is None:
        return "skip", "no venv interpreter"
    venv_py = venv_python_path(venv_dir)
    if not (venv_py.is_file() or venv_py.is_symlink()):
        return "skip", "no venv interpreter"
    source = _interpreter_source(venv_dir)
    if source is None or not _is_uv_macos_store(source):
        return "skip", "interpreter not uv-managed (stable path)"
    if not venv_py.is_symlink():
        source_file = _interpreter_file(source)
        if source_file is not None and _anchor_layout_complete(
            venv_dir, source_file, require_marker=True
        ):
            return "active", str(venv_py)
        return "stale", str(venv_py)
    return "missing", str(venv_py)
