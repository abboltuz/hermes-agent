"""Use the Browser Use CLI 3.0 (https://browser-use.com) for browser automation

When browser.backend is "browser-use", the model gets ``browser_exec`` tool
instead of default browser tools
"""

import atexit
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import is_truthy_value

logger = logging.getLogger(__name__)

_BACKEND_KEY = "browser-use"
BACKEND_DISABLED = "off"

# Cloud daemon names become the BU_NAME env var
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Internal marker set by _resolve_backend_cdp on the env dict when the
# resolved browser is EXCLUSIVE to this named session (per-name provider
# browser, or a named Browser Use cloud browser). Popped before the
# subprocess launches — never exported to the CLI.
_PRIVATE_BROWSER_SENTINEL = "_HERMES_BU_PRIVATE_BROWSER"

# Preamble prepended on SHARED browsers (local Chrome / CDP override). Named
# Harness daemons already create a dedicated tab, so Hermes claims that exact
# current target. The unnamed default needs an explicit task-owned target.
# Both paths persist the target id under the private runtime so cleanup can
# close exactly that target on Harness 0.1.9 and 0.1.10.
_OWN_TAB_PREAMBLE = """\
# hermes: claim one exact task/session tab (once per daemon process)
def _hermes_ensure_own_tab():
    import json as _json, os as _os
    from browser_harness import helpers as _helpers

    _target_file = _os.environ.get("HERMES_BH_TARGET_FILE")
    _instance_id = _os.environ.get("HERMES_BH_INSTANCE_ID")
    _mode = _os.environ.get("HERMES_BH_TARGET_MODE")
    if not _target_file or not _instance_id or _mode not in ("claim", "create"):
        raise RuntimeError("HERMES_BROWSER_CONTROL_WEDGED: missing owned-tab lifecycle metadata")
    _name = _os.environ.get("BU_NAME", "default")
    _tid = None
    _tmp = None
    try:
        from browser_harness import _ipc as _bipc
        _dpid = _bipc.pid_path(_name).read_text().strip() or "0"
        if _dpid == "0":
            raise RuntimeError("empty Browser Harness daemon pid")
    except Exception as _exc:
        raise RuntimeError(
            "HERMES_BROWSER_CONTROL_WEDGED: could not verify owned-tab daemon: "
            + str(_exc)
        ) from _exc

    _existing = None
    try:
        with open(_target_file, encoding="utf-8") as _fh:
            _candidate = _json.load(_fh)
        if (
            isinstance(_candidate, dict)
            and _candidate.get("managed_by") == "hermes-browser-use"
            and _candidate.get("instance_id") == _instance_id
            and isinstance(_candidate.get("target_id"), str)
        ):
            _existing = _candidate
    except Exception:
        _existing = None

    try:
        _targets = cdp("Target.getTargets").get("targetInfos", [])
        _target_ids = {
            _item.get("targetId") for _item in _targets
            if isinstance(_item, dict) and _item.get("type") == "page"
        }
        if (
            _existing
            and _existing.get("daemon_pid") == _dpid
            and _existing["target_id"] in _target_ids
        ):
            return
        if _existing and _existing["target_id"] in _target_ids:
            cdp("Target.closeTarget", targetId=_existing["target_id"])

        if _mode == "claim":
            _reply = _helpers._send({"meta": "current_tab"})
            _tid = _reply.get("targetId") if isinstance(_reply, dict) else None
            _origin = "harness-dedicated"
        else:
            _tid = cdp(
                "Target.createTarget", url="about:blank", background=True
            ).get("targetId")
            _origin = "hermes-created"
        if not _tid:
            raise RuntimeError("Browser Harness returned no target id")

        _payload = {
            "managed_by": "hermes-browser-use",
            "instance_id": _instance_id,
            "daemon_pid": _dpid,
            "target_id": _tid,
            "origin": _origin,
        }
        _tmp = "%s.%s.tmp" % (_target_file, _os.getpid())
        with open(_tmp, "w", encoding="utf-8") as _fh:
            _json.dump(_payload, _fh, sort_keys=True)
        _os.replace(_tmp, _target_file)
        if _mode == "create":
            switch_tab(_tid)
    except Exception as _exc:
        if _tmp:
            try:
                _os.unlink(_tmp)
            except OSError:
                pass
        if _tid:
            try:
                cdp("Target.closeTarget", targetId=_tid)
            except Exception:
                pass
        raise RuntimeError(
            "HERMES_BROWSER_CONTROL_WEDGED: could not establish exact owned tab: "
            + str(_exc)
        ) from _exc
_hermes_ensure_own_tab()
del _hermes_ensure_own_tab
"""

# A daemon can outlive the CLI package that spawned it. Record strict Cloud
# shutdown authority only when Hermes observed this exact private runtime with
# no daemon endpoint immediately before the CLI invocation. The preamble runs
# after Harness has spawned/connected, so it can bind the spawning package
# version to the daemon's exact PID without changing the external package.
_DAEMON_CONTRACT_PREAMBLE = """\
# hermes: bind daemon cleanup capability to the exact spawned process
def _hermes_record_daemon_contract():
    import json as _json, os as _os

    if _os.environ.get("HERMES_BH_RECORD_DAEMON_CONTRACT") != "1":
        return
    _contract_file = _os.environ.get("HERMES_BH_DAEMON_CONTRACT_FILE")
    _instance_id = _os.environ.get("HERMES_BH_INSTANCE_ID")
    if not _contract_file or not _instance_id:
        raise RuntimeError(
            "HERMES_BROWSER_CONTROL_WEDGED: missing daemon-contract metadata"
        )
    try:
        from browser_harness import _ipc as _bipc
        from browser_harness import admin as _badmin

        _name = _os.environ.get("BU_NAME", "default")
        _daemon_pid = _bipc.pid_path(_name).read_text().strip()
        if (
            not _daemon_pid.isdigit()
            or int(_daemon_pid) <= 0
            or int(_daemon_pid) >= (1 << 31)
        ):
            raise RuntimeError("invalid Browser Harness daemon pid")
        _daemon_started = _badmin._process_start_time(int(_daemon_pid))
        if _daemon_started is None:
            raise RuntimeError("could not fingerprint Browser Harness daemon process")
        _version = str(_badmin._version() or "unknown")
        try:
            _parts = tuple(int(_part) for _part in _version.split(".")[:3])
        except (TypeError, ValueError):
            _parts = ()
        _payload = {
            "managed_by": "hermes-browser-use",
            "instance_id": _instance_id,
            "daemon_pid": _daemon_pid,
            "daemon_started": _daemon_started,
            "harness_version": _version,
            "strict_cloud_shutdown": _parts >= (0, 1, 10),
        }
        _tmp = "%s.%s.tmp" % (_contract_file, _os.getpid())
        with open(_tmp, "w", encoding="utf-8") as _fh:
            _json.dump(_payload, _fh, sort_keys=True)
        if _os.name != "nt":
            _os.chmod(_tmp, 0o600)
        _os.replace(_tmp, _contract_file)
    except Exception as _exc:
        raise RuntimeError(
            "HERMES_BROWSER_CONTROL_WEDGED: could not bind cleanup authority "
            "to the exact Browser Harness daemon: " + str(_exc)
        ) from _exc
_hermes_record_daemon_contract()
del _hermes_record_daemon_contract
"""

_DEFAULT_TIMEOUT_S = 300
_MIN_TIMEOUT_S = 5
_MAX_TIMEOUT_S = 1800
_STDERR_CAP_CHARS = 4000
_DIALOG_RELEASE_TIMEOUT_S = 5.0
_LIFECYCLE_CHECK_INTERVAL_S = 30.0
_LIFECYCLE_RELOAD_TIMEOUT_S = 70.0
_LIFECYCLE_VERIFY_TIMEOUT_S = 3.0
_OWNER_FILE = ".hermes-owner.json"
_SESSION_FILE = ".hermes-session.json"
_TARGET_FILE = ".hermes-target.json"
_DAEMON_FILE = ".hermes-daemon.json"
_REAP_LOCK_FILE = ".hermes-reap.lock"
_CLEANUP_RESULT_PREFIX = "[hermes-browser-cleanup] "

# Browser Harness 0.1.9/0.1.10 waits synchronously for mouseReleased even
# when that event opened a native JavaScript dialog.  The daemon can still
# answer its pending-dialog metadata request on another IPC connection, so
# dispatch release on a daemon thread and reconcile the two outcomes.  This
# is a caller-side compatibility layer over Harness's existing CDP + IPC
# protocol; it does not synthesize a DOM click or weaken trusted hit-testing.
def _dialog_safe_click_preamble(
    timeout_s: float = _DIALOG_RELEASE_TIMEOUT_S,
) -> str:
    timeout = max(0.05, float(timeout_s))
    return f'''\
# hermes: keep trusted CDP clicks responsive around native JS dialogs
import json as _hermes_json
import threading as _hermes_threading
import time as _hermes_time
from browser_harness import helpers as _hermes_bh_helpers
_hermes_stock_click_at_xy = click_at_xy
_hermes_stock_helper_cdp = getattr(_hermes_bh_helpers, "cdp", cdp)

def _hermes_pending_dialog():
    try:
        _send = getattr(_hermes_bh_helpers, "_send", None)
        if callable(_send):
            _reply = _send({{"meta": "pending_dialog"}})
            return _reply.get("dialog") if isinstance(_reply, dict) else None
        _info = page_info()
        return _info.get("dialog") if isinstance(_info, dict) else None
    except Exception as _exc:
        raise RuntimeError(
            "HERMES_BROWSER_CONTROL_WEDGED: Browser Harness IPC could not "
            "inspect the pending JavaScript dialog: " + str(_exc)
        ) from _exc

def _hermes_dialog_notice(_dialog):
    _notice = {{
        "status": "dialog_pending",
        "dialog": _dialog,
        "next_action": (
            "Inspect the dialog, then call "
            "cdp('Page.handleJavaScriptDialog', accept=True/False, "
            "promptText='...') before continuing."
        ),
    }}
    print("[hermes-browser] " + _hermes_json.dumps(_notice, ensure_ascii=False))
    return _notice

def click_at_xy(x, y, button="left", clicks=1):
    _already_pending = _hermes_pending_dialog()
    if _already_pending:
        raise RuntimeError(
            "A JavaScript dialog is already pending; handle it with "
            "Page.handleJavaScriptDialog before sending another click."
        )
    _state = {{"dialog": None}}

    def _safe_cdp(_method, *args, **params):
        if _method != "Input.dispatchMouseEvent":
            return _hermes_stock_helper_cdp(_method, *args, **params)
        _event_type = params.get("type")
        if _event_type == "mousePressed":
            try:
                _result = _hermes_stock_helper_cdp(_method, *args, **params)
            except BaseException as _exc:
                raise RuntimeError(
                    "HERMES_BROWSER_INPUT_WEDGED: "
                    "Input.dispatchMouseEvent(mousePressed) failed: " + str(_exc)
                ) from _exc
            _state["dialog"] = _hermes_pending_dialog()
            return _result
        if _event_type != "mouseReleased":
            return _hermes_stock_helper_cdp(_method, *args, **params)
        if _state["dialog"]:
            return {{}}

        _done = _hermes_threading.Event()
        _release_error = []

        def _release():
            try:
                _hermes_stock_helper_cdp(_method, *args, **params)
            except BaseException as _exc:
                _release_error.append(_exc)
            finally:
                _done.set()

        _hermes_threading.Thread(
            target=_release, name="hermes-browser-mouse-release", daemon=True,
        ).start()
        _deadline = _hermes_time.monotonic() + {timeout!r}
        while _hermes_time.monotonic() < _deadline:
            if _done.wait(0.02):
                if _release_error:
                    raise RuntimeError(
                        "HERMES_BROWSER_INPUT_WEDGED: "
                        "Input.dispatchMouseEvent(mouseReleased) failed: "
                        + str(_release_error[0])
                    ) from _release_error[0]
                return {{}}
            _dialog = _hermes_pending_dialog()
            if _dialog:
                _state["dialog"] = _dialog
                return {{}}

        raise RuntimeError(
            "HERMES_BROWSER_INPUT_WEDGED: "
            "Input.dispatchMouseEvent(mouseReleased) did not complete within "
            "{timeout:.2f}s and no JavaScript dialog was reported; the Chrome "
            "CDP Input domain may be wedged. The exact Hermes-owned Harness "
            "daemon will be stopped before this result is returned."
        )

    # Harness installs its trace wrapper before exec(). Call that exact stock
    # helper so recorder.observe('click_at_xy', ...), BH_DEBUG_CLICKS, and any
    # compatible helper extension still run. Only its CDP dispatch seam is
    # temporarily made dialog-safe for this call.
    _previous_helper_cdp = getattr(_hermes_bh_helpers, "cdp", None)
    _hermes_bh_helpers.cdp = _safe_cdp
    try:
        _result = _hermes_stock_click_at_xy(
            x, y, button=button, clicks=clicks
        )
    finally:
        if _previous_helper_cdp is None:
            try:
                delattr(_hermes_bh_helpers, "cdp")
            except AttributeError:
                pass
        else:
            _hermes_bh_helpers.cdp = _previous_helper_cdp
    if _state["dialog"]:
        return _hermes_dialog_notice(_state["dialog"])
    return _result

click_at_xy.__bh_traced__ = bool(
    getattr(_hermes_stock_click_at_xy, "__bh_traced__", False)
)
click_at_xy.__wrapped__ = _hermes_stock_click_at_xy

'''

# Filesystem-safe task ids for per-task workspace dirs.
_TASK_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Screenshot paths printed by capture_screenshot() in the exec output.
# Two alternatives: POSIX absolute (/tmp/shot.png) and Windows drive-letter
# absolute (C:\Users\...\shot.png or C:/Users/.../shot.png). Browser Use on
# Windows prints native paths — the POSIX-only pattern silently dropped them
# and screenshot_path / the multimodal attach never fired (#83884).
_IMAGE_PATH_RE = re.compile(
    r"((?:[A-Za-z]:[\\/]|/)[^\s\"']+?\.(?:png|jpe?g|webp))", re.IGNORECASE
)

# http(s) URL literals in exec code checked against browser_navigate's policy
_URL_RE = re.compile(r"https?://[^\s'\"\\)]+", re.IGNORECASE)


def _blocked_url_in_code(code: str) -> Optional[str]:
    """Return an error if a URL literal fails the built-in navigation checks."""
    from tools.browser_tool import evaluate_url_safety

    for url in _URL_RE.findall(code or ""):
        err = evaluate_url_safety(url)
        if err:
            return err.get("error", "Blocked: unsafe URL")
    return None


def _base_subprocess_env() -> dict:
    from tools.browser_tool import _build_browser_env

    env = _build_browser_env()
    # The browser-use CLI runs under its own Python (uv tool / uvx), which
    # may differ from Hermes's venv Python. PYTHONPATH/PYTHONHOME inherited
    # from the agent process point at Hermes's venv site-packages, and a
    # child interpreter honors them ahead of its own site-packages — so the
    # CLI imports compiled C-extensions (e.g. pydantic_core) built for the
    # wrong interpreter and crashes on ABI mismatch (#83427, #84841, #86006,
    # #86104). Strip both — the CLI manages its own environment and never
    # needs Hermes's import path.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Same class of hazard, PATH flavor: profile-spawned workers (kanban
    # bots, cron jobs) can hand down a PATH of only version-manager dirs,
    # which kills the uv trampoline before the CLI's Python starts. Floor
    # the PATH so coreutils are always reachable (see below).
    env["PATH"] = _floor_subprocess_path(env.get("PATH", ""))
    env.setdefault("ANONYMIZED_TELEMETRY", "false")
    return env


def _floor_subprocess_path(path: str) -> str:
    """Guarantee core system dirs survive onto the CLI subprocess PATH.

    Profile workers can inherit a PATH holding only version-manager dirs
    (observed: the nvm node dir repeated 7x, nothing else). That is fatal
    for the uv-installed browser-use binary: its POSIX sh trampoline
    resolves ``dirname``/``realpath`` through PATH, so without /usr/bin it
    dies with ``realpath: not found … exec: /python: not found`` (exit
    127) before its own Python ever starts. Reuses browser_tool's
    ``_merge_browser_path`` floor — same hazard, same sane-dir list — and
    falls back to appending FHS bin dirs if that import is unavailable.
    Windows .cmd shims don't trampoline through PATH, so no-op there.
    """
    if os.name == "nt":
        return path
    try:
        from tools.browser_tool import _merge_browser_path

        return _merge_browser_path(path or "")
    except Exception:
        pass
    parts = [p for p in (path or "").split(os.pathsep) if p]
    existing = set(parts)
    for directory in (
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ):
        if directory not in existing and os.path.isdir(directory):
            parts.append(directory)
    return os.pathsep.join(parts)


def _read_browser_cfg() -> dict:
    """Return the ``browser:`` config section, or {} on any failure."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        cfg = cfg_get(read_raw_config(), "browser", default={})
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.debug("Could not read browser config section: %s", e)
        return {}


def get_browser_backend() -> str:
    """Return the configured browser backend key ("" = unset → default).

    YAML 1.1 parses an unquoted ``off`` as boolean False — a hand-edited
    ``backend: off`` must mean BACKEND_DISABLED, not "unset". (True has no
    sensible backend meaning; normalize it to unset.)
    """
    raw = _read_browser_cfg().get("backend")
    if raw is False:
        return BACKEND_DISABLED
    if raw is True:
        return ""
    return str(raw or "").strip().lower()


def is_legacy_browser_use_cloud_config(browser_cfg: dict) -> bool:
    """True for pre-CLI direct-API Browser Use cloud configs"""
    if not isinstance(browser_cfg, dict):
        return False
    if browser_cfg.get("backend"):
        return False  # an explicit backend choice wins
    provider = str(browser_cfg.get("cloud_provider") or "").strip().lower()
    if provider not in {"browser-use", ""}:
        return False  # explicit local/Browserbase/… choices win
    if is_truthy_value(browser_cfg.get("use_gateway"), default=False):
        return False
    # Camofox is selected via env var, not cloud_provider — a Camofox user
    # with a stray BROWSER_USE_API_KEY must keep their explicit choice.
    try:
        from tools.browser_camofox import is_camofox_mode

        if is_camofox_mode():
            return False
    except Exception as e:
        logger.debug("Camofox activity check failed during migration: %s", e)
    return bool(os.getenv("BROWSER_USE_API_KEY"))


def is_browser_use_cli_mode() -> bool:
    """True when the Browser Use CLI replaces the built-in browser stack.

    Browser Use mode is the DEFAULT: an unset ``browser.backend`` ("") enables
    it whenever the browser-use CLI is runnable (installed binary or uvx).
    Set ``browser.backend: off`` (or ``/browser use off``) for the built-in
    browser_* tools.

    Camofox always falls back to the built-in tools regardless of
    ``browser.backend`` — it is Firefox-based with a custom HTTP API and no
    CDP surface, so the CDP-only browser-use harness cannot drive it.
    """
    try:
        from tools.browser_camofox import is_camofox_mode

        if is_camofox_mode():
            return False
    except Exception as e:
        logger.debug("Camofox activity check failed: %s", e)
    backend = get_browser_backend()
    if backend:
        return backend == _BACKEND_KEY
    if is_legacy_browser_use_cloud_config(_read_browser_cfg()):
        return True
    # Default (backend unset): Browser Use mode when the CLI can run at all;
    # otherwise keep the built-in tools so browsing never silently breaks.
    return _find_cli() is not None


_NOTICE_STAMP_NAME = ".browser_use_default_notice"
_NOTICE_INTERVAL_S = 24 * 3600


def default_downgrade_notice() -> Optional[str]:
    """One-line notice when the default Browser Use backend silently downgraded.

    Returns the notice string when ``browser.backend`` is unset (Browser Use
    would be the default) but the CLI is not runnable, so the session fell
    back to the built-in browser tools. Rate-limited to once per 24h via a
    stamp file so it nudges without nagging. Returns ``None`` otherwise.
    """
    try:
        if get_browser_backend():
            return None  # explicit choice — nothing downgraded
        try:
            from tools.browser_camofox import is_camofox_mode

            if is_camofox_mode():
                return None
        except Exception:
            pass
        if _find_cli() is not None:
            return None

        from hermes_constants import get_hermes_home

        stamp = Path(get_hermes_home()) / "cache" / _NOTICE_STAMP_NAME
        try:
            if 0 <= time.time() - stamp.stat().st_mtime < _NOTICE_INTERVAL_S:
                return None
        except OSError:
            pass
        try:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.touch()
        except OSError:
            pass
        return (
            "Browser Use CLI not found — using the built-in browser tools. "
            "Run `hermes tools` (Browser Automation → Browser Use) to install it, "
            "or `browser.backend: off` in config.yaml to silence this."
        )
    except Exception as e:  # pragma: no cover — a notice must never break startup
        logger.debug("browser-use downgrade notice failed: %s", e)
        return None


def _managed_bin_dir() -> Optional[str]:
    """Hermes' own bin dir ($HERMES_HOME/bin) — where install.sh puts uv/uvx
    and where install_cli() links the browser-use binary."""
    try:
        from hermes_constants import get_hermes_home

        return str(Path(get_hermes_home()) / "bin")
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("Could not resolve managed bin dir: %s", e)
        return None


def _user_local_bin_dir() -> Optional[str]:
    """The standard user-level tool dir (~/.local/bin on POSIX; uv's default
    tool bin dir on Windows). Desktop/TUI workers may start with a minimal
    PATH that omits it even when `uv tool install browser-use` put the
    binary there."""
    try:
        if os.name == "nt":
            base = os.environ.get("APPDATA")
            if base:
                return str(Path(base) / "uv" / "bin")
            return None
        return str(Path(os.path.expanduser("~")) / ".local" / "bin")
    except Exception as e:  # pragma: no cover — defensive
        logger.debug("Could not resolve user-local bin dir: %s", e)
        return None


def _find_cli() -> Optional[List[str]]:
    """Locate the browser-use CLI, or None when it can't be run.

    MANAGED-FIRST resolution: Hermes' own ``$HERMES_HOME/bin`` copy — the
    one every browser backend selection installs and updates via
    ``install_cli()`` — always wins, so all sessions drive one canonical,
    Hermes-controlled binary. PATH and the user-level tool dir
    (~/.local/bin / %APPDATA%\\uv\\bin, where a manual ``uv tool install``
    links binaries) are fallbacks for setups that never ran our install,
    and cover Desktop/TUI workers that spawn with a minimal PATH. The uvx
    zero-install path (same probe order) is the final fallback.
    """
    probe_paths = (_managed_bin_dir(), None, _user_local_bin_dir())
    for probe_path in probe_paths:
        if probe_path is None or probe_path:
            direct = shutil.which("browser-use", path=probe_path)
            if direct:
                return [direct]
    for probe_path in probe_paths:
        if probe_path is None or probe_path:
            uvx = shutil.which("uvx", path=probe_path)
            if uvx:
                return [uvx, "browser-use"]
    return None


def install_cli(timeout_s: int = 600) -> Tuple[bool, str]:
    """Install the browser-use CLI persistently via ``uv tool install``.

    Resolution order for uv: Hermes' managed uv (bootstrapped on demand via
    ``hermes_cli.managed_uv.ensure_uv``) → uv on PATH. The binary is linked
    into ``$HERMES_HOME/bin`` (``UV_TOOL_BIN_DIR``) so ``_find_cli()``
    resolves it for every profile without touching the user's PATH.

    Returns ``(ok, message)`` — never raises.
    """
    # MANAGED-FIRST: only the managed copy short-circuits the install. A
    # browser-use found on PATH is a user-level side install — it must NOT
    # prevent provisioning the canonical Hermes-managed copy, or resolution
    # stays pinned to a binary we don't control (version drift, no updates
    # through hermes tools).
    bin_dir = _managed_bin_dir()
    if bin_dir:
        managed = shutil.which("browser-use", path=bin_dir)
        if managed:
            return True, f"browser-use CLI already installed ({managed})"

    uv_bin: Optional[str] = None
    try:
        from hermes_cli.managed_uv import ensure_uv

        uv_bin = str(ensure_uv() or "") or None
    except Exception as e:
        logger.debug("Managed uv bootstrap unavailable: %s", e)
    if not uv_bin:
        uv_bin = shutil.which("uv")
    if not uv_bin:
        return False, (
            "uv is not available and could not be bootstrapped. Install uv "
            "(https://docs.astral.sh/uv/) and run `uv tool install browser-use`."
        )

    env = dict(os.environ)
    env["UV_NO_CONFIG"] = "1"
    if bin_dir:
        try:
            Path(bin_dir).mkdir(parents=True, exist_ok=True)
            env["UV_TOOL_BIN_DIR"] = bin_dir
        except OSError as e:
            logger.debug("Could not prepare %s: %s", bin_dir, e)

    try:
        result = subprocess.run(
            [uv_bin, "tool", "install", "browser-use"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"`uv tool install browser-use` timed out after {timeout_s}s"
    except Exception as e:
        return False, f"Failed to run `uv tool install browser-use`: {e}"

    if result.returncode != 0:
        tail = "\n".join(
            (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        )
        return False, f"`uv tool install browser-use` failed:\n{tail}"

    found = _find_cli()
    if not found or len(found) != 1:
        return False, (
            "install reported success but the browser-use binary is still "
            "not resolvable — run `uv tool install browser-use` manually"
        )
    return True, f"browser-use CLI installed ({found[0]})"


def _workspace_dir(task_id: Optional[str]) -> Optional[str]:
    """Stable per-task scratch dir that persists across browser_exec calls"""
    existing = os.environ.get("BH_AGENT_WORKSPACE")
    if existing:
        return existing
    try:
        from pathlib import Path

        from hermes_constants import get_hermes_home

        safe = _TASK_ID_SAFE_RE.sub("_", str(task_id or "default"))[:80] or "default"
        path = Path(get_hermes_home()) / "cache" / "browser-use" / "workspace" / safe
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    except Exception as e:
        logger.debug("browser_exec workspace unavailable: %s", e)
        return None


# ---------------------------------------------------------------------------
# Browser Harness lifecycle ownership
# ---------------------------------------------------------------------------

_browser_use_sessions: Dict[str, Dict[str, Any]] = {}
_browser_use_lifecycle_lock = threading.RLock()
_browser_use_cleanup_stop = threading.Event()
_browser_use_cleanup_thread: Optional[threading.Thread] = None
_browser_use_instance: Optional[Dict[str, Any]] = None
_browser_use_owner_pid = os.getpid()


def _reset_browser_use_lifecycle_after_fork() -> None:
    """Discard inherited parent ownership in a freshly forked child.

    A child must never run cleanup for daemon state copied from its
    parent. Locks and thread events are also process-local after a fork.
    """
    global _browser_use_cleanup_stop
    global _browser_use_cleanup_thread
    global _browser_use_instance
    global _browser_use_lifecycle_lock
    global _browser_use_owner_pid
    global _browser_use_sessions

    _browser_use_sessions = {}
    _browser_use_lifecycle_lock = threading.RLock()
    _browser_use_cleanup_stop = threading.Event()
    _browser_use_cleanup_thread = None
    _browser_use_instance = None
    _browser_use_owner_pid = os.getpid()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_browser_use_lifecycle_after_fork)


def _browser_use_inactivity_timeout() -> int:
    """Resolve the same public timeout contract as the built-in backend."""
    try:
        from tools.browser_tool import _get_session_inactivity_timeout

        return max(30, int(_get_session_inactivity_timeout()))
    except Exception:
        raw = os.environ.get("BROWSER_INACTIVITY_TIMEOUT", "120")
        try:
            return max(30, int(raw))
        except (TypeError, ValueError):
            return 120


def _browser_use_process_start(pid: int) -> Optional[float]:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _browser_use_owner_is_live(owner: Dict[str, Any]) -> Optional[bool]:
    """Return live/dead only when owner identity can be verified.

    ``None`` means malformed or temporarily unverifiable.  Orphan recovery
    must fail closed in that case rather than treating inspection failure as
    proof that another Hermes process is dead.
    """
    pid = owner.get("pid")
    started = owner.get("process_started_at")
    if type(pid) is not int or pid <= 0 or not isinstance(started, (int, float)):
        return None
    try:
        import psutil
    except Exception:
        return None
    try:
        actual = float(psutil.Process(pid).create_time())
    except psutil.NoSuchProcess:
        return False
    except Exception:
        return None
    return abs(actual - float(started)) < 0.01


def _ensure_private_dir(path: Path) -> Path:
    """Create a non-symlinked owner-only directory or fail closed."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"unsafe Browser Harness lifecycle path: {path}")
    if os.name != "nt":
        path.chmod(0o700)
    return path


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(3)}"
    )
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    if os.name != "nt":
        tmp.chmod(0o600)
    os.replace(tmp, path)


def _browser_use_profile_root() -> Path:
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home()).expanduser().resolve()
    if os.name == "nt":
        return home / "cache" / "browser-use" / "runtime"
    # AF_UNIX paths are capped at 104 bytes on macOS. A profile digest keeps
    # roots short while preserving profile isolation. /tmp is the canonical
    # short socket root on macOS; Linux's tempfile root is normally /tmp.
    tmp_root = Path("/tmp") if sys.platform == "darwin" else Path(tempfile.gettempdir())
    profile = hashlib.sha256(str(home).encode("utf-8")).hexdigest()[:12]
    return tmp_root / f"hermes-bu-{profile}"


def _browser_use_instance_state() -> Dict[str, Any]:
    global _browser_use_instance

    profile_root = _ensure_private_dir(_browser_use_profile_root())
    pid = os.getpid()
    with _browser_use_lifecycle_lock:
        if (
            _browser_use_instance is not None
            and _browser_use_instance.get("pid") == pid
            and _browser_use_instance.get("profile_root") == profile_root
        ):
            return _browser_use_instance

        started = _browser_use_process_start(pid)
        if started is None:
            raise RuntimeError("could not verify the Hermes owner process identity")
        instance_id = f"{pid}-{secrets.token_hex(6)}"
        instance_root = _ensure_private_dir(profile_root / instance_id)
        owner = {
            "managed_by": "hermes-browser-use",
            "instance_id": instance_id,
            "pid": pid,
            "process_started_at": started,
        }
        _atomic_json(instance_root / _OWNER_FILE, owner)
        _browser_use_instance = {
            **owner,
            "profile_root": profile_root,
            "instance_root": instance_root,
        }
        return _browser_use_instance


def _browser_use_session_key(task_id: Optional[str], session: str) -> str:
    # An explicit name is the public, cross-task browser identity already used
    # by BU_NAME and provider cache keys. Keep one daemon for that same named
    # browser. The unnamed default has no caller-supplied identity, so scope it
    # to the Hermes task and never let unrelated tasks share harness state.
    identity = f"named\0{session}" if session else f"task\0{task_id or 'default'}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _write_browser_use_session_state(state: Dict[str, Any]) -> None:
    _atomic_json(
        state["runtime_dir"] / _SESSION_FILE,
        {
            "managed_by": "hermes-browser-use",
            "instance_id": state["instance_id"],
            "session": str(state.get("session") or ""),
            "last_activity": float(state["last_activity"]),
            "cleanup_state": str(state.get("cleanup_state") or "active"),
        },
    )


def _prepare_browser_use_lifecycle(
    env: Dict[str, str],
    command: List[str],
    task_id: Optional[str],
    session: str,
    workspace: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Bind one invocation to an exact Hermes-owned Harness runtime.

    Explicit BH_* paths are operator-owned. Hermes passes them through and
    deliberately does not register, stop, reap, or remove those runtimes.
    """
    if env.get("BH_RUNTIME_DIR") or env.get("BH_TMP_DIR"):
        return None, None

    try:
        instance = _browser_use_instance_state()
        key = _browser_use_session_key(task_id, session)
        runtime_dir = _ensure_private_dir(instance["instance_root"] / f"s-{key}")
        artifact_base = (
            Path(workspace)
            if workspace
            else instance["instance_root"] / "artifacts" / key
        )
        call_id = f"{time.time_ns():x}-{secrets.token_hex(5)}"
        artifact_dir = _ensure_private_dir(
            artifact_base / "browser-artifacts" / call_id
        )
    except Exception as exc:
        return None, f"Could not create a private Browser Harness runtime: {exc}"

    env["BH_RUNTIME_DIR"] = str(runtime_dir)
    env["BH_TMP_DIR"] = str(artifact_dir)
    env["HERMES_BH_TARGET_FILE"] = str(runtime_dir / _TARGET_FILE)
    env["HERMES_BH_DAEMON_CONTRACT_FILE"] = str(runtime_dir / _DAEMON_FILE)
    env["HERMES_BH_INSTANCE_ID"] = str(instance["instance_id"])
    env.pop("BH_RUNTIME_DIR_SHARED", None)
    env.pop("BH_TMP_DIR_SHARED", None)

    runtime_key = str(runtime_dir)
    now = time.time()
    with _browser_use_lifecycle_lock:
        state = _browser_use_sessions.get(runtime_key)
        if state is None:
            state = {
                "runtime_dir": runtime_dir,
                "instance_id": instance["instance_id"],
                "command": list(command),
                "session": session,
                "last_activity": now,
                "cleanup_state": "active",
                "operation_lock": threading.Lock(),
            }
            _browser_use_sessions[runtime_key] = state
        else:
            state["command"] = list(command)
            state["last_activity"] = now
        _write_browser_use_session_state(state)
    _start_browser_use_cleanup_thread()
    return state, None


def _browser_use_runtime_has_endpoint(runtime_dir: Path) -> bool:
    return any(
        path.exists() or path.is_symlink()
        for path in (
            runtime_dir / "bu.sock",
            runtime_dir / "bu.port",
            runtime_dir / "bu.pid",
        )
    )


def _browser_use_should_record_daemon_contract(state: Dict[str, Any]) -> bool:
    runtime_dir = Path(state["runtime_dir"])
    if _browser_use_runtime_has_endpoint(runtime_dir):
        return False
    marker_path = runtime_dir / _DAEMON_FILE
    if not marker_path.exists() and not marker_path.is_symlink():
        return True
    # A prior confirmed shutdown proves the recorded daemon is gone; a new
    # invocation may bind its replacement. An active/stale contract without
    # an endpoint is never overwritten because its process may still exist.
    return (
        state.get("cleanup_state") == "confirmed"
        and not marker_path.is_symlink()
        and _safe_json_object(marker_path) is not None
    )


def _browser_use_reload_env(state: Dict[str, Any]) -> Dict[str, str]:
    env = _base_subprocess_env()
    env["BH_RUNTIME_DIR"] = str(state["runtime_dir"])
    env["BH_TMP_DIR"] = str(state["runtime_dir"] / "cleanup")
    env.pop("BH_RUNTIME_DIR_SHARED", None)
    env.pop("BH_TMP_DIR_SHARED", None)
    if state.get("session"):
        env["BU_NAME"] = str(state["session"])
    else:
        env.pop("BU_NAME", None)
    return env


def _browser_use_owned_target(state: Dict[str, Any]) -> Tuple[Optional[str], str]:
    marker_path = Path(state["runtime_dir"]) / _TARGET_FILE
    if not marker_path.exists() and not marker_path.is_symlink():
        return None, ""
    marker = _safe_json_object(marker_path)
    expected_instance = state.get("instance_id")
    if (
        not marker
        or marker.get("managed_by") != "hermes-browser-use"
        or not expected_instance
        or marker.get("instance_id") != expected_instance
        or marker.get("origin") not in {"harness-dedicated", "hermes-created"}
        or not isinstance(marker.get("daemon_pid"), str)
        or not marker["daemon_pid"].isdigit()
        or not isinstance(marker.get("target_id"), str)
        or not marker["target_id"]
    ):
        return None, f"owned target marker is malformed or unverifiable: {marker_path}"
    return marker["target_id"], ""


def _browser_use_version_has_strict_cloud_shutdown(version: str) -> bool:
    try:
        parts = tuple(int(part) for part in str(version).split(".")[:3])
    except (TypeError, ValueError):
        return False
    return parts >= (0, 1, 10)


def _browser_use_daemon_contract(
    state: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    marker_path = Path(state["runtime_dir"]) / _DAEMON_FILE
    if not marker_path.exists() and not marker_path.is_symlink():
        return None, ""
    marker = _safe_json_object(marker_path)
    expected_instance = state.get("instance_id")
    if (
        not marker
        or marker.get("managed_by") != "hermes-browser-use"
        or not expected_instance
        or marker.get("instance_id") != expected_instance
        or not isinstance(marker.get("daemon_pid"), str)
        or not marker["daemon_pid"].isdigit()
        or int(marker["daemon_pid"]) <= 0
        or int(marker["daemon_pid"]) >= (1 << 31)
        or isinstance(marker.get("daemon_started"), bool)
        or not isinstance(marker.get("daemon_started"), (int, float, str))
        or str(marker["daemon_started"]) == ""
        or not isinstance(marker.get("harness_version"), str)
        or type(marker.get("strict_cloud_shutdown")) is not bool
        or marker["strict_cloud_shutdown"]
        != _browser_use_version_has_strict_cloud_shutdown(
            marker["harness_version"]
        )
    ):
        return None, f"daemon contract marker is malformed or unverifiable: {marker_path}"
    return marker, ""


def _browser_use_cleanup_program() -> str:
    """Program executed inside the installed Harness environment.

    A daemon may outlive its spawning CLI package. Cloud cleanup therefore
    requires a durable capability marker bound to the live daemon PID, rather
    than inferring daemon behavior from the current admin module. Local/CDP
    shutdown remains confirmable through authenticated IPC plus exact
    PID/endpoint verification.
    """
    prefix = json.dumps(_CLEANUP_RESULT_PREFIX)
    return f'''\
stop_remote_daemon(None) if False else None
# Both supported Harness run.py versions skip ensure_daemon() when code starts
# with this admin expression. The call above is unreachable: cleanup must
# inspect the existing owned daemon and must never auto-spawn one.
import json as _json, os as _os
from browser_harness import admin as _admin
from browser_harness import _ipc as _ipc

_name = _os.environ.get("BU_NAME", "default")
_owned_target = _os.environ.get("HERMES_BH_OWNED_TARGET_ID") or None
_strict_cloud_pid = _os.environ.get("HERMES_BH_STRICT_CLOUD_DAEMON_PID") or None
_strict_cloud_started = _os.environ.get("HERMES_BH_STRICT_CLOUD_DAEMON_STARTED") or None
_result = {{"ok": False, "version": str(_admin._version() or "unknown")}}

def _emit(_ok, _error="", **_extra):
    _result.update(_extra)
    _result["ok"] = bool(_ok)
    if _error:
        _result["error"] = str(_error)
    print({prefix} + _json.dumps(_result, sort_keys=True), flush=True)

def _request(_payload, _timeout=5.0):
    _conn, _token = _ipc.connect(_name, timeout=_timeout)
    try:
        return _ipc.request(_conn, _token, _payload)
    finally:
        _conn.close()

try:
    try:
        _ping = _request({{"meta": "ping"}})
    except Exception as _exc:
        raise RuntimeError(
            "owned daemon did not answer an authenticated ping: " + str(_exc)
        ) from _exc
    if not isinstance(_ping, dict) or _ping.get("pong") is not True:
        raise RuntimeError("owned daemon did not answer an authenticated ping")
    _kind = str(_ping.get("browser_kind") or "unknown")
    _result["browser_kind"] = _kind
    _ping_pid = _ping.get("pid")
    _live_started = (
        _admin._process_start_time(_ping_pid)
        if _kind == "cloud" and type(_ping_pid) is int and _strict_cloud_pid
        else None
    )
    if _kind == "cloud" and (
        not _strict_cloud_pid
        or not _strict_cloud_started
        or str(_ping_pid) != _strict_cloud_pid
        or str(_live_started) != _strict_cloud_started
    ):
        raise RuntimeError(
            "the live daemon does not prove strict cloud shutdown authority; "
            "Hermes cannot confirm Browser Use Cloud stop, profile persistence, "
            "or billing cleanup. Start this owned runtime with Browser Harness "
            "0.1.10+ before retrying cleanup"
        )

    if _owned_target:
        _before = cdp("Target.getTargets").get("targetInfos", [])
        _before_ids = {{
            _item.get("targetId") for _item in _before
            if isinstance(_item, dict) and _item.get("type") == "page"
        }}
        if _owned_target in _before_ids:
            cdp("Target.closeTarget", targetId=_owned_target)
        _after = cdp("Target.getTargets").get("targetInfos", [])
        _after_ids = {{
            _item.get("targetId") for _item in _after
            if isinstance(_item, dict) and _item.get("type") == "page"
        }}
        if _owned_target in _after_ids:
            raise RuntimeError(
                "exact Hermes-owned browser target remained after close: "
                + _owned_target
            )
        _result["target_closed"] = True

    # The live daemon owns the shutdown contract. Direct authenticated IPC
    # avoids inferring its behavior from a newer/older current admin module.
    _shutdown = _request({{"meta": "shutdown"}})
    if (
        not isinstance(_shutdown, dict)
        or _shutdown.get("ok") is not True
        or bool(_shutdown.get("error"))
    ):
        _error = _shutdown.get("error") if isinstance(_shutdown, dict) else None
        raise RuntimeError(_error or "Browser Harness daemon did not confirm shutdown")
    _emit(True)
except BaseException as _exc:
    _emit(False, _exc)
    raise SystemExit(86)
'''


def _parse_browser_use_cleanup_result(stdout: str) -> Optional[Dict[str, Any]]:
    for line in reversed((stdout or "").splitlines()):
        if not line.startswith(_CLEANUP_RESULT_PREFIX):
            continue
        try:
            parsed = json.loads(line[len(_CLEANUP_RESULT_PREFIX) :])
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _mark_browser_use_cleanup_confirmed(state: Dict[str, Any]) -> Tuple[bool, str]:
    runtime_dir = Path(state["runtime_dir"])
    marker_path = runtime_dir / _SESSION_FILE
    marker = _safe_json_object(marker_path)
    if (
        not marker
        or marker.get("managed_by") != "hermes-browser-use"
        or marker.get("instance_id") != state.get("instance_id")
    ):
        return False, (
            "daemon cleanup was confirmed but durable runtime ownership is "
            f"missing or unverifiable: {marker_path}"
        )
    marker["cleanup_state"] = "confirmed"
    try:
        _atomic_json(marker_path, marker)
    except OSError as exc:
        return False, (
            "daemon cleanup was confirmed but retirement authority could not "
            f"be persisted at {marker_path}: {exc}"
        )
    state["cleanup_state"] = "confirmed"
    return True, ""


def _reload_browser_use_session(state: Dict[str, Any]) -> Tuple[bool, str]:
    command = list(state.get("command") or [])
    if not command:
        return False, "browser-use command is unavailable"
    runtime_dir = Path(state["runtime_dir"])
    pid_path = runtime_dir / "bu.pid"
    daemon_identity: Optional[Tuple[int, float]] = None
    if pid_path.is_symlink():
        return False, f"refusing symlinked Browser Harness pid file: {pid_path}"
    try:
        daemon_pid = int(pid_path.read_text(encoding="utf-8").strip())
        daemon_started = _browser_use_process_start(daemon_pid)
        if daemon_pid > 0 and daemon_started is not None:
            daemon_identity = (daemon_pid, daemon_started)
    except (OSError, TypeError, ValueError):
        pass
    owned_target, target_error = _browser_use_owned_target(state)
    if target_error:
        return False, target_error
    daemon_contract, contract_error = _browser_use_daemon_contract(state)
    if contract_error:
        return False, contract_error
    cleanup_env = _browser_use_reload_env(state)
    if owned_target:
        cleanup_env["HERMES_BH_OWNED_TARGET_ID"] = owned_target
    else:
        cleanup_env.pop("HERMES_BH_OWNED_TARGET_ID", None)
    if daemon_contract and daemon_contract["strict_cloud_shutdown"]:
        cleanup_env["HERMES_BH_STRICT_CLOUD_DAEMON_PID"] = daemon_contract[
            "daemon_pid"
        ]
        cleanup_env["HERMES_BH_STRICT_CLOUD_DAEMON_STARTED"] = str(
            daemon_contract["daemon_started"]
        )
    else:
        cleanup_env.pop("HERMES_BH_STRICT_CLOUD_DAEMON_PID", None)
        cleanup_env.pop("HERMES_BH_STRICT_CLOUD_DAEMON_STARTED", None)
    try:
        proc = subprocess.run(
            command,
            input=_browser_use_cleanup_program(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=cleanup_env,
            timeout=_LIFECYCLE_RELOAD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, (
            "strict Browser Harness cleanup timed out after "
            f"{_LIFECYCLE_RELOAD_TIMEOUT_S:.0f}s"
        )
    except OSError as exc:
        return False, f"could not launch strict Browser Harness cleanup: {exc}"
    cleanup_result = _parse_browser_use_cleanup_result(proc.stdout)
    if not cleanup_result or cleanup_result.get("ok") is not True:
        detail = ""
        if cleanup_result:
            detail = str(cleanup_result.get("error") or "").strip()
        if not detail:
            detail = (proc.stderr or proc.stdout or "").strip()
        if len(detail) > 1000:
            detail = detail[-1000:]
        return False, (
            "strict Browser Harness cleanup was not confirmed"
            + (f" (exit {proc.returncode}): {detail}" if detail else "")
        )
    if proc.returncode != 0:
        return False, (
            "strict Browser Harness cleanup reported success but exited "
            f"{proc.returncode}"
        )

    if daemon_identity is not None:
        daemon_pid, daemon_started = daemon_identity
        deadline = time.monotonic() + _LIFECYCLE_VERIFY_TIMEOUT_S
        while time.monotonic() < deadline:
            actual = _browser_use_process_start(daemon_pid)
            if actual is None or abs(actual - daemon_started) >= 0.01:
                break
            time.sleep(0.05)
        else:
            return False, (
                "strict cleanup returned but exact owned daemon "
                f"pid {daemon_pid} (start {daemon_started:.6f}) is still alive"
            )

    leftovers = [
        path.name
        for path in (runtime_dir / "bu.sock", runtime_dir / "bu.port", pid_path)
        if path.exists() or path.is_symlink()
    ]
    if leftovers:
        return False, (
            "strict cleanup returned but owned runtime endpoints remain: "
            + ", ".join(leftovers)
        )
    return _mark_browser_use_cleanup_confirmed(state)


def _retire_browser_use_runtime(state: Dict[str, Any]) -> Tuple[bool, str]:
    """Remove only the exact, confirmed, structurally empty owned runtime."""
    runtime_dir = Path(state["runtime_dir"])
    expected_instance = state.get("instance_id")
    if (
        not expected_instance
        or not runtime_dir.name.startswith("s-")
        or runtime_dir.parent.name != expected_instance
        or runtime_dir.is_symlink()
        or runtime_dir.parent.is_symlink()
        or not runtime_dir.is_dir()
    ):
        return False, f"owned runtime path is unsafe or unverifiable: {runtime_dir}"

    marker_path = runtime_dir / _SESSION_FILE
    marker = _safe_json_object(marker_path)
    if marker is None:
        # Crash window after unlinking the final marker but before rmdir().
        # In-memory confirmed state may finish only an exactly empty runtime,
        # never one containing data. The dead-owner reaper has the equivalent
        # empty-directory rule at its verified parent boundary.
        try:
            empty = not any(runtime_dir.iterdir())
        except OSError as exc:
            return False, (
                f"could not inspect markerless owned runtime {runtime_dir}: {exc}"
            )
        if state.get("cleanup_state") == "confirmed" and empty:
            try:
                runtime_dir.rmdir()
                return True, ""
            except OSError as exc:
                return False, f"could not retire empty owned runtime {runtime_dir}: {exc}"
        return False, f"owned runtime marker is missing or unverifiable: {marker_path}"
    if (
        marker.get("managed_by") != "hermes-browser-use"
        or marker.get("instance_id") != expected_instance
        or marker.get("cleanup_state") != "confirmed"
    ):
        return False, f"owned runtime cleanup is not durably confirmed: {marker_path}"

    cleanup_dir = runtime_dir / "cleanup"
    allowed = {_SESSION_FILE, _TARGET_FILE, _DAEMON_FILE, cleanup_dir.name}
    try:
        unexpected = sorted(
            child.name for child in runtime_dir.iterdir() if child.name not in allowed
        )
    except OSError as exc:
        return False, (
            f"could not inspect owned runtime before retirement {runtime_dir}: {exc}"
        )
    if unexpected:
        return False, (
            f"owned runtime contains unexpected entries and was retained: {runtime_dir}: "
            + ", ".join(unexpected)
        )

    for marker_name in (_TARGET_FILE, _DAEMON_FILE):
        marker_file = runtime_dir / marker_name
        if marker_file.is_symlink() or (
            marker_file.exists() and not marker_file.is_file()
        ):
            return False, f"owned runtime contains unsafe marker path: {marker_file}"
    if cleanup_dir.is_symlink() or (cleanup_dir.exists() and not cleanup_dir.is_dir()):
        return False, f"owned cleanup path is unsafe: {cleanup_dir}"
    if cleanup_dir.exists():
        try:
            cleanup_entries = sorted(child.name for child in cleanup_dir.iterdir())
        except OSError as exc:
            return False, f"could not inspect owned cleanup directory {cleanup_dir}: {exc}"
        if cleanup_entries:
            return False, (
                "owned cleanup directory contains unexpected entries and was retained: "
                f"{cleanup_dir}: " + ", ".join(cleanup_entries)
            )

    try:
        if cleanup_dir.exists():
            cleanup_dir.rmdir()
        (runtime_dir / _TARGET_FILE).unlink(missing_ok=True)
        (runtime_dir / _DAEMON_FILE).unlink(missing_ok=True)
        # The session marker is the final ownership proof. Remove it only
        # after every other bounded retirement step has completed.
        marker_path.unlink()
        runtime_dir.rmdir()
    except OSError as exc:
        return False, f"owned runtime retirement failed for {runtime_dir}: {exc}"
    return True, ""


def _forget_browser_use_session(state: Dict[str, Any]) -> Tuple[bool, str]:
    retired, detail = _retire_browser_use_runtime(state)
    if not retired:
        return False, detail
    with _browser_use_lifecycle_lock:
        _browser_use_sessions.pop(str(state["runtime_dir"]), None)
    return True, ""


def _cleanup_browser_use_sessions(
    *, now: Optional[float] = None, force: bool = False
) -> List[str]:
    current = time.time() if now is None else float(now)
    timeout = _browser_use_inactivity_timeout()
    with _browser_use_lifecycle_lock:
        candidates = list(_browser_use_sessions.values())

    failures: List[str] = []
    for state in candidates:
        if not force and current - float(state["last_activity"]) <= timeout:
            continue
        operation_lock = state["operation_lock"]
        acquired = (
            operation_lock.acquire(timeout=5)
            if force
            else operation_lock.acquire(blocking=False)
        )
        if not acquired:
            continue
        try:
            # Re-check after acquiring: a call may have refreshed activity
            # while the cleanup worker was waiting.
            if not force and current - float(state["last_activity"]) <= timeout:
                continue
            retired = False
            if state.get("cleanup_state") == "confirmed":
                ok, detail = True, ""
            else:
                ok, detail = _reload_browser_use_session(state)
            if ok:
                retired, detail = _forget_browser_use_session(state)
                if retired:
                    continue
                message = (
                    "Browser Harness runtime retirement failed for "
                    f"{state['runtime_dir']}: {detail}"
                )
            else:
                message = (
                    f"Browser Harness cleanup failed for {state['runtime_dir']}: "
                    f"{detail}"
                )
            if not ok or not retired:
                failures.append(message)
                logger.warning(message)
        finally:
            operation_lock.release()
    return failures


def _browser_use_cleanup_worker() -> None:
    while not _browser_use_cleanup_stop.wait(_LIFECYCLE_CHECK_INTERVAL_S):
        try:
            _cleanup_browser_use_sessions()
            _reap_orphaned_browser_use_instances()
        except Exception as exc:
            logger.warning("Browser Harness lifecycle worker failed: %s", exc)


def _start_browser_use_cleanup_thread() -> None:
    global _browser_use_cleanup_thread
    with _browser_use_lifecycle_lock:
        if (
            _browser_use_cleanup_thread is not None
            and _browser_use_cleanup_thread.is_alive()
        ):
            return
        _browser_use_cleanup_stop.clear()
        _browser_use_cleanup_thread = threading.Thread(
            target=_browser_use_cleanup_worker,
            daemon=True,
            name="browser-use-lifecycle",
        )
        _browser_use_cleanup_thread.start()
    _reap_orphaned_browser_use_instances()


def _stop_browser_use_cleanup_thread() -> None:
    global _browser_use_cleanup_thread
    _browser_use_cleanup_stop.set()
    thread = _browser_use_cleanup_thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=2)
    _browser_use_cleanup_thread = None


def _safe_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _acquire_browser_use_reap_claim(
    instance_root: Path,
) -> Tuple[Optional[Any], bool, str]:
    """Acquire a crash-released kernel lock for one orphan instance.

    The lock file lives beside the instance, not inside it. This lets native
    Windows retire the instance while its msvcrt handle remains open, while a
    stable never-unlinked inode preserves the POSIX anti-replacement race
    property. The file may survive a killed reaper; the kernel lock cannot.
    ``busy`` means another live process currently owns the claim.
    """
    lock_path = instance_root.parent / f".{instance_root.name}{_REAP_LOCK_FILE}"
    if lock_path.is_symlink():
        return None, False, f"refusing symlinked orphan-reaper claim: {lock_path}"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags, 0o600)
        handle = os.fdopen(fd, "r+b")
    except OSError as exc:
        return None, False, f"could not open orphan-reaper claim {lock_path}: {exc}"
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                handle.close()
                if getattr(exc, "winerror", None) in {33, 36}:
                    return None, True, ""
                return None, False, f"could not lock orphan-reaper claim: {exc}"
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                return None, True, ""

        started = _browser_use_process_start(os.getpid())
        if started is None:
            raise RuntimeError("could not verify orphan-reaper process identity")
        claim = {
            "managed_by": "hermes-browser-use-reaper",
            "claim_id": secrets.token_hex(8),
            "pid": os.getpid(),
            "process_started_at": started,
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(claim, sort_keys=True).encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
        return handle, False, ""
    except Exception as exc:
        try:
            handle.close()
        except OSError:
            pass
        return None, False, f"could not establish orphan-reaper claim: {exc}"


def _release_browser_use_reap_claim(handle: Any) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _reap_orphaned_browser_use_instances() -> List[str]:
    """Reap only dead Hermes instances under the current profile root."""
    try:
        profile_root = _browser_use_profile_root()
    except Exception:
        return []
    if not profile_root.is_dir() or profile_root.is_symlink():
        return []

    current_root = (
        _browser_use_instance.get("instance_root")
        if _browser_use_instance is not None
        else None
    )
    failures: List[str] = []
    for instance_root in profile_root.iterdir():
        if (
            instance_root == current_root
            or instance_root.is_symlink()
            or not instance_root.is_dir()
        ):
            continue
        owner = _safe_json_object(instance_root / _OWNER_FILE)
        owner_is_live = _browser_use_owner_is_live(owner) if owner else None
        if (
            not owner
            or owner.get("managed_by") != "hermes-browser-use"
            or owner.get("instance_id") != instance_root.name
        ):
            continue
        if owner_is_live is True:
            continue
        if owner_is_live is None:
            failures.append(
                f"Browser Harness orphan owner is unverifiable: {instance_root}"
            )
            continue

        claim, claim_busy, claim_error = _acquire_browser_use_reap_claim(
            instance_root
        )
        if claim_busy:
            continue
        if claim is None:
            failures.append(claim_error)
            continue
        all_reaped = True
        scan_completed = False
        try:
            command = _find_cli()
            for runtime_dir in instance_root.glob("s-*"):
                if runtime_dir.is_symlink() or not runtime_dir.is_dir():
                    all_reaped = False
                    failures.append(
                        f"Browser Harness orphan runtime path is unsafe: {runtime_dir}"
                    )
                    continue
                persisted = _safe_json_object(runtime_dir / _SESSION_FILE)
                if persisted is None:
                    inspect_error = None
                    try:
                        markerless_empty = not any(runtime_dir.iterdir())
                    except OSError as exc:
                        markerless_empty = False
                        inspect_error = (
                            f"Browser Harness markerless runtime could not be inspected "
                            f"for {runtime_dir}: {exc}"
                        )
                        failures.append(inspect_error)
                    if markerless_empty:
                        try:
                            runtime_dir.rmdir()
                            continue
                        except OSError as exc:
                            failures.append(
                                f"Browser Harness empty markerless runtime could not be "
                                f"retired for {runtime_dir}: {exc}"
                            )
                    elif inspect_error is None:
                        failures.append(
                            f"Browser Harness orphan runtime has no verifiable ownership "
                            f"marker and was retained: {runtime_dir}"
                        )
                    all_reaped = False
                    continue
                if (
                    persisted.get("managed_by") != "hermes-browser-use"
                    or persisted.get("instance_id") != instance_root.name
                ):
                    all_reaped = False
                    failures.append(
                        f"Browser Harness orphan runtime ownership is unverifiable: "
                        f"{runtime_dir}"
                    )
                    continue
                state = {
                    "runtime_dir": runtime_dir,
                    "command": list(command or []),
                    "session": str(persisted.get("session") or ""),
                    "instance_id": instance_root.name,
                    "cleanup_state": str(persisted.get("cleanup_state") or "active"),
                }
                if state["cleanup_state"] == "confirmed":
                    ok, detail = True, ""
                else:
                    ok, detail = _reload_browser_use_session(state)
                if not ok:
                    all_reaped = False
                    failures.append(
                        f"Browser Harness orphan cleanup failed for {runtime_dir}: {detail}"
                    )
                    continue
                retired, detail = _retire_browser_use_runtime(state)
                if not retired:
                    all_reaped = False
                    failures.append(
                        "Browser Harness orphan runtime could not be retired "
                        f"after cleanup for {runtime_dir}: {detail}"
                    )
            scan_completed = True
        finally:
            # The claim file is a stable sibling of the instance and remains
            # locked throughout retirement on every host, including Windows.
            if scan_completed and all_reaped:
                try:
                    retained = {
                        child.name
                        for child in instance_root.iterdir()
                        if child.name not in {_OWNER_FILE, _REAP_LOCK_FILE}
                    }
                    if not retained:
                        (instance_root / _OWNER_FILE).unlink(missing_ok=True)
                        # Legacy in-instance claim from pre-migration builds.
                        (instance_root / _REAP_LOCK_FILE).unlink(missing_ok=True)
                        instance_root.rmdir()
                except OSError as exc:
                    failures.append(
                        f"Browser Harness orphan instance could not be retired "
                        f"for {instance_root}: {exc}"
                    )
            _release_browser_use_reap_claim(claim)
    for failure in failures:
        logger.warning(failure)
    return failures


def _reset_browser_use_lifecycle_for_tests() -> None:
    global _browser_use_instance
    _stop_browser_use_cleanup_thread()
    _cleanup_browser_use_sessions(force=True)
    with _browser_use_lifecycle_lock:
        _browser_use_sessions.clear()
        _browser_use_instance = None


def _shutdown_browser_use_lifecycle() -> None:
    _stop_browser_use_cleanup_thread()
    failures = _cleanup_browser_use_sessions(force=True)
    for failure in failures:
        logger.error(failure)


atexit.register(_shutdown_browser_use_lifecycle)


def _find_screenshot(stdout: str, since: float) -> Optional[str]:
    """Return the last screenshot path printed during this exec, or None.

    Only accepts files that exist and were written after the exec started
    """
    for path in reversed(_IMAGE_PATH_RE.findall(stdout or "")):
        try:
            if os.path.isfile(path) and os.path.getmtime(path) >= since - 1:
                return path
        except OSError:
            continue
    return None


def _native_screenshot_result(result: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
    """Build a multimodal tool result attaching path for vision models"""
    try:
        from pathlib import Path

        from tools.vision_tools import (
            _EMBED_MAX_DIMENSION,
            _EMBED_TARGET_BYTES,
            _resize_image_for_vision,
            _should_use_native_vision_fast_path,
        )

        if not _should_use_native_vision_fast_path():
            return None
        # History-reuse cap (#92699): this data URL bakes into the tool
        # result and is re-sent on every later turn — same policy as the
        # vision_analyze / browser_vision native embeds (256 KB / 1568 px,
        # JPEG quality ladder instead of PNG dimension-halving).
        data_url = _resize_image_for_vision(
            Path(path),
            mime_type="image/png",
            max_base64_bytes=_EMBED_TARGET_BYTES,
            max_dimension=_EMBED_MAX_DIMENSION,
            force_jpeg=True,
        )
        text = json.dumps(result, ensure_ascii=False)
        return {
            "_multimodal": True,
            "content": [
                {
                    "type": "text",
                    "text": (
                        text
                        + "\n\nThe screenshot from this call is attached — "
                        "inspect it with your native vision."
                    ),
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
            "text_summary": text,
            "meta": {"screenshot_path": path, "native_vision": True},
        }
    except Exception as e:
        logger.debug("Native screenshot attach failed (falling back to text): %s", e)
        return None


def _resolve_backend_cdp(
    env: dict, task_id: Optional[str], session_name: str = ""
) -> Optional[str]:
    """Point the harness at the configured browser backend's CDP endpoint.

    Resolution order (first hit wins):

    1. ``BU_CDP_WS`` / ``BU_CDP_URL`` already in the environment — explicit
       user/operator override, passed through untouched.
    2. ``BROWSER_CDP_URL`` env / ``browser.cdp_url`` config override — the
       ``/browser connect`` path, same precedence the built-in tools honor.
    3. A configured cloud browser provider (Browserbase, Firecrawl, Nous
       gateway/Browser Use cloud, …): reuse the legacy stack's
       ``_get_session_info()`` so browser_exec shares the SAME provider
       session machinery — per-task session cache, expiry replacement,
       inactivity reaper, and atexit cleanup — instead of duplicating it.
    4. Nothing configured: return None; the harness attaches to local
       Chrome (or Browser Use cloud via BU_AUTOSPAWN for legacy configs).

    ``session_name`` (the tool's ``session`` argument / BU_NAME) keys the
    provider session cache when set, so every distinct name gets its OWN
    cloud browser and the same name reuses one — that is what makes named
    sessions actually concurrent-safe on provider backends instead of all
    names sharing a single per-task browser.

    Returns an error string on provider failure, None on success.
    """
    if env.get("BU_CDP_WS") or env.get("BU_CDP_URL"):
        return None

    try:
        from tools.browser_tool import (
            _get_cdp_override,
            _get_cloud_provider,
            _get_session_info,
        )
    except Exception as e:  # pragma: no cover — stubbed browser_tool in tests
        logger.debug("browser_tool backend resolution unavailable: %s", e)
        return None

    try:
        override = _get_cdp_override()
    except Exception:
        override = ""
    if override:
        env["BU_CDP_URL" if override.startswith(("http://", "https://")) else "BU_CDP_WS"] = override
        return None

    try:
        provider = _get_cloud_provider()
    except Exception as e:
        logger.debug("Cloud provider lookup failed: %s", e)
        provider = None
    if provider is None:
        return None

    # Browser Use direct-API configs: the CLI talks to Browser Use cloud
    # natively (BU_AUTOSPAWN / auth login) — routing through the legacy
    # provider here would just create a second, redundant session. The
    # Nous-gateway variant (use_gateway: true) DOES resolve through the
    # provider: the gateway provisions the cloud browser server-side and
    # returns its CDP URL, giving subscribers CLI mode with no raw key.
    provider_key = str(getattr(provider, "name", "") or "").strip().lower()
    if provider_key == _BACKEND_KEY and not is_truthy_value(
        _read_browser_cfg().get("use_gateway"), default=False
    ):
        # Named BU cloud browsers are exclusive to their daemon — no shared
        # tab to isolate from.
        env[_PRIVATE_BROWSER_SENTINEL] = "1"
        return None

    try:
        # Named sessions get their OWN provider browser, keyed by name so the
        # same name reuses one browser across calls and tasks, and different
        # names never collide. Unnamed calls keep the per-task key.
        cache_key = f"bu-named-{session_name}" if session_name else (task_id or "browser-exec-default")
        session_info = _get_session_info(cache_key)
    except Exception as e:
        return (
            f"Cloud browser provider {type(provider).__name__} failed to "
            f"provide a session: {e}. Fix the provider configuration or "
            "switch backends via `hermes tools` → Browser Automation."
        )
    cdp = str((session_info or {}).get("cdp_url") or "")
    if not cdp:
        return (
            f"Cloud browser provider {type(provider).__name__} returned no "
            "CDP endpoint, so Browser Use mode cannot drive it. Switch to "
            "the built-in browser tools for this provider."
        )
    env["BU_CDP_URL" if cdp.startswith(("http://", "https://")) else "BU_CDP_WS"] = cdp
    # A provider browser keyed bu-named-<name> is exclusive to this session —
    # the own-tab preamble is unnecessary there (it would just leak a blank
    # tab into a browser nobody else touches).
    if session_name:
        env[_PRIVATE_BROWSER_SENTINEL] = "1"
    return None


def browser_exec(
    code: str,
    session: str = "",
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    task_id: Optional[str] = None,
):
    """Run Python code through the browser-use CLI, and return its output"""
    from tools.registry import tool_error, tool_result

    if not code or not code.strip():
        return tool_error("No code provided. Pass Python that uses the pre-imported helpers, e.g. new_tab(\"https://example.com\") then print(page_info()).")

    blocked = _blocked_url_in_code(code)
    if blocked:
        return tool_error(blocked)

    cmd = _find_cli()
    if not cmd:
        return tool_error(
            "browser-use CLI not found on PATH, and uvx is unavailable for a "
            "zero-install run. Install it with `uv tool install browser-use` "
            "(or `pipx install browser-use`), then run `browser-use --doctor` "
            "to verify the setup."
        )

    env = _base_subprocess_env()
    if session:
        if not _SESSION_RE.match(session):
            return tool_error(
                f"Invalid session name {session!r}: use 1-64 letters, digits, "
                "dashes, or underscores (e.g. 'r7k2')."
            )
        env["BU_NAME"] = session
    # Route through the configured browser backend (Browserbase, Firecrawl,
    # Nous gateway, CDP override, local Chrome, …). Named sessions compose
    # with the backend: BU_NAME namespaces the harness daemon (its IPC
    # socket, log, and pid), and on provider backends the name additionally
    # keys its own cloud browser — so concurrent sessions stop clobbering
    # each other's daemon (#86894). Browser Use direct-API cloud configs
    # are the one exception: the CLI manages named cloud browsers natively,
    # and _resolve_backend_cdp skips provider resolution for them.
    backend_err = _resolve_backend_cdp(env, task_id, session_name=session)
    if backend_err:
        return tool_error(backend_err)

    private_browser = env.pop(_PRIVATE_BROWSER_SENTINEL, None)

    workspace = _workspace_dir(task_id)
    if workspace:
        env["BH_AGENT_WORKSPACE"] = workspace

    lifecycle_state, lifecycle_error = _prepare_browser_use_lifecycle(
        env, cmd, task_id, session, workspace
    )
    if lifecycle_error:
        return tool_error(lifecycle_error)

    lifecycle_preamble = ""
    if lifecycle_state is not None:
        lifecycle_preamble += _DAEMON_CONTRACT_PREAMBLE

    # On shared browsers, claim one exact target under the managed lifecycle.
    # Named Harness daemons already create a dedicated tab; unnamed task
    # runtimes create one here. Operator-owned runtimes and private provider/
    # cloud browsers are never given a Hermes-owned target.
    if not private_browser and lifecycle_state is not None:
        env["HERMES_BH_TARGET_MODE"] = "claim" if session else "create"
        lifecycle_preamble += _OWN_TAB_PREAMBLE
    else:
        env.pop("HERMES_BH_TARGET_MODE", None)
    code = _dialog_safe_click_preamble() + lifecycle_preamble + code

    # BU_AUTOSPAWN makes the CLI start a Browser Use cloud browser when no
    # local Chrome/CDP endpoint is reachable (their API key authenticates it)
    if "BU_AUTOSPAWN" not in env and is_legacy_browser_use_cloud_config(_read_browser_cfg()):
        env["BU_AUTOSPAWN"] = "1"

    try:
        timeout = max(_MIN_TIMEOUT_S, min(int(timeout_s), _MAX_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_S

    # Windows: hide the console the .cmd shim would flash (as browser_tool does)
    popen_extra: dict = {}
    if os.name == "nt":
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            popen_extra["creationflags"] = windows_hide_flags()
            _si = subprocess.STARTUPINFO()
            _si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            popen_extra["startupinfo"] = _si
        except Exception as e:
            logger.debug("Windows hide-flags unavailable: %s", e)

    operation_context = (
        lifecycle_state["operation_lock"]
        if lifecycle_state is not None
        else contextlib.nullcontext()
    )
    started = time.time()
    daemon_cleanup: Optional[Dict[str, Any]] = None
    with operation_context:
        if lifecycle_state is not None:
            record_contract = _browser_use_should_record_daemon_contract(
                lifecycle_state
            )
            lifecycle_state["last_activity"] = started
            lifecycle_state["cleanup_state"] = "active"
            _write_browser_use_session_state(lifecycle_state)
            env["HERMES_BH_RECORD_DAEMON_CONTRACT"] = (
                "1" if record_contract else "0"
            )
        try:
            proc = subprocess.run(
                cmd,
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                **popen_extra,
            )
        except subprocess.TimeoutExpired:
            detail = ""
            if lifecycle_state is not None:
                stopped, cleanup_detail = _reload_browser_use_session(
                    lifecycle_state
                )
                if stopped:
                    retired, retirement_detail = _forget_browser_use_session(
                        lifecycle_state
                    )
                    if retired:
                        detail = (
                            " The exact Hermes-owned Harness daemon and runtime "
                            "were retired."
                        )
                    else:
                        detail = (
                            " The exact Hermes-owned Harness daemon was stopped, "
                            "but runtime retirement failed: "
                            f"{retirement_detail}."
                        )
                else:
                    detail = f" Exact daemon cleanup failed: {cleanup_detail}."
            else:
                detail = (
                    " The Harness runtime is operator-owned, so Hermes did "
                    "not stop or reap it."
                )
            return tool_error(
                f"browser-use exec timed out after {timeout}s.{detail} "
                "Split the work into several calls that append to workspace "
                "files; anything already written there is preserved."
            )
        except OSError as e:
            return tool_error(f"Failed to launch browser-use CLI: {e}")

        stderr_full = (proc.stderr or "").strip()
        if lifecycle_state is not None:
            lifecycle_state["last_activity"] = time.time()
            _write_browser_use_session_state(lifecycle_state)
            if any(
                marker in stderr_full
                for marker in (
                    "HERMES_BROWSER_INPUT_WEDGED:",
                    "HERMES_BROWSER_CONTROL_WEDGED:",
                )
            ):
                stopped, cleanup_detail = _reload_browser_use_session(
                    lifecycle_state
                )
                daemon_cleanup = {
                    "attempted": True,
                    "success": stopped,
                }
                if cleanup_detail:
                    daemon_cleanup["error"] = cleanup_detail
                if stopped:
                    retired, retirement_detail = _forget_browser_use_session(
                        lifecycle_state
                    )
                    daemon_cleanup["success"] = retired
                    if not retired:
                        daemon_cleanup["daemon_stopped"] = True
                        daemon_cleanup["error"] = retirement_detail

    result = {
        "success": proc.returncode == 0,
        "exit_code": proc.returncode,
        "output": proc.stdout,
    }
    if workspace:
        result["workspace"] = workspace
    if session:
        result["session"] = session
    if daemon_cleanup is not None:
        result["daemon_cleanup"] = daemon_cleanup
    stderr = (proc.stderr or "").strip()
    if stderr:
        if len(stderr) > _STDERR_CAP_CHARS:
            stderr = stderr[:_STDERR_CAP_CHARS] + "\n… (stderr truncated)"
        result["stderr"] = stderr

    screenshot = _find_screenshot(proc.stdout, started)
    if screenshot:
        result["screenshot_path"] = screenshot
        native = _native_screenshot_result(result, screenshot)
        if native is not None:
            return native
    return tool_result(result)


# The tool description is the CLI's skill, fetched from browser-use skill
_HEADER_BASE = (
    "Drive a real web browser via the Browser Use CLI. The `code` argument "
    "is executed as full Python (standard library available) after a small "
    "Hermes compatibility bootstrap, with the CLI's pre-imported browser "
    "helpers; stdout comes back in the result. Start `code` with a "
    "one-line comment describing the step for the user in plain, "
    "non-technical language, max 60 chars (e.g. `# Searching Amazon for "
    "paper towels`) — the UI displays it as the step label.\n\n"
    "STATE: the browser session and the workspace persist across calls; "
    "Python variables do NOT (each call is a fresh interpreter). The "
    "workspace is a stable directory — path in $BH_AGENT_WORKSPACE and "
    "returned as `workspace` in every result. For multi-item tasks "
    "('collect all N products / every entry / the full table'), append each "
    "batch to a JSON/CSV file in the workspace as you go, then read it back "
    "to assemble the final answer; define reusable functions in "
    "agent_helpers.py there — the harness auto-imports it into every call. "
    "Do aggregation in code, not in your head: dedupe, count, sort, and "
    "format with Python inside the exec. Screenshots use a distinct, "
    "persistent artifact directory for every call, so keep and return the "
    "exact path printed by that call. Native JavaScript dialogs are never "
    "answered implicitly: `click_at_xy` returns and prints "
    "`dialog_pending`; inspect it and call "
    "`cdp('Page.handleJavaScriptDialog', accept=...)` before continuing. "
    "Before giving a final answer on a "
    "multi-item task, verify the collected count against what was asked "
    "and go back for anything missing.\n\n"
    "Batch each sub-procedure (navigate, wait, extract, act) into one call "
    "— do not spend a call per action — but for long extractions prefer "
    "several medium calls that append to workspace files over one giant "
    "call, so progress survives timeouts. For an isolated concurrent "
    "browser session (parallel tasks that must not share tabs), pass "
    "session=<name> (never BU_NAME env syntax) and reuse the same name on "
    "every related call."
)

_HEADER_VISION = (
    " Screenshots are attached to your context automatically: when the exec "
    "output contains a capture_screenshot() path, the image arrives with "
    "this tool's result and you inspect it directly with your own vision — "
    "never send browser screenshots to a separate vision tool."
)

_HEADER_TEXT_ONLY = (
    " Your model cannot view images, so work text-first: page_info() for "
    "state, js() for reading/extracting DOM text, fill_input(selector, "
    "text) for inputs, and js(\"document.querySelector('…').click()\") for "
    "clicks — skip the screenshot-driven workflow described below."
)

_DESCRIPTION_HEADER = _HEADER_BASE  # back-compat alias for external imports

# NOTE: browser_exec is additionally gated at tool-definition time — sessions
# whose resolved toolsets do not include ``terminal`` never see it (see
# model_tools._compute_tool_definitions). The check_fn registered below only
# answers "is Browser Use mode configured"; surface policy lives with the
# session, not in the process-wide TTL-cached check_fn.


def _description_header() -> str:
    """Header tailored to whether the active model can see images natively"""
    try:
        from tools.vision_tools import _should_use_native_vision_fast_path

        if _should_use_native_vision_fast_path():
            return _HEADER_BASE + _HEADER_VISION
    except Exception:
        pass
    return _HEADER_BASE + _HEADER_TEXT_ONLY

_skill_text_cache: Optional[str] = None
_skill_text_fetched = False

# Pinned quick-reference for the CLI's pre-imported helpers. Replaces the
# live ``browser-use skill`` fetch: embedding whatever text the installed CLI
# version prints would ship uncontrolled third-party content into every
# session's system-side schema (version drift across machines, supply-chain
# exposure, and a byte-unstable prompt). A/B benchmarked Aug 2026 (108 runs,
# opus-4.8 + kimi-k3, 6 multi-step tasks x 3 reps): header-only schema went
# 36/36 vs 36/36 for the full skill dump at ~equal tokens (-60% vs the
# legacy browser_* toolset either way). The pinned digest below keeps the
# first-call reliability of the helper names without the 7.7KB dump.
_HELPERS_DIGEST = (
    "\n\nHELPERS (pre-imported): new_tab(url) opens/navigates (use for the "
    "FIRST navigation), goto_url(url) navigates the current tab, "
    "wait_for_load() after navigation, page_info() summarizes the current "
    "page state, js(expr) evaluates a JS expression and returns its value "
    "(js('document.title'); wrap function bodies as js('(() => {...})()') — "
    "a bare '() => {...}' returns the function itself, uncalled), "
    "fill_input(selector, text) types into inputs, click_at_xy(x, y) clicks "
    "viewport coordinates, capture_screenshot() saves and prints a "
    "screenshot path, cdp('Domain.method', **kwargs) is raw CDP — "
    "cdp('Accessibility.getFullAXTree')['nodes'] lists every element's "
    "role/name/backendDOMNodeId (filter in Python before printing; it is "
    "thousands of nodes), then cdp('DOM.getBoxModel', backendNodeId=n) gives "
    "click coordinates. ensure_real_tab() recovers from a stale/internal "
    "tab. Login walls: stop and ask the user; never guess credentials."
)


def _cli_skill_text() -> str:
    """Deprecated: always returns "" — the schema uses the pinned header.

    Kept so tests and any external callers keep importing a stable symbol;
    see _HELPERS_DIGEST for the rationale (benchmark-backed removal of the
    live ``browser-use skill`` fetch).
    """
    return _skill_text_cache or ""


def _dynamic_schema_overrides() -> dict:
    return {"description": _description_header() + _HELPERS_DIGEST}


BROWSER_EXEC_SCHEMA = {
    "name": "browser_exec",
    # Static fallback, used only when the CLI (and uvx) is unavailable
    "description": (
        _HEADER_BASE
        + _HELPERS_DIGEST
        + "\n\n(The browser-use CLI is not installed yet. Install it with "
        "`uv tool install browser-use`.)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute using the pre-imported browser helpers. Use print(...) for any data you need back.",
            },
            "session": {
                "type": "string",
                "description": "Named daemon/tab isolation (sets BU_NAME): each name gets its own harness daemon, and Hermes pins a separate tab on shared local/CDP browsers; cloud providers may additionally allocate a separate browser. This does not universally create a separate browser profile or storage partition. Reusing a name reconnects to that logical session across tasks; omitting it uses a task-scoped default runtime.",
            },
            "timeout_s": {
                "type": "integer",
                "description": f"Max seconds to wait for the code to finish (default {_DEFAULT_TIMEOUT_S}, max {_MAX_TIMEOUT_S}).",
                "default": _DEFAULT_TIMEOUT_S,
            },
        },
        "required": ["code"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="browser_exec",
    toolset="browser-use",
    schema=BROWSER_EXEC_SCHEMA,
    handler=lambda args, **kw: browser_exec(
        code=args.get("code", ""),
        session=args.get("session", "") or "",
        timeout_s=args.get("timeout_s", _DEFAULT_TIMEOUT_S),
        task_id=kw.get("task_id"),
    ),
    check_fn=is_browser_use_cli_mode,
    dynamic_schema_overrides=_dynamic_schema_overrides,
    emoji="🌐",
)
