"""Antigravity account and OAuth JSON-RPC handlers.

The gateway retains only one synchronous, plugin-local account service per
profile home. Each client attaches to the same installation-owned Antigravity
pool; profile-scoped RPC lifetimes do not isolate accounts. This adapter
only validates RPC shapes, delegates, and emits the
already materialized safe results.
"""
from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from typing import Any, Callable

from agent.antigravity_bridge_transport import AntigravityBridgeError
from hermes_constants import get_hermes_home
from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# These error values intentionally do not disclose bridge output or exception
# text.  -32602 is the established JSON-RPC invalid-parameters response.
_ANTIGRAVITY_INVALID_PARAMS = (-32602, "invalid Antigravity parameters")
_ANTIGRAVITY_BRIDGE_FAILURE = (5201, "Antigravity account service unavailable")
_ANTIGRAVITY_INTERNAL_FAILURE = (5202, "Antigravity account service failed")
_MAX_SAFE_INTEGER = (1 << 53) - 1
_antigravity_services: dict[str, Any] = {}
_antigravity_services_lock = threading.RLock()
_antigravity_services_condition = threading.Condition(_antigravity_services_lock)
_antigravity_services_pending: set[str] = set()
_antigravity_services_closures = 0
_antigravity_services_shutdown = False


def _antigravity_unbound_factory() -> Any:
    raise RuntimeError("Antigravity service factory has not been installed")


_antigravity_service_factory: Callable[[], Any] = _antigravity_unbound_factory
logger = logging.getLogger(__name__)


def _ok(rid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _antigravity_is_uuid(value: Any) -> bool:
    if type(value) is not str or len(value) != 36:
        return False
    if any(value[index] != "-" for index in (8, 13, 18, 23)):
        return False
    compact = value.replace("-", "")
    return (
        len(compact) == 32
        and all(char in "0123456789abcdef" for char in compact)
        and value[14] in "12345"
        and value[19] in "89ab"
    )


def _antigravity_is_account_id(value: Any) -> bool:
    return type(value) is str and value.startswith("acct_") and _antigravity_is_uuid(value[5:])


def _antigravity_is_project_id(value: Any) -> bool:
    return type(value) is str and len(value) <= 4096 and not any(
        ord(char) < 32 or ord(char) == 127 for char in value
    )


def _antigravity_profile_key() -> str:
    """Use the active profile-home context as the only service/client scope."""
    return str(Path(get_hermes_home()).resolve())


def _antigravity_default_service_factory():
    # Plugin discovery establishes the importable model_providers namespace
    # without changing the provider registry or the plugin ABI.
    import model_tools  # noqa: F401

    service_module = importlib.import_module(
        "plugins.model_providers.antigravity.account_service"
    )
    return service_module.AntigravityAccountService()


def _antigravity_service():
    """Return a profile-local service without racing shutdown admission."""
    global _antigravity_services_closures
    key = _antigravity_profile_key()
    with _antigravity_services_condition:
        while key in _antigravity_services_pending:
            _antigravity_services_condition.wait()
        if _antigravity_services_shutdown:
            raise RuntimeError("Antigravity account service is shutting down")
        service = _antigravity_services.get(key)
        if service is not None:
            return service
        _antigravity_services_pending.add(key)

    try:
        service = _antigravity_service_factory()
    except Exception:
        with _antigravity_services_condition:
            _antigravity_services_pending.remove(key)
            _antigravity_services_condition.notify_all()
        raise

    with _antigravity_services_condition:
        if not _antigravity_services_shutdown:
            _antigravity_services[key] = service
            _antigravity_services_pending.remove(key)
            _antigravity_services_condition.notify_all()
            return service
        _antigravity_services_closures += 1

    try:
        service.close()
    except Exception:
        logger.warning("Antigravity account service cleanup failed")
    with _antigravity_services_condition:
        _antigravity_services_pending.remove(key)
        _antigravity_services_closures -= 1
        _antigravity_services_condition.notify_all()
    raise RuntimeError("Antigravity account service is shutting down")


def _antigravity_failure(rid, exc: Exception) -> dict:
    if isinstance(exc, AntigravityBridgeError):
        code, message = _ANTIGRAVITY_BRIDGE_FAILURE
    else:
        code, message = _ANTIGRAVITY_INTERNAL_FAILURE
    return _err(rid, code, message)


def _antigravity_call(rid, callback):
    try:
        return callback()
    except Exception as exc:
        return _antigravity_failure(rid, exc)


def _shutdown_antigravity_services() -> None:
    """Close every service exactly once and reject later service admission."""
    global _antigravity_services_closures, _antigravity_services_shutdown
    with _antigravity_services_condition:
        _antigravity_services_shutdown = True
        services = list(_antigravity_services.values())
        _antigravity_services.clear()
        _antigravity_services_closures += len(services)
        _antigravity_services_condition.notify_all()
    for service in services:
        try:
            service.close()
        except Exception:
            logger.warning("Antigravity account service cleanup failed")
        with _antigravity_services_condition:
            _antigravity_services_closures -= 1
            _antigravity_services_condition.notify_all()
    with _antigravity_services_condition:
        while _antigravity_services_pending or _antigravity_services_closures:
            _antigravity_services_condition.wait()


@method("antigravity.accounts.list")
@_profile_scoped
def _(rid, params: dict) -> dict:
    return _antigravity_call(
        rid, lambda: _ok(rid, {"accounts": _antigravity_service().list_accounts()})
    )


@method("antigravity.accounts.enabled")
@_profile_scoped
def _(rid, params: dict) -> dict:
    account_id, enabled = params.get("account_id"), params.get("enabled")
    if not _antigravity_is_account_id(account_id) or type(enabled) is not bool:
        return _err(rid, *_ANTIGRAVITY_INVALID_PARAMS)
    return _antigravity_call(
        rid,
        lambda: _ok(
            rid,
            {"snapshot": _antigravity_service().set_account_enabled(account_id, enabled)},
        ),
    )


@method("antigravity.accounts.priority")
@_profile_scoped
def _(rid, params: dict) -> dict:
    account_id, priority = params.get("account_id"), params.get("priority")
    if (
        not _antigravity_is_account_id(account_id)
        or type(priority) is not int
        or not 1 <= priority <= _MAX_SAFE_INTEGER
    ):
        return _err(rid, *_ANTIGRAVITY_INVALID_PARAMS)
    return _antigravity_call(
        rid,
        lambda: _ok(
            rid,
            {"snapshot": _antigravity_service().set_account_priority(account_id, priority)},
        ),
    )


@method("antigravity.accounts.remove")
@_profile_scoped
def _(rid, params: dict) -> dict:
    account_id = params.get("account_id")
    if not _antigravity_is_account_id(account_id):
        return _err(rid, *_ANTIGRAVITY_INVALID_PARAMS)
    return _antigravity_call(
        rid, lambda: _ok(rid, {"snapshot": _antigravity_service().remove_account(account_id)})
    )


@method("antigravity.oauth.start")
@_profile_scoped
def _(rid, params: dict) -> dict:
    project_id = params.get("project_id", "")
    if not _antigravity_is_project_id(project_id):
        return _err(rid, *_ANTIGRAVITY_INVALID_PARAMS)
    return _antigravity_call(
        rid, lambda: _ok(rid, _antigravity_service().start_oauth(project_id))
    )


@method("antigravity.oauth.poll")
@_profile_scoped
def _(rid, params: dict) -> dict:
    session_id = params.get("session_id")
    if not _antigravity_is_uuid(session_id):
        return _err(rid, *_ANTIGRAVITY_INVALID_PARAMS)
    return _antigravity_call(
        rid, lambda: _ok(rid, _antigravity_service().poll_oauth(session_id))
    )


@method("antigravity.oauth.cancel")
@_profile_scoped
def _(rid, params: dict) -> dict:
    session_id = params.get("session_id")
    if not _antigravity_is_uuid(session_id):
        return _err(rid, *_ANTIGRAVITY_INVALID_PARAMS)
    return _antigravity_call(
        rid, lambda: _ok(rid, _antigravity_service().cancel_oauth(session_id))
    )


def register(server) -> None:
    """Bind handlers and profile-local lifecycle state onto ``server``."""
    server._antigravity_services = {}
    server._antigravity_services_lock = threading.RLock()
    server._antigravity_services_condition = threading.Condition(
        server._antigravity_services_lock
    )
    server._antigravity_services_pending = set()
    server._antigravity_services_closures = 0
    server._antigravity_services_shutdown = False
    server._antigravity_service_factory = _antigravity_default_service_factory
    server._ANTIGRAVITY_INVALID_PARAMS = _ANTIGRAVITY_INVALID_PARAMS
    server._ANTIGRAVITY_BRIDGE_FAILURE = _ANTIGRAVITY_BRIDGE_FAILURE
    server._ANTIGRAVITY_INTERNAL_FAILURE = _ANTIGRAVITY_INTERNAL_FAILURE
    server._MAX_SAFE_INTEGER = _MAX_SAFE_INTEGER
    from agent.antigravity_bridge_transport import AntigravityBridgeError

    server.AntigravityBridgeError = AntigravityBridgeError
    globals_to_rebind = (
        _antigravity_is_uuid,
        _antigravity_is_account_id,
        _antigravity_is_project_id,
        _antigravity_profile_key,
        _antigravity_default_service_factory,
        _antigravity_service,
        _antigravity_failure,
        _antigravity_call,
        _shutdown_antigravity_services,
    )
    namespace = vars(server)
    for helper in globals_to_rebind:
        setattr(
            server,
            helper.__name__,
            type(helper)(
                helper.__code__,
                namespace,
                helper.__name__,
                helper.__defaults__,
                helper.__closure__,
            ),
        )
    _registry.install(server)

    def validate_profile(handler):
        def wrapper(rid, params):
            profile = params.get("profile") if isinstance(params, dict) else None
            if profile is None or (isinstance(profile, str) and not profile.strip()):
                return handler(rid, params)
            if not isinstance(profile, str):
                return server._err(rid, *server._ANTIGRAVITY_INVALID_PARAMS)
            try:
                from hermes_cli import profiles as profiles_mod

                canonical = profiles_mod.normalize_profile_name(profile)
                profiles_mod.validate_profile_name(canonical)
                home = Path(profiles_mod.get_profile_dir(canonical)).resolve(strict=True)
                if canonical == "default":
                    if (
                        not home.is_dir()
                        or home != profiles_mod._get_default_hermes_home().resolve(strict=True)
                    ):
                        raise ValueError("default profile home mismatch")
                else:
                    root = profiles_mod._get_profiles_root().resolve(strict=True)
                    candidate = root / canonical
                    if (
                        candidate.parent != root
                        or candidate.is_symlink()
                        or not candidate.is_dir()
                        or home != candidate.resolve(strict=True)
                    ):
                        raise ValueError("profile home is not an accepted profile directory")
            except Exception:
                return server._err(rid, *server._ANTIGRAVITY_INVALID_PARAMS)
            scoped_params = dict(params)
            scoped_params["profile"] = canonical
            return handler(rid, scoped_params)

        return wrapper

    for name, handler in tuple(server._methods.items()):
        if name.startswith("antigravity."):
            server._methods[name] = validate_profile(handler)
