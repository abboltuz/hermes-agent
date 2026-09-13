"""Validate and freeze worker model pins without coupling profile configuration.

The legacy required-model sentinel opts a profile into explicit card routing.
Ordinary upstream profiles retain their existing model inheritance behavior.
"""

import json
import os
from pathlib import Path

REQUIRED_MODEL = "__KANBAN_CARD_MODEL_REQUIRED__"
WORKER_ROUTE_ENV = "HERMES_KANBAN_MODEL_ROUTE"


def _is_dynamic_provider(provider):
    from hermes_cli.providers import normalize_provider

    name = provider.strip().lower()
    if name.startswith("custom:"):
        name = name.split(":", 1)[1].strip() or "custom"
    # Cards carry no explicit endpoint. Bare custom therefore cannot identify
    # a strict route; use a named custom provider instead of ambient discovery.
    return normalize_provider(name) in {"auto", "main", "actual", "moa", "custom"}


def card_model_route(task, profile_home=None):
    model = str(getattr(task, "model_override", None) or "").strip()
    provider = str(getattr(task, "provider_override", None) or "").strip()
    required = False
    if profile_home:
        from hermes_cli.config import load_config_readonly

        path = Path(profile_home) / "config.yaml"
        # Dispatch must fail closed: a corrupt profile must not silently fall
        # back to defaults and bypass the required-route sentinel.
        config = load_config_readonly(path, strict=True)
        configured = config.get("model") or {}
        configured_model = configured.get("default") if isinstance(configured, dict) else configured
        required = configured_model == REQUIRED_MODEL
    if model == REQUIRED_MODEL or (required and (not model or not provider)):
        raise ValueError("Card requires an explicit provider and model before worker startup")
    if provider and not model:
        raise ValueError("Card provider requires a model")
    if model.lower() == "auto" or _is_dynamic_provider(provider):
        raise ValueError("Card model overrides must name a concrete route")
    # Model-only overrides and profile inheritance remain supported upstream.
    return {"provider": provider, "model": model} if provider and model else None


def worker_model_route():
    """Read the dispatcher's internal process-scoped route, never a user option."""
    raw = os.environ.get(WORKER_ROUTE_ENV)
    if not os.environ.get("HERMES_KANBAN_TASK") or not raw:
        return None
    route = json.loads(raw)
    if (not isinstance(route, dict) or set(route) != {"provider", "model"}
            or not all(isinstance(v, str) and v.strip() for v in route.values())):
        raise ValueError("Invalid worker model route snapshot")
    route = {key: value.strip() for key, value in route.items()}
    if (route["model"] == REQUIRED_MODEL or route["model"].lower() == "auto"
            or _is_dynamic_provider(route["provider"])):
        raise ValueError("Invalid worker model route snapshot")
    return route


def judge_route_for_task(task):
    """Prefer this worker's launch snapshot over a subsequently edited card."""
    if str(getattr(task, "id", "")) == os.environ.get("HERMES_KANBAN_TASK"):
        route = worker_model_route()
        if route:
            return route
    return card_model_route(task)
