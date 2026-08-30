from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def antigravity_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _profile(monkeypatch):
    from providers import get_provider_profile

    profile = get_provider_profile("antigravity")
    assert profile is not None
    calls = []

    def fetch_models(**kwargs):
        calls.append(kwargs)
        return fetch_models.result

    fetch_models.result = None
    monkeypatch.setattr(profile, "fetch_models", fetch_models)
    return profile, calls, fetch_models


def _write_cache(home, models):
    (home / "provider_models_cache.json").write_text(
        json.dumps(
            {
                "antigravity": {
                    "fp": "stale",
                    "at": time.time(),
                    "models": models,
                }
            }
        ),
        encoding="utf-8",
    )


def test_live_antigravity_catalog_is_used_by_picker(antigravity_home, monkeypatch):
    from hermes_cli import models

    _profile_obj, calls, fetch = _profile(monkeypatch)
    fetch.result = ["antigravity-gemini-3-pro", "antigravity-claude-sonnet-4-6"]

    assert models.provider_model_ids("antigravity") == fetch.result
    assert calls == [{}]
    assert not (antigravity_home / "provider_models_cache.json").exists()


def test_live_empty_is_authoritative_for_direct_and_cached_picker(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    _profile_obj, calls, fetch = _profile(monkeypatch)
    _write_cache(antigravity_home, ["stale-antigravity-model"])
    fetch.result = []

    assert models.provider_model_ids("antigravity") == []
    assert models.cached_provider_model_ids("antigravity", force_refresh=True) == []
    assert len(calls) == 2
    saved = json.loads((antigravity_home / "provider_models_cache.json").read_text())
    assert saved["antigravity"]["models"] == ["stale-antigravity-model"]


def test_discovery_failure_uses_profile_fallback_without_cache_write(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    profile, calls, fetch = _profile(monkeypatch)
    assert fetch.result is None

    assert models.provider_model_ids("antigravity") == list(profile.fallback_models)
    assert models.cached_provider_model_ids("antigravity", force_refresh=True) == list(
        profile.fallback_models
    )
    assert len(calls) == 2
    assert not (antigravity_home / "provider_models_cache.json").exists()


def test_discovery_exception_uses_profile_fallback_without_leaking_error(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    profile = __import__("providers", fromlist=["get_provider_profile"]).get_provider_profile(
        "antigravity"
    )
    assert profile is not None
    calls = []

    def fetch_models(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("private bridge failure")

    monkeypatch.setattr(profile, "fetch_models", fetch_models)

    assert models.provider_model_ids("antigravity") == list(profile.fallback_models)
    assert models.cached_provider_model_ids("antigravity", force_refresh=True) == list(
        profile.fallback_models
    )
    assert len(calls) == 2
    assert not (antigravity_home / "provider_models_cache.json").exists()


def test_google_antigravity_alias_uses_antigravity_picker_without_stealing_google(
    antigravity_home, monkeypatch
):
    from hermes_cli import models

    _profile_obj, calls, fetch = _profile(monkeypatch)
    fetch.result = ["antigravity-gemini-live"]

    assert models.normalize_provider("google-antigravity") == "antigravity"
    assert models.provider_model_ids("google-antigravity") == fetch.result
    assert calls == [{}]
    assert models.normalize_provider("google") == "gemini"
    assert models.provider_model_ids("google") == models.provider_model_ids("gemini")


def test_cursor_and_api_key_provider_controls_keep_existing_paths(monkeypatch):
    from hermes_cli import models
    from providers import get_provider_profile

    cursor = get_provider_profile("cursor")
    assert cursor is not None
    monkeypatch.setattr(cursor, "fetch_models", lambda **kwargs: ["cursor-live"])
    assert models.provider_model_ids("cursor") == ["cursor-live"]

    api_key_profile = get_provider_profile("openrouter")
    if api_key_profile is not None:
        monkeypatch.setattr(
            api_key_profile,
            "fetch_models",
            lambda **kwargs: ["openrouter-live"],
        )
    # Static fallback remains available without credentials/network access.
    assert models.provider_model_ids("openrouter")
