"""Tests for parallel model-catalog prefetch and thread-safe cache writes.

Regression tests for the serial /v1/models bottleneck: when the 1h disk cache
lapses, ``list_authenticated_providers()`` previously fetched each authed
provider's model list serially. With 10+ providers this stacked to 15-30s of
blocking HTTP round-trips. The parallel prefetch warms stale cache entries
concurrently via ThreadPoolExecutor before the serial picker loop starts.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_prefetch_singleflight():
    from hermes_cli import model_switch
    import hermes_cli.models as models_mod

    prefetch_lock = getattr(model_switch, "_prefetch_singleflight_lock", None)
    prefetch_state = getattr(model_switch, "_prefetch_singleflight", None)
    if prefetch_lock is not None and prefetch_state is not None:
        with prefetch_lock:
            prefetch_state.clear()
    with models_mod._swr_refresh_lock:
        models_mod._swr_refresh_inflight.clear()
    cursor_lock = getattr(models_mod, "_cursor_refresh_lock", None)
    if cursor_lock is not None:
        with cursor_lock:
            inflight = getattr(models_mod, "_cursor_refresh_inflight", None)
            failed_at = getattr(models_mod, "_cursor_refresh_failed_at", None)
            if inflight is not None:
                inflight.clear()
            if failed_at is not None:
                failed_at.clear()
    yield
    if prefetch_lock is not None and prefetch_state is not None:
        with prefetch_lock:
            prefetch_state.clear()
    with models_mod._swr_refresh_lock:
        models_mod._swr_refresh_inflight.clear()
    if cursor_lock is not None:
        with cursor_lock:
            inflight = getattr(models_mod, "_cursor_refresh_inflight", None)
            failed_at = getattr(models_mod, "_cursor_refresh_failed_at", None)
            if inflight is not None:
                inflight.clear()
            if failed_at is not None:
                failed_at.clear()


# ---------------------------------------------------------------------------
# Thread-safe cache entry update (hermes_cli/models.py)
# ---------------------------------------------------------------------------

class TestUpdateProviderCacheEntry:
    """Verify ``update_provider_cache_entry`` writes safely under concurrency."""

    def test_writes_new_entry(self, tmp_path, monkeypatch):
        """A new entry is persisted to the cache file."""
        import hermes_cli.models as mod

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)

        with patch.object(mod, "_credential_fingerprint", return_value="fp1"):
            mod.update_provider_cache_entry("openrouter", ["m1", "m2"])

        cache = mod._load_provider_models_cache()
        assert "openrouter" in cache
        assert cache["openrouter"]["models"] == ["m1", "m2"]
        assert cache["openrouter"]["fp"] == "fp1"

    def test_does_not_clobber_other_entries(self, tmp_path, monkeypatch):
        """Concurrent writes to different providers don't lose entries."""
        import hermes_cli.models as mod

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)

        # Seed with one entry
        with patch.object(mod, "_credential_fingerprint", return_value="fp_a"):
            mod.update_provider_cache_entry("provider_a", ["a1"])

        # Write a second entry
        with patch.object(mod, "_credential_fingerprint", return_value="fp_b"):
            mod.update_provider_cache_entry("provider_b", ["b1"])

        cache = mod._load_provider_models_cache()
        assert "provider_a" in cache
        assert cache["provider_a"]["models"] == ["a1"]
        assert "provider_b" in cache
        assert cache["provider_b"]["models"] == ["b1"]

    def test_skips_empty_models(self, tmp_path, monkeypatch):
        """Empty model lists are not written to cache."""
        import hermes_cli.models as mod

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)

        mod.update_provider_cache_entry("empty_provider", [])
        cache = mod._load_provider_models_cache()
        assert "empty_provider" not in cache

    def test_concurrent_writes_no_lost_entries(self, tmp_path, monkeypatch):
        """Multiple threads writing different providers concurrently — all land."""
        import hermes_cli.models as mod
        import concurrent.futures

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)

        providers = [f"prov_{i}" for i in range(10)]

        with patch.object(mod, "_credential_fingerprint", side_effect=lambda p: f"fp_{p}"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                list(executor.map(
                    lambda p: mod.update_provider_cache_entry(p, [f"model_{p}"]),
                    providers,
                ))

        cache = mod._load_provider_models_cache()
        for p in providers:
            assert p in cache, f"{p} was lost in concurrent write"
            assert cache[p]["models"] == [f"model_{p}"]


# ---------------------------------------------------------------------------
# Parallel prefetch (hermes_cli/model_switch.py)
# ---------------------------------------------------------------------------

def _install_non_cursor_waiter_probe(model_switch):
    """Return a lock wrapper that signals when a waiter sees an in-flight slug."""
    real_lock = model_switch._prefetch_singleflight_lock
    waiter_saw_inflight = threading.Event()

    class _GuardLock:
        def __enter__(self):
            real_lock.acquire()
            if model_switch._prefetch_singleflight:
                waiter_saw_inflight.set()
            return self

        def __exit__(self, exc_type, exc, tb):
            real_lock.release()
            return False

    return _GuardLock(), waiter_saw_inflight


class TestPrefetchProviderModelsParallel:
    """Verify ``_prefetch_provider_models_parallel`` fetches concurrently."""

    def test_prefetch_executor_uses_callers_profile_context(self, tmp_path, monkeypatch):
        """Executor work retains the profile home and scoped Cursor credential."""
        from agent.secret_scope import (
            get_secret,
            is_multiplex_active,
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        was_multiplexed = is_multiplex_active()
        home = tmp_path / "profile-a"
        observed = []
        monkeypatch.setenv("CURSOR_API_KEY", "process-only-sentinel")
        set_multiplex_active(True)
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope({"CURSOR_API_KEY": "scoped-a"})
        try:
            def fetch(slug, force_refresh=False):
                observed.append((slug, get_hermes_home(), get_secret("CURSOR_API_KEY")))
                return ["live-cursor"]

            with patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
                 patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
                 patch("hermes_cli.models.cached_provider_model_ids", side_effect=fetch):
                _prefetch_provider_models_parallel(["cursor"])
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)
            set_multiplex_active(was_multiplexed)

        assert observed == [("cursor", home, "scoped-a")]

    def test_skips_all_fresh_entries(self, monkeypatch):
        """When all cache entries are fresh, no fetch is made."""
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        fresh_cache = {
            "openrouter": {"fp": "fp", "at": time.time(), "models": ["m1"]},
            "anthropic": {"fp": "fp", "at": time.time(), "models": ["m2"]},
        }

        with patch("hermes_cli.models._load_provider_models_cache", return_value=fresh_cache), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids") as fetch:
            _prefetch_provider_models_parallel(["openrouter", "anthropic"])

        fetch.assert_not_called()

    def test_fetches_only_stale_entries(self, monkeypatch):
        """Only providers with stale/missing cache entries are fetched."""
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache = {
            "fresh_prov": {"fp": "fp_f", "at": time.time(), "models": ["m1"]},
        }

        fetch_calls = []

        def mock_fetch(slug, force_refresh=False):
            fetch_calls.append(slug)
            return [f"model_{slug}"]

        with patch("hermes_cli.models._load_provider_models_cache", return_value=cache), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp_f"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch), \
             patch("hermes_cli.models.update_provider_cache_entry"):
            _prefetch_provider_models_parallel(["fresh_prov", "stale_prov"])

        assert "fresh_prov" not in fetch_calls
        assert "stale_prov" in fetch_calls

    def test_cursor_stale_after_five_minutes_while_same_age_non_cursor_stays_fresh(self):
        """A 6-minute Cursor row is prefetched; an equally aged OpenRouter row is not."""
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        age = 301
        cache = {
            "cursor": {"fp": "fp", "at": time.time() - age, "models": ["cursor-old"]},
            "openrouter": {"fp": "fp", "at": time.time() - age, "models": ["or-old"]},
        }
        fetch_calls = []

        def mock_fetch(slug, force_refresh=False):
            fetch_calls.append((slug, force_refresh))
            return [f"model_{slug}"]

        with patch("hermes_cli.models._load_provider_models_cache", return_value=cache), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch), \
             patch("hermes_cli.models.update_provider_cache_entry"):
            _prefetch_provider_models_parallel(["cursor", "openrouter"])

        assert fetch_calls == [("cursor", True)]

    def test_fetches_in_parallel(self, monkeypatch):
        """Multiple providers are fetched concurrently, not serially."""
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        # Track overlap: if serial, no two fetches should overlap in time.
        active = []
        max_concurrent = [0]
        lock = __import__("threading").Lock()

        def mock_fetch(slug, force_refresh=False):
            with lock:
                active.append(slug)
                max_concurrent[0] = max(max_concurrent[0], len(active))
            time.sleep(0.05)  # simulate network latency
            with lock:
                active.remove(slug)
            return [f"model_{slug}"]

        slugs = [f"prov_{i}" for i in range(6)]

        with patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch), \
             patch("hermes_cli.models.update_provider_cache_entry"):
            _prefetch_provider_models_parallel(slugs)

        assert max_concurrent[0] > 1, "fetches were serial, not parallel"

    def test_swallows_exceptions(self):
        """A failing provider fetch doesn't raise — best-effort."""
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        def mock_fetch(slug, force_refresh=False):
            raise ConnectionError("simulated network failure")

        with patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch), \
             patch("hermes_cli.models.update_provider_cache_entry"):
            # Should not raise
            _prefetch_provider_models_parallel(["failing_prov"])

    def test_empty_list_is_noop(self):
        """Empty provider list does nothing."""
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        with patch("hermes_cli.models.cached_provider_model_ids") as fetch:
            _prefetch_provider_models_parallel([])
        fetch.assert_not_called()

    def test_overlapping_stale_openrouter_prefetches_share_one_forced_discovery(self):
        """Two overlapping stale OpenRouter prefetches share one forced discovery."""
        from hermes_cli import model_switch
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        release = threading.Event()
        entered = threading.Event()
        start = threading.Barrier(2)
        calls = []
        lock = threading.Lock()
        guard_lock, waiter_saw_inflight = _install_non_cursor_waiter_probe(model_switch)

        def mock_fetch(slug, force_refresh=False):
            with lock:
                calls.append((slug, force_refresh))
            entered.set()
            assert release.wait(timeout=2)
            return ["or-live"]

        def worker():
            start.wait(timeout=2)
            _prefetch_provider_models_parallel(["openrouter"])

        with patch.object(model_switch, "_prefetch_singleflight_lock", guard_lock), \
             patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch), \
             patch("hermes_cli.models.update_provider_cache_entry"):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            assert entered.wait(timeout=2)
            assert waiter_saw_inflight.wait(timeout=2)
            assert all(thread.is_alive() for thread in threads)
            release.set()
            for thread in threads:
                thread.join(timeout=2)
            assert all(not thread.is_alive() for thread in threads)

        assert calls == [("openrouter", True)]

    def test_overlapping_stale_openrouter_prefetches_share_one_forced_discovery_on_failure(self):
        """Waiters share a failing OpenRouter owner attempt and still terminate."""
        from hermes_cli import model_switch
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        release = threading.Event()
        entered = threading.Event()
        start = threading.Barrier(2)
        calls = []
        lock = threading.Lock()
        guard_lock, waiter_saw_inflight = _install_non_cursor_waiter_probe(model_switch)

        def mock_fetch(slug, force_refresh=False):
            with lock:
                calls.append((slug, force_refresh))
            entered.set()
            assert release.wait(timeout=2)
            raise ConnectionError("simulated openrouter failure")

        def worker():
            start.wait(timeout=2)
            _prefetch_provider_models_parallel(["openrouter"])

        with patch.object(model_switch, "_prefetch_singleflight_lock", guard_lock), \
             patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch), \
             patch("hermes_cli.models.update_provider_cache_entry"):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            assert entered.wait(timeout=2)
            assert waiter_saw_inflight.wait(timeout=2)
            assert all(thread.is_alive() for thread in threads)
            release.set()
            for thread in threads:
                thread.join(timeout=2)
            assert all(not thread.is_alive() for thread in threads)

        assert calls == [("openrouter", True)]

    def test_cursor_prefetch_does_not_enter_generic_singleflight(self):
        """Cursor stays on the models.py coordinator, not the generic guard."""
        from hermes_cli import model_switch
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        release = threading.Event()
        entered = threading.Event()
        saw_cursor_in_guard = []

        def mock_fetch(slug, force_refresh=False):
            state = getattr(model_switch, "_prefetch_singleflight", None)
            lock = getattr(model_switch, "_prefetch_singleflight_lock", None)
            if state is None or lock is None:
                saw_cursor_in_guard.append(False)
            else:
                with lock:
                    saw_cursor_in_guard.append("cursor" in state)
            entered.set()
            assert release.wait(timeout=2)
            return ["cursor-live"]

        worker_done = threading.Event()

        def worker():
            _prefetch_provider_models_parallel(["cursor"])
            worker_done.set()

        with patch("hermes_cli.models._load_provider_models_cache", return_value={}), \
             patch("hermes_cli.models._credential_fingerprint", return_value="fp"), \
             patch("hermes_cli.models.cached_provider_model_ids", side_effect=mock_fetch), \
             patch("hermes_cli.models.update_provider_cache_entry"):
            thread = threading.Thread(target=worker)
            thread.start()
            assert entered.wait(timeout=2)
            assert thread.is_alive()
            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert worker_done.is_set()

        assert saw_cursor_in_guard == [False]


def _write_provider_cache(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_provider_cache(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_cursor_fetch_models(monkeypatch, result):
    from providers import get_provider_profile

    profile = get_provider_profile("cursor")
    assert profile is not None
    if callable(result):
        monkeypatch.setattr(profile, "fetch_models", result)
    else:
        monkeypatch.setattr(profile, "fetch_models", lambda **_kwargs: result)
    return profile


class TestPrefetchCursorFallbackAndCoalesce:
    """Composed prefetch must preserve a named Cursor catalog on discovery
    failure, coalesce overlapping same-provider refreshes, and still expose a
    successful live catalog on the ordinary same-pass read.
    """

    def test_prefetch_cursor_failure_preserves_named_entry_and_freshness(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        original_at = time.time() - 400
        fp = mod._credential_fingerprint("cursor")
        _write_provider_cache(cache_path, {
            "cursor": {"fp": fp, "at": original_at, "models": ["named-a", "named-b"]},
        })
        _patch_cursor_fetch_models(monkeypatch, None)

        _prefetch_provider_models_parallel(["cursor"])

        saved = _read_provider_cache(cache_path)
        assert saved["cursor"]["models"] == ["named-a", "named-b"]
        assert saved["cursor"]["at"] == original_at

    def test_overlapping_same_provider_prefetches_share_one_live_attempt_on_success(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        fp = mod._credential_fingerprint("cursor")
        _write_provider_cache(cache_path, {
            "cursor": {
                "fp": fp,
                "at": time.time() - 400,
                "models": ["named-a", "named-b"],
            },
        })

        release = threading.Event()
        entered_live = threading.Event()
        live_calls = []
        start_prefetch = threading.Barrier(2)

        def fake_fetch(**_kwargs):
            live_calls.append("cursor")
            entered_live.set()
            assert release.wait(timeout=2)
            return ["named-new"]

        _patch_cursor_fetch_models(monkeypatch, fake_fetch)

        def worker():
            start_prefetch.wait(timeout=2)
            _prefetch_provider_models_parallel(["cursor"])

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        assert entered_live.wait(timeout=2)
        assert all(thread.is_alive() for thread in threads)
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        assert all(not thread.is_alive() for thread in threads)
        assert live_calls == ["cursor"]
        assert _read_provider_cache(cache_path)["cursor"]["models"] == ["named-new"]

    def test_overlapping_same_provider_prefetches_share_one_live_attempt_on_failure(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        original_at = time.time() - 400
        fp = mod._credential_fingerprint("cursor")
        _write_provider_cache(cache_path, {
            "cursor": {"fp": fp, "at": original_at, "models": ["named-a", "named-b"]},
        })

        release = threading.Event()
        entered_live = threading.Event()
        live_calls = []
        start_prefetch = threading.Barrier(2)

        def fake_fetch(**_kwargs):
            live_calls.append("cursor")
            entered_live.set()
            assert release.wait(timeout=2)
            return None

        _patch_cursor_fetch_models(monkeypatch, fake_fetch)

        def worker():
            start_prefetch.wait(timeout=2)
            _prefetch_provider_models_parallel(["cursor"])

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        assert entered_live.wait(timeout=2)
        assert all(thread.is_alive() for thread in threads)
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        assert all(not thread.is_alive() for thread in threads)
        assert live_calls == ["cursor"]
        saved = _read_provider_cache(cache_path)
        assert saved["cursor"]["models"] == ["named-a", "named-b"]
        assert saved["cursor"]["at"] == original_at

    def test_different_provider_prefetches_remain_parallel(self, tmp_path, monkeypatch):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        overlap = threading.Barrier(2)

        def cursor_fetch(**_kwargs):
            overlap.wait(timeout=2)
            return ["cursor-live"]

        _patch_cursor_fetch_models(monkeypatch, cursor_fetch)

        def openrouter_ids(provider, *, force_refresh=False):
            if str(provider) == "openrouter":
                overlap.wait(timeout=2)
                return ["or-live"]
            return real_provider_model_ids(provider, force_refresh=force_refresh)

        real_provider_model_ids = mod.provider_model_ids
        monkeypatch.setattr(mod, "provider_model_ids", openrouter_ids)

        def run_cursor():
            _prefetch_provider_models_parallel(["cursor"])

        def run_openrouter():
            _prefetch_provider_models_parallel(["openrouter"])

        threads = [
            threading.Thread(target=run_cursor),
            threading.Thread(target=run_openrouter),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        assert all(not thread.is_alive() for thread in threads)

    def test_successful_cursor_prefetch_visible_on_ordinary_same_pass_read(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        fp = mod._credential_fingerprint("cursor")
        _write_provider_cache(cache_path, {
            "cursor": {
                "fp": fp,
                "at": time.time() - 400,
                "models": ["named-a", "named-b"],
            },
        })
        _patch_cursor_fetch_models(monkeypatch, ["named-new-a", "named-new-b"])

        _prefetch_provider_models_parallel(["cursor"])
        out = mod.cached_provider_model_ids("cursor")

        assert out == ["named-new-a", "named-new-b"]


# ---------------------------------------------------------------------------
# Integration: prefetch is called from list_authenticated_providers
# ---------------------------------------------------------------------------

class TestPrefetchIntegration:
    """Verify ``list_authenticated_providers`` triggers parallel prefetch."""

    def test_prefetch_called_with_more_than_3_providers(self):
        """When >3 providers are authed, parallel prefetch is invoked."""
        from hermes_cli import model_switch

        slugs = [f"prov_{i}" for i in range(5)]
        captured_slugs = []

        def mock_collect(data, curated, excluded):
            return slugs

        with patch.object(model_switch, "_collect_authed_provider_slugs", side_effect=mock_collect), \
             patch.object(model_switch, "_prefetch_provider_models_parallel") as prefetch:
            try:
                model_switch.list_authenticated_providers()
            except Exception:
                pass  # we only care about the prefetch call
            captured_slugs = prefetch.call_args[0][0] if prefetch.called else []

        assert prefetch.called
        assert captured_slugs == slugs

    def test_prefetch_skipped_with_3_or_fewer_providers(self):
        """When ≤3 providers are authed, parallel prefetch is skipped."""
        from hermes_cli import model_switch

        slugs = ["prov_a", "prov_b"]

        def mock_collect(data, curated, excluded):
            return slugs

        with patch.object(model_switch, "_collect_authed_provider_slugs", side_effect=mock_collect), \
             patch.object(model_switch, "_prefetch_provider_models_parallel") as prefetch:
            try:
                model_switch.list_authenticated_providers()
            except Exception:
                pass

        prefetch.assert_not_called()

    def test_prefetch_cursor_even_with_three_or_fewer_providers(self):
        """A configured Cursor provider still enters bounded prefetch at ≤3 authed providers."""
        from hermes_cli import model_switch

        slugs = ["openrouter", "cursor"]

        def mock_collect(data, curated, excluded):
            return slugs

        with patch.object(model_switch, "_collect_authed_provider_slugs", side_effect=mock_collect), \
             patch.object(model_switch, "_prefetch_provider_models_parallel") as prefetch:
            try:
                model_switch.list_authenticated_providers()
            except Exception:
                pass

        prefetch.assert_called_once()
        assert prefetch.call_args[0][0] == ["cursor"]

    def test_prefetch_skipped_on_refresh(self):
        """When refresh=True, prefetch is skipped (serial path force-refreshes)."""
        from hermes_cli import model_switch

        with patch.object(model_switch, "_collect_authed_provider_slugs") as collect, \
             patch.object(model_switch, "_prefetch_provider_models_parallel") as prefetch:
            try:
                model_switch.list_authenticated_providers(refresh=True)
            except Exception:
                pass

        collect.assert_not_called()
        prefetch.assert_not_called()


class _FrozenClock:
    def __init__(self, now: float):
        self.now = now

    def time(self) -> float:
        return self.now


_REAL_THREAD = threading.Thread


class _InlineSWRThread(_REAL_THREAD):
    """Run Cursor SWR daemon work inline; leave executor threads real."""

    def start(self):
        if str(self.name or "").startswith("model-cache-swr-"):
            self.run()
            return
        _REAL_THREAD.start(self)


def _install_inline_swr_threads(monkeypatch):
    import hermes_cli.models as mod

    monkeypatch.setattr(mod.threading, "Thread", _InlineSWRThread)


def _install_clock(monkeypatch, clock: _FrozenClock):
    import hermes_cli.models as mod
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(mod.time, "time", clock.time)
    monkeypatch.setattr(model_switch.time, "time", clock.time)


def _counting_cursor_fetch(monkeypatch, live_calls, result=None, hold=None):
    def fake_fetch(**_kwargs):
        live_calls.append("cursor")
        if hold is not None:
            entered, release = hold
            entered.set()
            assert release.wait(timeout=2)
        if callable(result):
            return result()
        return result

    _patch_cursor_fetch_models(monkeypatch, fake_fetch)


class TestCursorAutomatedRefreshCoordinator:
    """Prefetch, Cursor SWR, and the ordinary cached read must share one
    automated discovery attempt until the next TTL boundary.
    """

    def test_stale_prefetch_then_ordinary_read_after_failure_is_one_raw_fetch(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        _install_inline_swr_threads(monkeypatch)
        original_at = time.time() - 400
        fp = mod._credential_fingerprint("cursor")
        _write_provider_cache(cache_path, {
            "cursor": {"fp": fp, "at": original_at, "models": ["named-a", "named-b"]},
        })
        live_calls = []
        _counting_cursor_fetch(monkeypatch, live_calls, result=None)

        _prefetch_provider_models_parallel(["cursor"])
        out = mod.cached_provider_model_ids("cursor")

        assert out == ["named-a", "named-b"]
        saved = _read_provider_cache(cache_path)
        assert saved["cursor"]["models"] == ["named-a", "named-b"]
        assert saved["cursor"]["at"] == original_at
        assert live_calls == ["cursor"]

    def test_cold_prefetch_then_ordinary_read_after_failure_is_one_raw_fetch(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        _install_inline_swr_threads(monkeypatch)
        live_calls = []
        _counting_cursor_fetch(monkeypatch, live_calls, result=None)

        _prefetch_provider_models_parallel(["cursor"])
        out = mod.cached_provider_model_ids("cursor")

        assert out == ["auto"]
        if cache_path.exists():
            saved = _read_provider_cache(cache_path)
            assert "cursor" not in saved
        assert live_calls == ["cursor"]

    def test_failed_attempt_suppressed_at_299s_retries_at_301s(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        _install_inline_swr_threads(monkeypatch)
        clock = _FrozenClock(1_000_000.0)
        _install_clock(monkeypatch, clock)
        original_at = clock.now - 400
        fp = "fp"
        monkeypatch.setattr(mod, "_credential_fingerprint", lambda _p: fp)
        _write_provider_cache(cache_path, {
            "cursor": {"fp": fp, "at": original_at, "models": ["named-a"]},
        })
        live_calls = []
        _counting_cursor_fetch(monkeypatch, live_calls, result=None)

        _prefetch_provider_models_parallel(["cursor"])
        assert live_calls == ["cursor"]

        clock.now += 299
        assert mod.cached_provider_model_ids("cursor") == ["named-a"]
        _prefetch_provider_models_parallel(["cursor"])
        assert live_calls == ["cursor"]

        clock.now += 2
        _prefetch_provider_models_parallel(["cursor"])
        assert live_calls == ["cursor", "cursor"]
        assert _read_provider_cache(cache_path)["cursor"]["at"] == original_at

    def test_credential_fingerprint_change_retries_immediately(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        _install_inline_swr_threads(monkeypatch)
        fps = ["fp-old"]
        monkeypatch.setattr(mod, "_credential_fingerprint", lambda _p: fps[0])
        _write_provider_cache(cache_path, {
            "cursor": {
                "fp": "fp-old",
                "at": time.time() - 400,
                "models": ["named-old"],
            },
        })
        live_calls = []
        _counting_cursor_fetch(monkeypatch, live_calls, result=None)

        _prefetch_provider_models_parallel(["cursor"])
        assert mod.cached_provider_model_ids("cursor") == ["named-old"]
        assert live_calls == ["cursor"]

        fps[0] = "fp-new"
        out = mod.cached_provider_model_ids("cursor")
        assert out == ["auto"]
        assert live_calls == ["cursor", "cursor"]

    def test_manual_cache_clear_retries_immediately(self, tmp_path, monkeypatch):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        _install_inline_swr_threads(monkeypatch)
        live_calls = []
        _counting_cursor_fetch(monkeypatch, live_calls, result=None)

        _prefetch_provider_models_parallel(["cursor"])
        assert mod.cached_provider_model_ids("cursor") == ["auto"]
        assert live_calls == ["cursor"]

        mod.clear_provider_models_cache("cursor")
        out = mod.cached_provider_model_ids("cursor")
        assert out == ["auto"]
        assert live_calls == ["cursor", "cursor"]

    def test_overlapping_stale_and_cold_callers_share_one_failure(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        original_at = time.time() - 400
        fp = mod._credential_fingerprint("cursor")
        _write_provider_cache(cache_path, {
            "cursor": {"fp": fp, "at": original_at, "models": ["named-a", "named-b"]},
        })
        live_calls = []
        entered = threading.Event()
        release = threading.Event()
        start = threading.Barrier(3)
        _counting_cursor_fetch(
            monkeypatch, live_calls, result=None, hold=(entered, release)
        )
        results = {}

        def run_prefetch():
            start.wait(timeout=2)
            _prefetch_provider_models_parallel(["cursor"])
            results["prefetch"] = True

        def run_ordinary_stale():
            start.wait(timeout=2)
            results["stale"] = mod.cached_provider_model_ids("cursor")

        def run_ordinary_after_clear():
            start.wait(timeout=2)
            results["cold"] = mod.cached_provider_model_ids("cursor")

        threads = [
            threading.Thread(target=run_prefetch),
            threading.Thread(target=run_ordinary_stale),
            threading.Thread(target=run_ordinary_after_clear),
        ]
        for thread in threads:
            thread.start()
        assert entered.wait(timeout=2)
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        assert all(not thread.is_alive() for thread in threads)
        assert live_calls == ["cursor"]
        assert results["stale"] == ["named-a", "named-b"]
        assert results["cold"] in (["named-a", "named-b"], ["auto"])
        saved = _read_provider_cache(cache_path)
        assert saved["cursor"]["models"] == ["named-a", "named-b"]
        assert saved["cursor"]["at"] == original_at

    def test_overlapping_cold_callers_share_one_success(self, tmp_path, monkeypatch):
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        live_calls = []
        entered = threading.Event()
        release = threading.Event()
        start = threading.Barrier(2)
        _counting_cursor_fetch(
            monkeypatch, live_calls, result=["named-new"], hold=(entered, release)
        )
        results = []

        def worker(kind):
            start.wait(timeout=2)
            if kind == "prefetch":
                _prefetch_provider_models_parallel(["cursor"])
                results.append(("prefetch", mod.cached_provider_model_ids("cursor")))
            else:
                results.append(("ordinary", mod.cached_provider_model_ids("cursor")))

        threads = [
            threading.Thread(target=worker, args=("prefetch",)),
            threading.Thread(target=worker, args=("ordinary",)),
        ]
        for thread in threads:
            thread.start()
        assert entered.wait(timeout=2)
        assert all(thread.is_alive() for thread in threads)
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        assert all(not thread.is_alive() for thread in threads)
        assert live_calls == ["cursor"]
        assert {pair[1][0] if pair[1] else None for pair in results} == {"named-new"}
        assert _read_provider_cache(cache_path)["cursor"]["models"] == ["named-new"]

    def test_anti_weakening_swr_or_ordinary_bypass_makes_composed_oracle_fail(
        self, tmp_path, monkeypatch
    ):
        """Bypassing the coordinator from SWR or the ordinary read must
        produce the 2 != 1 composed-oracle failure.
        """
        import hermes_cli.models as mod
        from hermes_cli.model_switch import _prefetch_provider_models_parallel

        cache_path = tmp_path / "provider_models_cache.json"
        monkeypatch.setattr(mod, "_provider_models_cache_path", lambda: cache_path)
        _install_inline_swr_threads(monkeypatch)
        fp = mod._credential_fingerprint("cursor")
        _write_provider_cache(cache_path, {
            "cursor": {
                "fp": fp,
                "at": time.time() - 400,
                "models": ["named-a"],
            },
        })
        live_calls = []
        _counting_cursor_fetch(monkeypatch, live_calls, result=None)

        def bypass_swr(cache_key, refresh_fn=None):
            if cache_key == "cursor":
                mod._cursor_discovered_models()

        monkeypatch.setattr(mod, "_spawn_swr_refresh", bypass_swr)
        _prefetch_provider_models_parallel(["cursor"])
        mod.cached_provider_model_ids("cursor")
        assert live_calls == ["cursor", "cursor"]

        live_calls.clear()
        if cache_path.exists():
            cache_path.unlink()
        monkeypatch.setattr(
            mod,
            "cached_provider_model_ids",
            lambda provider, *args, **kwargs: (
                mod._cursor_discovered_models() or list(mod._PROVIDER_MODELS.get("cursor", []))
            ),
        )
        _prefetch_provider_models_parallel(["cursor"])
        mod.cached_provider_model_ids("cursor")
        assert live_calls == ["cursor", "cursor"]
