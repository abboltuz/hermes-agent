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

    with model_switch._prefetch_singleflight_lock:
        model_switch._prefetch_singleflight.clear()
    yield
    with model_switch._prefetch_singleflight_lock:
        model_switch._prefetch_singleflight.clear()


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

class TestPrefetchProviderModelsParallel:
    """Verify ``_prefetch_provider_models_parallel`` fetches concurrently."""

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
