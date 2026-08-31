# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportArgumentType=false
"""Tests for the v1.5.9 window-aware rate-limit auto-wait.

Before v1.5.9 the auto-wait used a single 6h safety cap
(``HERMES_RL_AUTOWAIT_MAX_S``) regardless of *which* Anthropic subscription
window tripped.  In practice that meant a weekly (7d) limit hit aborts
the run with "this is bigger than safety-cap" instead of waiting through
the reset — defeating the auto-wait for any agent that runs heavy enough
to exhaust a Pro plan's weekly cap.

v1.5.9 adds:
  * Detection of the throttled window via
    ``anthropic-ratelimit-unified-{5h,1d,7d}-status`` headers.
  * Per-window safety caps
    (``HERMES_RL_AUTOWAIT_MAX_5H_S`` / ``_1D_S`` / ``_7D_S``).
  * When multiple windows are throttled at once, the longest wait wins
    (sleeping through the 7d reset also clears the 5h window).

These tests exercise all of the above without touching the network.  Wall
clock is mocked via a ``time.sleep`` shim so the suite stays sub-second
even when exercising the wait loop.
"""

import time
from types import SimpleNamespace

import pytest

from anthropic_billing_bypass import (
    _rl_detect_throttled_window,
    _rl_parse_timestamp,
    _rl_sleep_until_reset,
    _rl_seconds_until_reset,
    _rl_window_cap,
    _RL_WINDOW_DEFAULTS_S,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_429_with_headers(headers: dict):
    """Build a fake RateLimitError carrying the given response headers."""
    resp = SimpleNamespace(headers=headers, status_code=429)

    class _FakeRateLimitError(Exception):
        status_code = 429
        response = resp

    return _FakeRateLimitError("rate limited")


def _make_agent():
    a = SimpleNamespace()
    a._touches = []  # type: ignore[attr-defined]
    a._interrupt_requested = False  # type: ignore[attr-defined]

    def _touch(desc):
        a._touches.append((time.time(), desc))  # type: ignore[attr-defined]

    a._touch_activity = _touch  # type: ignore[attr-defined]
    return a


@pytest.fixture
def fast_sleep(monkeypatch):
    """Replace ``time.sleep`` with a no-op so the wait loop returns instantly.

    We still need the *logic* to think it slept, so we also patch
    ``time.time`` to advance by 1s on every call — that way the
    ``deadline - time.time()`` check inside the loop terminates after a
    single iteration.
    """
    real_time = time.time
    real_monotonic = time.monotonic

    fake_now = [real_time()]

    def fake_sleep(secs):
        # Advance virtual clock faster than wall to break the deadline.
        fake_now[0] += max(secs, 60.0)  # jump well past any sub-second deadline

    def fake_time():
        return fake_now[0]

    def fake_monotonic():
        return fake_now[0]

    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setattr("anthropic_billing_bypass.time.sleep", fake_sleep)
    monkeypatch.setattr("time.time", fake_time)
    monkeypatch.setattr("anthropic_billing_bypass.time.time", fake_time)
    monkeypatch.setattr("time.monotonic", fake_monotonic)
    monkeypatch.setattr("anthropic_billing_bypass.time.monotonic", fake_monotonic)
    return fake_now


# ---------------------------------------------------------------------------
# _rl_parse_timestamp — lifted out of _rl_seconds_until_reset in v1.5.9.
# ---------------------------------------------------------------------------


def test_parse_timestamp_epoch_seconds_in_future():
    """Anthropic's unified-reset is an absolute epoch → return delta."""
    now = 1_700_000_000.0
    future = now + 3600
    assert _rl_parse_timestamp(str(future), now) == pytest.approx(3600.0)


def test_parse_timestamp_epoch_zero_treated_as_relative_zero():
    """A small raw integer like '0' is treated as 0 seconds (already reset)."""
    assert _rl_parse_timestamp("0", time.time()) == 0.0


def test_parse_timestamp_rfc3339():
    now = 1_700_000_000.0
    future_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + 3600))
    assert _rl_parse_timestamp(future_iso, now) == pytest.approx(3600.0, abs=1.0)


def test_parse_timestamp_garbage_returns_none():
    assert _rl_parse_timestamp("not a timestamp", time.time()) is None
    assert _rl_parse_timestamp("", time.time()) is None


# ---------------------------------------------------------------------------
# _rl_detect_throttled_window
# ---------------------------------------------------------------------------


def test_detect_weekly_throttle_wins_over_5h():
    """When both 5h and 7d are throttled, we pick the 7d wait (longest)."""
    now = time.time()
    future_5h = now + 600        # 10 min
    future_7d = now + 86_400 * 5  # 5 days
    headers = {
        "anthropic-ratelimit-unified-5h-status": "throttled",
        "anthropic-ratelimit-unified-5h-reset": str(int(future_5h)),
        "anthropic-ratelimit-unified-7d-status": "throttled",
        "anthropic-ratelimit-unified-7d-reset": str(int(future_7d)),
    }
    win, secs = _rl_detect_throttled_window(headers, now)
    assert win == "7d"
    assert secs == pytest.approx(86_400 * 5, abs=2.0)


def test_detect_5h_only():
    """When only 5h is throttled, we use the 5h window."""
    now = time.time()
    future_5h = now + 1800
    headers = {
        "anthropic-ratelimit-unified-5h-status": "throttled",
        "anthropic-ratelimit-unified-5h-reset": str(int(future_5h)),
        "anthropic-ratelimit-unified-7d-status": "allowed",
        "anthropic-ratelimit-unified-7d-reset": str(int(now + 86_400)),
    }
    win, secs = _rl_detect_throttled_window(headers, now)
    assert win == "5h"
    assert secs == pytest.approx(1800.0, abs=1.0)


def test_detect_no_throttle_returns_none():
    """No throttled window → return (None, None)."""
    headers = {
        "anthropic-ratelimit-unified-5h-status": "allowed",
        "anthropic-ratelimit-unified-7d-status": "allowed",
    }
    assert _rl_detect_throttled_window(headers, time.time()) == (None, None)


def test_detect_daily_window():
    """Some enterprise tiers expose a 1d window — it must be detected too."""
    now = time.time()
    future_1d = now + 12 * 3600
    headers = {
        "anthropic-ratelimit-unified-1d-status": "throttled",
        "anthropic-ratelimit-unified-1d-reset": str(int(future_1d)),
    }
    win, secs = _rl_detect_throttled_window(headers, now)
    assert win == "1d"
    assert secs == pytest.approx(12 * 3600, abs=2.0)


def test_detect_throttled_status_case_insensitive():
    """Status comparison must be case-insensitive (Anthropic varies)."""
    now = time.time()
    headers = {
        "anthropic-ratelimit-unified-5h-status": "Throttled",
        "anthropic-ratelimit-unified-5h-reset": str(int(now + 600)),
    }
    win, _ = _rl_detect_throttled_window(headers, now)
    assert win == "5h"


def test_detect_throttled_without_reset_is_ignored():
    """If status=throttled but no reset header, that window doesn't drive the wait."""
    headers = {
        "anthropic-ratelimit-unified-7d-status": "throttled",
        # no -reset
    }
    assert _rl_detect_throttled_window(headers, time.time()) == (None, None)


# ---------------------------------------------------------------------------
# _rl_seconds_until_reset — public entry point
# ---------------------------------------------------------------------------


def test_seconds_until_reset_weekly_picks_7d_window():
    """End-to-end: 7d-throttled 429 returns window='7d' and ~7d seconds."""
    now = time.time()
    future_7d = now + 6 * 86_400  # 6 days
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-5h-status": "allowed",
        "anthropic-ratelimit-unified-7d-status": "throttled",
        "anthropic-ratelimit-unified-7d-reset": str(int(future_7d)),
    })
    win, secs = _rl_seconds_until_reset(err)
    assert win == "7d"
    assert secs == pytest.approx(6 * 86_400, abs=2.0)


def test_seconds_until_reset_5h_when_only_5h_throttled():
    now = time.time()
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-5h-status": "throttled",
        "anthropic-ratelimit-unified-5h-reset": str(int(now + 600)),
        "anthropic-ratelimit-unified-7d-status": "allowed",
    })
    win, secs = _rl_seconds_until_reset(err)
    assert win == "5h"
    assert secs == pytest.approx(600.0, abs=1.0)


def test_seconds_until_reset_falls_back_to_unified_when_no_window():
    """When no window-specific status header is present, fall back to the
    unified ``anthropic-ratelimit-unified-reset`` (window=None)."""
    now = time.time()
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-reset": str(int(now + 1200)),
    })
    win, secs = _rl_seconds_until_reset(err)
    assert win is None
    assert secs == pytest.approx(1200.0, abs=1.0)


def test_seconds_until_reset_falls_back_to_retry_after():
    """``retry-after`` (windowless) must still work when no ratelimit headers."""
    err = _make_429_with_headers({"retry-after": "30"})
    win, secs = _rl_seconds_until_reset(err)
    assert win is None
    assert secs == 30.0


def test_seconds_until_reset_returns_none_for_empty():
    """No parseable info → (None, None)."""
    err = _make_429_with_headers({})
    assert _rl_seconds_until_reset(err) == (None, None)


# ---------------------------------------------------------------------------
# _rl_window_cap — per-window safety cap
# ---------------------------------------------------------------------------


def test_window_cap_defaults():
    assert _rl_window_cap("5h") == 6 * 3600
    assert _rl_window_cap("1d") == 26 * 3600
    assert _rl_window_cap("7d") == int(7.5 * 86_400)


def test_window_cap_falls_back_to_legacy():
    """Unknown window → fall back to ``HERMES_RL_AUTOWAIT_MAX_S`` (default 6h)."""
    assert _rl_window_cap(None) == 6 * 3600
    assert _rl_window_cap("unknown") == 6 * 3600


def test_window_cap_env_override(monkeypatch):
    """Per-window env vars override the defaults."""
    monkeypatch.setenv("HERMES_RL_AUTOWAIT_MAX_7D_S", "1000")
    assert _rl_window_cap("7d") == 1000
    # Other windows keep their defaults
    assert _rl_window_cap("5h") == 6 * 3600


def test_window_defaults_match_documented_table():
    """Sanity check: the documented table matches the in-code constants."""
    assert _RL_WINDOW_DEFAULTS_S == {
        "5h": 21_600,
        "1d": 93_600,
        "7d": 648_000,
    }


# ---------------------------------------------------------------------------
# _rl_sleep_until_reset — the actual wait-and-retry path.
# These use fast_sleep so we don't block on multi-day timeouts.
# ---------------------------------------------------------------------------


def test_weekly_wait_not_blocked_by_legacy_6h_cap(monkeypatch, fast_sleep):
    """A 7d hit must NOT be rejected by the legacy 6h ``HERMES_RL_AUTOWAIT_MAX_S``.

    This is the core bug the user reported: the old code would say
    "⏱️ Лимит Claude исчерпан, но сброс через 5d 12h — это больше
    safety-cap (6h). Останавливаюсь."  v1.5.9 must let the weekly reset
    proceed up to the per-window 7d cap.

    We simulate "old code would have bailed" by setting the legacy cap
    to 1s, then the new code must STILL proceed (because it uses the
    7d-specific cap of 10s, not the legacy 1s).
    """
    agent = _make_agent()
    now = time.time()
    # Reset 2s out — above the (mocked) 1s legacy cap, below the 10s 7d cap.
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-7d-status": "throttled",
        "anthropic-ratelimit-unified-7d-reset": str(int(now + 2)),
    })

    monkeypatch.setenv("HERMES_RL_AUTOWAIT_BUFFER_S", "0")
    monkeypatch.setenv("HERMES_RL_AUTOWAIT_MAX_S", "1")  # legacy cap shrunk
    monkeypatch.setenv("HERMES_RL_AUTOWAIT_MAX_7D_S", "10")

    result = _rl_sleep_until_reset(agent, err, attempt=1)
    assert result is True, (
        "v1.5.9 must NOT bail on 7d hits — per-window cap should govern"
    )


def test_weekly_wait_bails_when_above_per_window_cap(monkeypatch):
    """If the parsed 7d reset is genuinely too large (e.g. > 7.5d default cap),
    the per-window cap must still catch it so we don't hang forever.
    """
    agent = _make_agent()
    now = time.time()
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-7d-status": "throttled",
        "anthropic-ratelimit-unified-7d-reset": str(int(now + 10 * 86_400)),
    })

    monkeypatch.setenv("HERMES_RL_AUTOWAIT_BUFFER_S", "0")
    # Default 7d cap (7.5d = 648000s) applies.
    assert _rl_window_cap("7d") == int(7.5 * 86_400)
    result = _rl_sleep_until_reset(agent, err, attempt=1)
    assert result is False  # 10d > 7.5d cap → bail


def test_5h_wait_capped_by_5h_cap_not_by_7d_cap(monkeypatch):
    """A 5h-window hit must use the 5h cap (6h default), not the 7d cap.

    Regression guard: a naive refactor could accidentally apply the
    weekly cap to all waits, which would re-introduce the original bug
    for the more common 5h case.
    """
    agent = _make_agent()
    now = time.time()
    # 12h reset — above 5h cap (6h), below 7d cap.
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-5h-status": "throttled",
        "anthropic-ratelimit-unified-5h-reset": str(int(now + 12 * 3600)),
    })

    monkeypatch.setenv("HERMES_RL_AUTOWAIT_BUFFER_S", "0")
    result = _rl_sleep_until_reset(agent, err, attempt=1)
    assert result is False  # 5h cap (6h) < 12h → bail


def test_weekly_wait_passes_when_below_7d_cap(monkeypatch, fast_sleep):
    """A normal 7d reset under the cap must wait and return True."""
    agent = _make_agent()
    now = time.time()
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-7d-status": "throttled",
        "anthropic-ratelimit-unified-7d-reset": str(int(now + 2)),
    })

    monkeypatch.setenv("HERMES_RL_AUTOWAIT_BUFFER_S", "0")
    monkeypatch.setenv("HERMES_RL_AUTOWAIT_MAX_7D_S", "10")
    result = _rl_sleep_until_reset(agent, err, attempt=1)
    assert result is True


def test_touch_desc_mentions_window_name(fast_sleep):
    """The liveness-tick description should name the window so it's clear
    in the gateway what we're waiting on."""
    agent = _make_agent()
    now = time.time()
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-7d-status": "throttled",
        "anthropic-ratelimit-unified-7d-reset": str(int(now + 2)),
    })
    _rl_sleep_until_reset(agent, err, attempt=1)
    descs = [t[1] for t in agent._touches]  # type: ignore[attr-defined]
    assert any("7d" in d for d in descs), (
        f"expected window name in touch desc, got: {descs!r}"
    )


def test_legacy_no_window_uses_unified_label(fast_sleep):
    """When the 429 has no window-specific status, the touch desc should
    fall back to the legacy 'unified' wording (not crash, not omit)."""
    agent = _make_agent()
    now = time.time()
    err = _make_429_with_headers({
        "anthropic-ratelimit-unified-reset": str(int(now + 2)),
    })
    _rl_sleep_until_reset(agent, err, attempt=1)
    descs = [t[1] for t in agent._touches]  # type: ignore[attr-defined]
    assert any("unified" in d for d in descs), (
        f"expected 'unified' fallback in touch desc, got: {descs!r}"
    )
