# pyright: reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false
"""Tests for the headerless-429 bail-out (PR #27).

A 429 carrying no ``anthropic-ratelimit-unified-*`` headers is ambiguous: it is
either a real quota hit whose header we failed to parse, or a third-party
detection rejection that retrying can never clear. Left unbounded, the latter
turns one poisoned credential into a permanent 429 heartbeat, so the auto-wait
bails after ``HERMES_RL_AUTOWAIT_HEADERLESS_ATTEMPTS`` attempts.

A 429 *with* reset headers is an unambiguous quota hit and must keep waiting.
"""

import types

import pytest

import anthropic_billing_bypass as bp


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers
        self.status_code = 429


class _Fake429(Exception):
    """Mimics anthropic.RateLimitError closely enough for the header parser."""

    def __init__(self, headers, body):
        super().__init__(f"Error code: 429 - {body}")
        self.status_code = 429
        self.response = _FakeResponse(headers)
        self.body = body


class _VirtualClock:
    """Stand-in for the ``time`` module so waits are instant but still ordered."""

    def __init__(self, start=1_800_000_000.0):
        self._now = start

    def time(self):
        return self._now

    def monotonic(self):
        return self._now

    def sleep(self, seconds):
        self._now += max(0.0, seconds)

    def localtime(self, secs=None):
        import time as _real_time

        return _real_time.localtime(secs if secs is not None else self._now)

    def strftime(self, fmt, t=None):
        import time as _real_time

        return _real_time.strftime(fmt, t if t is not None else self.localtime())


@pytest.fixture
def autowait(monkeypatch):
    """Auto-wait with a virtual clock, captured emissions and short knobs."""
    clock = _VirtualClock()
    emitted = []

    monkeypatch.setattr(bp, "time", clock)
    monkeypatch.setattr(bp, "_rl_emit", lambda agent, msg: emitted.append(msg))
    monkeypatch.setattr(bp, "_rl_touch", lambda agent, desc: None)
    monkeypatch.setattr(bp, "_rl_interrupted", lambda agent: False)
    monkeypatch.setenv("HERMES_RL_AUTOWAIT_DEFAULT_S", "1")
    monkeypatch.setenv("HERMES_RL_AUTOWAIT_HEADERLESS_ATTEMPTS", "2")

    agent = types.SimpleNamespace()

    def run(exc, attempt):
        emitted.clear()
        return bp._rl_sleep_until_reset(agent, exc, attempt)

    return types.SimpleNamespace(run=run, emitted=emitted, clock=clock)


def _headerless_429():
    return _Fake429(
        headers={},
        body={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Error"},
        },
    )


def _quota_429(clock):
    return _Fake429(
        headers={
            "anthropic-ratelimit-unified-5h-status": "throttled",
            "anthropic-ratelimit-unified-5h-reset": str(int(clock.time()) + 2),
        },
        body={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "quota"},
        },
    )


@pytest.mark.parametrize("attempt", [1, 2])
def test_headerless_429_waits_within_attempt_budget(autowait, attempt):
    assert autowait.run(_headerless_429(), attempt) is True


@pytest.mark.parametrize("attempt", [3, 4, 5])
def test_headerless_429_bails_past_attempt_budget(autowait, attempt):
    assert autowait.run(_headerless_429(), attempt) is False


def test_headerless_429_bail_explains_why(autowait):
    autowait.run(_headerless_429(), 3)

    assert autowait.emitted, "bail-out must tell the user why it stopped"
    assert "not a quota" in autowait.emitted[-1].lower()


@pytest.mark.parametrize("attempt", [1, 2, 3, 4, 5])
def test_quota_429_with_reset_headers_always_waits(autowait, attempt):
    assert autowait.run(_quota_429(autowait.clock), attempt) is True


def test_headerless_attempt_budget_is_configurable(autowait, monkeypatch):
    monkeypatch.setenv("HERMES_RL_AUTOWAIT_HEADERLESS_ATTEMPTS", "4")

    assert autowait.run(_headerless_429(), 4) is True
    assert autowait.run(_headerless_429(), 5) is False
