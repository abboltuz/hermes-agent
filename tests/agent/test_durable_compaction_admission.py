"""Exercise the public compression boundary with real profile-scoped storage."""

from types import SimpleNamespace

import pytest

from agent import conversation_compression as compression
from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


def agent(db, *, generation=0, model="route-a"):
    return SimpleNamespace(
        session_id="chat", _session_db=db,
        _conversation_root_id=lambda: "chat",
        _compression_generation=generation,
        context_compressor=SimpleNamespace(model=model, context_length=8192),
    )


def real_agent(db, **policy):
    instance = agent(db)
    instance.context_compressor = ContextCompressor(
        model="test-model", provider="custom", config_context_length=8192,
        quiet_mode=True, **policy,
    )
    return instance


def test_restart_and_new_generation_cannot_repeat_same_failure(tmp_path, monkeypatch):
    calls = []
    messages = [{"role": "user", "content": "continue"}]

    def no_progress(_agent, messages, system_message, **_kwargs):
        calls.append(1)
        return messages, system_message

    monkeypatch.setattr(compression, "_compress_context_impl", no_progress)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        compression.compress_context(agent(db), messages, "policy", trigger="preflight")
    with SessionDB(path) as db:
        resumed = agent(db, generation=999)
        result = compression.compress_context(resumed, messages, "policy", trigger="overflow")
        assert result[0] is messages
        assert len(calls) == 1
        compression.compress_context(resumed, messages, "policy", trigger="manual", force=True)
        assert len(calls) == 2
        compression.compress_context(agent(db, model="route-b"), messages, "policy", trigger="preflight")
        assert len(calls) == 3


def test_receipt_records_cancellation_and_preserves_exception(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "continue"}]
    calls = []

    def interrupted(*_args, **_kwargs):
        calls.append(1)
        raise KeyboardInterrupt()

    monkeypatch.setattr(compression, "_compress_context_impl", interrupted)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        with pytest.raises(KeyboardInterrupt):
            compression.compress_context(agent(db), messages, "policy", trigger="preflight")
    with SessionDB(path) as db:
        compression.compress_context(agent(db), messages, "policy", trigger="preflight")
    assert len(calls) == 1


def test_admission_storage_failure_does_not_start_summarizer_or_mutate_history(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "continue"}]

    def unavailable(*_args, **_kwargs):
        raise OSError("storage unavailable")

    def unexpected(*_args, **_kwargs):
        pytest.fail("summarizer must not run without durable admission")

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", "test")
        db.append_messages_batch("chat", messages)
        before = db.get_messages("chat")
        monkeypatch.setattr(db, "claim_context_compaction", unavailable)
        monkeypatch.setattr(compression, "_compress_context_impl", unexpected)
        result = compression.compress_context(agent(db), messages, "policy", trigger="preflight")
        assert result == (messages, "policy")
        assert result[0] is messages
        assert db.get_messages("chat") == before


def test_failed_receipt_does_not_turn_success_into_error(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "continue"}]
    result = ([{"role": "user", "content": "compact"}], "policy")
    monkeypatch.setattr(compression, "_compress_context_impl", lambda *_a, **_k: result)

    def unavailable(*_args, **_kwargs):
        raise OSError("storage unavailable")

    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", "test")
        monkeypatch.setattr(db, "finish_context_compaction", unavailable)
        assert compression.compress_context(agent(db), messages, "policy", trigger="preflight") is result
        # No receipt means no automatic repeat, even from a new process owner.
        assert db.claim_context_compaction("chat", "another", "strategy", owner="new") == "deferred_lock"


def test_unreadable_policy_does_not_start_work_or_change_context(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("config unavailable")

    monkeypatch.setattr(compression, "_compression_strategy_fingerprint", unavailable)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", "test")
        messages = [{"role": "user", "content": "continue"}]
        result = compression.compress_context(agent(db), messages, "policy", trigger="preflight")
        assert result[0] is messages
        assert result[1] == "policy"


def test_real_structural_noop_arming_backoff_is_terminal_after_restart(tmp_path, monkeypatch):
    calls = []
    messages = [{"role": "user", "content": "short protected turn"}]

    def run_compressor(instance, messages, system, **_kwargs):
        calls.append(1)
        result = instance.context_compressor.compress(messages, current_tokens=9000)
        assert instance.context_compressor._automatic_compression_blocked()
        return result, system

    monkeypatch.setattr(compression, "_compress_context_impl", run_compressor)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
    with SessionDB(path) as db:
        # A newly constructed compressor has no process-local backoff.
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
    assert len(calls) == 1


@pytest.mark.parametrize("changed_policy", [
    {"base_url": "https://other.invalid/v1"},
    {"max_tokens": 1024},
    {"min_tail_user_messages": 3},
    {"threshold_tokens_cap": 7000},
])
def test_inherited_route_and_budget_changes_rearm_failed_job(tmp_path, monkeypatch, changed_policy):
    calls = []

    def no_progress(_instance, messages, system, **_kwargs):
        calls.append(1)
        return messages, system

    monkeypatch.setattr(compression, "_compress_context_impl", no_progress)
    messages = [{"role": "user", "content": "same source"}]
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
    with SessionDB(path) as db:
        compression.compress_context(real_agent(db, **changed_policy), messages, "policy", trigger="preflight")
    assert len(calls) == 2


def test_real_summary_timeout_is_not_rearmed_by_a_fresh_compressor(tmp_path, monkeypatch):
    calls = []
    messages = [{"role": "user" if index % 2 == 0 else "assistant",
                 "content": f"turn {index}: " + "context " * 30,
                 "origin_kind": "human_user" if index % 2 == 0 else "agent"}
                for index in range(15)]

    def timeout(**_kwargs):
        calls.append(1)
        raise TimeoutError("summary request timed out")

    def run_compressor(instance, messages, system, **_kwargs):
        result = instance.context_compressor.compress(messages, current_tokens=6000)
        assert instance.context_compressor._last_compress_aborted
        assert instance.context_compressor._automatic_compression_blocked()
        return result, system

    monkeypatch.setattr("agent.context_compressor.call_llm", timeout)
    monkeypatch.setattr(compression, "_compress_context_impl", run_compressor)
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        compression.compress_context(real_agent(db, protect_first_n=1, protect_last_n=1,
                                                abort_on_summary_failure=True),
                                     messages, "policy", trigger="preflight")
    assert calls
    attempts = len(calls)
    with SessionDB(path) as db:
        compression.compress_context(real_agent(db, protect_first_n=1, protect_last_n=1,
                                                abort_on_summary_failure=True),
                                     messages, "policy", trigger="preflight")
    assert len(calls) == attempts


def test_preexisting_cooldown_deferral_remains_retryable(tmp_path, monkeypatch):
    calls = []

    def no_progress(instance, messages, system, **_kwargs):
        if not instance.context_compressor._automatic_compression_blocked():
            calls.append(1)
        return messages, system

    monkeypatch.setattr(compression, "_compress_context_impl", no_progress)
    messages = [{"role": "user", "content": "same source"}]
    path = tmp_path / "state.db"
    with SessionDB(path) as db:
        db.create_session("chat", "test")
        waiting = real_agent(db)
        waiting.context_compressor._record_compression_failure_cooldown(60, "prior failure")
        compression.compress_context(waiting, messages, "policy", trigger="preflight")
        assert not calls
    with SessionDB(path) as db:
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
    assert len(calls) == 1


def test_auxiliary_fallback_and_api_mode_rearm_but_credential_rotation_does_not(tmp_path, monkeypatch):
    route = {"provider": "custom", "model": "same-model", "api_key": "fixture-only-a"}
    monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda _task: route)
    calls = []

    def no_progress(_instance, messages, system, **_kwargs):
        calls.append(1)
        return messages, system

    monkeypatch.setattr(compression, "_compress_context_impl", no_progress)
    messages = [{"role": "user", "content": "same source"}]
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("chat", "test")
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
        route["api_key"] = "fixture-only-b"
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
        assert len(calls) == 1
        route["api_mode"] = "codex_responses"
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
        assert len(calls) == 2
        route["fallback_chain"] = [{"provider": "custom", "model": "different-model"}]
        compression.compress_context(real_agent(db), messages, "policy", trigger="preflight")
        assert len(calls) == 3


def test_endpoint_credentials_are_excluded_from_strategy_identity(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        first = real_agent(db, base_url="https://user:fixture-a@example.invalid/v1?api_key=a&api-version=1")
        second = real_agent(db, base_url="https://user:fixture-b@example.invalid/v1?api_key=b&api-version=1")
        third = real_agent(db, base_url="https://example.invalid/v1?api-version=2")
        fingerprint = compression._compression_strategy_fingerprint
        assert fingerprint(first, None) == fingerprint(second, None)
        assert fingerprint(first, None) != fingerprint(third, None)
