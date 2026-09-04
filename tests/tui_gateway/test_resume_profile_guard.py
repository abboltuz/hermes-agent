"""Resume policy follows the owning store, not a thread's ambient profile."""

import threading

import pytest

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from hermes_state import SessionDB, SessionResumeTooLargeError
from tui_gateway import server


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("owner_limit", [0, 2])
def test_resume_applies_owner_profile_guard(
    tmp_path, monkeypatch, deferred, owner_limit
):
    launch = tmp_path / "launch"
    owner = tmp_path / "owner"
    for home, limit in ((launch, 2 if owner_limit == 0 else 0), (owner, owner_limit)):
        home.mkdir()
        (home / "config.yaml").write_text(
            f"sessions:\n  max_resume_messages: {limit}\n", encoding="utf-8"
        )
    with SessionDB(owner / "state.db") as db:
        db.create_session("chat", "desktop")
        db.append_message("chat", "user", "first")
        db.append_message("chat", "assistant", "answer")
        db.append_message("chat", "user", "second")

    observed = []
    progress = []
    worker_closed = threading.Event()
    real_resume_guard = SessionDB.assert_resume_safe
    real_model_guard = SessionDB.assert_export_safe
    real_close = SessionDB.close

    def resume_guard(db, target, *args, **kwargs):
        observed.append(get_hermes_home())
        return real_resume_guard(db, target, *args, **kwargs)

    def model_guard(db, target, *args, **kwargs):
        observed.append(get_hermes_home())
        return real_model_guard(db, target, *args, **kwargs)

    def close(db):
        real_close(db)
        worker_closed.set()

    monkeypatch.setattr(SessionDB, "assert_resume_safe", resume_guard)
    monkeypatch.setattr(SessionDB, "assert_export_safe", model_guard)
    monkeypatch.setattr(SessionDB, "close", close)
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_profile_home", lambda _name: owner)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_lazy_resume_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server,
        "_resume_hard_summary_callback",
        lambda _session: (lambda _batch, _previous: "bounded profile summary"),
    )
    monkeypatch.setattr(
        server, "_emit", lambda event, _sid, data: progress.append((event, data))
    )

    ambient = set_hermes_home_override(launch)
    try:
        response = server._methods["session.resume"](
            "open",
            {
                "session_id": "chat",
                "profile": "owner",
                "source": "desktop",
                "defer_history": deferred,
                "omit_messages": True,
            },
        )
        assert worker_closed.wait(timeout=3)
        assert get_hermes_home() == launch
        assert observed == [owner]
        if deferred:
            assert response["result"]["hydrating"] is True
            progress_events = [
                data
                for event, data in progress
                if event == "session.resume_progress"
            ]
            assert progress_events[0]["status"] == "loading"
            assert progress_events[-1]["status"] == (
                "failed" if owner_limit else "complete"
            )
        elif owner_limit:
            assert response["error"]["code"] == 4130
        else:
            assert "result" in response
    finally:
        reset_hermes_home_override(ambient)


@pytest.mark.parametrize(
    ("launch_required", "owner_required", "blocked"),
    [(False, True, True), (True, False, False)],
)
def test_hard_recovery_checkpoint_policy_uses_owner_profile(
    tmp_path,
    monkeypatch,
    launch_required,
    owner_required,
    blocked,
):
    launch = tmp_path / "launch"
    owner = tmp_path / "owner"
    for home, required in (
        (launch, launch_required),
        (owner, owner_required),
    ):
        home.mkdir()
        (home / "config.yaml").write_text(
            "compression:\n"
            f"  checkpoint_required: {'true' if required else 'false'}\n",
            encoding="utf-8",
        )

    db = SessionDB(owner / "state.db")
    db.create_session("chat", "desktop")
    for index in range(30):
        db.append_message(
            "chat",
            "user" if index % 2 == 0 else "assistant",
            f"message-{index}",
        )

    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_resume_hard_summary_callback",
        lambda _session: (lambda _batch, _previous: "bounded owner summary"),
    )
    sid = "owner-checkpoint-policy"
    session = {
        "history_lock": threading.Lock(),
        "profile_home": str(owner),
        "resume_preparation": {
            "attempt": 1,
            "phase": "history",
            "status": "preparing",
        },
    }
    server._sessions[sid] = session

    ambient = set_hermes_home_override(launch)
    try:
        if blocked:
            with pytest.raises(RuntimeError, match="BLOCKED_MISSING_PREREQUISITE"):
                server._recover_oversized_model_resume(
                    sid,
                    session,
                    "chat",
                    db,
                    SessionResumeTooLargeError(30, 10),
                    attempt=1,
                )
            assert len(db.get_messages("chat")) == 30
        else:
            server._recover_oversized_model_resume(
                sid,
                session,
                "chat",
                db,
                SessionResumeTooLargeError(30, 10),
                attempt=1,
            )
            assert len(db.get_model_resume_conversation("chat")) <= 5
            assert sum(
                row["compacted"] == 1
                for row in db.get_messages("chat", include_inactive=True)
            ) >= 30
        assert get_hermes_home() == launch
    finally:
        reset_hermes_home_override(ambient)
        db.close()


def test_hard_summary_callback_carries_credential_free_route_identity(monkeypatch):
    import agent.context_compressor as context_compressor
    import agent.conversation_compression as conversation_compression

    captured = {}

    class FakeCompressor:
        context_length = 96_000

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def _generate_summary(self, _messages):
            return "summary"

    monkeypatch.setattr(context_compressor, "ContextCompressor", FakeCompressor)
    monkeypatch.setattr(
        conversation_compression,
        "compression_strategy_fingerprint_for_engine",
        lambda compressor, focus: (
            "credential-free-route"
            if isinstance(compressor, FakeCompressor) and focus is None
            else "wrong"
        ),
    )

    callback = server._resume_hard_summary_callback(
        {
            "resume_runtime_overrides": {
                "provider_override": "custom:work",
                "model_override": {
                    "model": "model-a",
                    "provider": "custom:work",
                    "base_url": "https://model.invalid/v1",
                    "api_mode": "openai",
                },
            }
        }
    )

    assert callback._resume_context_window == 96_000
    assert callback._resume_strategy_identity == "credential-free-route"
    assert captured["model"] == "model-a"
    assert captured["provider"] == "custom:work"
    assert "api_key" not in captured


@pytest.mark.parametrize("owner_limit", [0, 2])
def test_launch_profile_guard_restores_foreign_caller_context(
    tmp_path, monkeypatch, owner_limit
):
    launch = tmp_path / "launch"
    foreign = tmp_path / "foreign"
    launch.mkdir()
    foreign.mkdir()
    (launch / "config.yaml").write_text(
        f"sessions:\n  max_resume_messages: {owner_limit}\n", encoding="utf-8"
    )
    (foreign / "config.yaml").write_text(
        f"sessions:\n  max_resume_messages: {0 if owner_limit else 2}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_hermes_home", launch)
    with SessionDB(launch / "state.db") as db:
        db.create_session("chat", "desktop")
        for role in ("user", "assistant", "user"):
            db.append_message("chat", role, "content")
        token = set_hermes_home_override(foreign)
        try:
            if owner_limit:
                with pytest.raises(SessionResumeTooLargeError):
                    server._assert_session_resume_safe(db, "chat")
            else:
                server._assert_session_resume_safe(db, "chat")
            assert get_hermes_home() == foreign
        finally:
            reset_hermes_home_override(token)


def test_model_only_resume_guard_counts_tip_not_display_lineage(tmp_path, monkeypatch):
    home = tmp_path / "owner"
    home.mkdir()
    (home / "config.yaml").write_text(
        "sessions:\n  max_resume_messages: 2\n", encoding="utf-8"
    )
    monkeypatch.setattr(server, "_hermes_home", home)

    with SessionDB(home / "state.db") as db:
        db.create_session("parent", "desktop")
        db.append_message("parent", "user", "first")
        db.append_message("parent", "assistant", "answer")
        db.append_message("parent", "user", "second")
        db.end_session("parent", "compression")
        db.create_session(
            "tip",
            "desktop",
            parent_session_id="parent",
        )
        db.append_message("tip", "user", "live tip")

        with pytest.raises(SessionResumeTooLargeError):
            server._assert_session_resume_safe(db, "tip", profile_home=home)

        server._assert_session_resume_safe(
            db,
            "tip",
            profile_home=home,
            model_only=True,
        )
