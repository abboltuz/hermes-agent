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
    real_guard = SessionDB.assert_resume_safe
    real_close = SessionDB.close

    def guard(db, target, *args, **kwargs):
        observed.append(get_hermes_home())
        return real_guard(db, target, *args, **kwargs)

    def close(db):
        real_close(db)
        worker_closed.set()

    monkeypatch.setattr(SessionDB, "assert_resume_safe", guard)
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
            statuses = [
                data["status"]
                for event, data in progress
                if event == "session.resume_progress"
            ]
            assert statuses == ["loading", "failed" if owner_limit else "complete"]
        elif owner_limit:
            assert response["error"]["code"] == 4130
        else:
            assert "result" in response
    finally:
        reset_hermes_home_override(ambient)


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
