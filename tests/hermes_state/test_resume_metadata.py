"""Opening a chat must not synchronize billing or materialize its prompt."""

from hermes_state import SessionDB


def test_resume_metadata_is_read_only_and_retains_runtime_identity(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session(
            "chat", "desktop", model="test/model", system_prompt="large policy " * 1000
        )
        expected = db.get_session("chat")

        def forbid_usage_write():
            raise AssertionError("usage write on open")

        with monkeypatch.context() as scoped:
            scoped.setattr(db, "flush_token_counts", forbid_usage_write)
            row = db.get_session_for_resume("chat")
            assert row == {
                key: value for key, value in expected.items()
                if key not in db._SESSION_COMPACT_EXCLUDED
            }
            assert db.get_session_for_resume("missing") is None
        assert db.get_session("chat") == expected
