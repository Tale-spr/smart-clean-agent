import shutil
import time
import unittest
import uuid
from pathlib import Path

from smart_clean_agent.services.session_service import create_session, delete_session, get_latest_session, list_sessions, load_session, save_session


class SessionServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"sessions_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = "1001"

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_create_session_creates_json_file(self):
        session_data = create_session(self.user_id, str(self.base_dir))

        self.assertEqual(session_data["user_id"], self.user_id)
        self.assertEqual(session_data["messages"], [])
        self.assertEqual(session_data["session_summary"], "")
        self.assertEqual(session_data["recent_history"], "")
        self.assertTrue((self.base_dir / self.user_id / f"{session_data['session_id']}.json").exists())

    def test_save_and_load_session_round_trip(self):
        session_data = create_session(self.user_id, str(self.base_dir))
        session_data["messages"] = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我来帮你。"},
        ]
        save_session(self.user_id, session_data, str(self.base_dir))

        loaded = load_session(self.user_id, session_data["session_id"], str(self.base_dir))

        self.assertEqual(loaded["messages"][0]["content"], "你好")
        self.assertIn("最近用户关注", loaded["session_summary"])
        self.assertIn("用户: 你好", loaded["recent_history"])

    def test_list_sessions_returns_latest_first(self):
        first = create_session(self.user_id, str(self.base_dir), title="会话1")
        time.sleep(1)
        second = create_session(self.user_id, str(self.base_dir), title="会话2")
        first["messages"] = [{"role": "user", "content": "更新"}]
        time.sleep(1)
        save_session(self.user_id, first, str(self.base_dir))

        sessions = list_sessions(self.user_id, str(self.base_dir))

        self.assertEqual(sessions[0]["session_id"], first["session_id"])
        self.assertEqual(sessions[1]["session_id"], second["session_id"])

    def test_delete_session_removes_file(self):
        session_data = create_session(self.user_id, str(self.base_dir))

        deleted = delete_session(self.user_id, session_data["session_id"], str(self.base_dir))

        self.assertTrue(deleted)
        self.assertIsNone(load_session(self.user_id, session_data["session_id"], str(self.base_dir)))

    def test_get_latest_session_returns_none_when_empty(self):
        self.assertIsNone(get_latest_session(self.user_id, str(self.base_dir)))


if __name__ == "__main__":
    unittest.main()

