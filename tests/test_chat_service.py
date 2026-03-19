import unittest
from unittest.mock import Mock, patch

from smart_clean_agent.services.chat_service import ChatServiceError, run_chat, run_report


class FakeAgent:
    def __init__(self, response: str):
        self.response = response
        self.runtime_context = None
        self.query = None

    def execute(self, query: str, runtime_context: dict) -> str:
        self.query = query
        self.runtime_context = runtime_context
        return self.response


class ChatServiceTestCase(unittest.TestCase):
    @patch("smart_clean_agent.services.chat_service.build_report_memory_summary", return_value="趋势摘要")
    @patch("smart_clean_agent.services.chat_service.refresh_report_memory", return_value={"trend_summary": "趋势摘要"})
    @patch("smart_clean_agent.services.chat_service.update_user_memory")
    @patch("smart_clean_agent.services.chat_service.load_report_memory", return_value={"monthly_records": [{"month": "2025-01"}], "trend_summary": "趋势摘要"})
    @patch("smart_clean_agent.services.chat_service.build_user_memory_summary", return_value="用户记忆")
    @patch("smart_clean_agent.services.chat_service.load_user_memory", return_value={"profile_snapshot": {"city": "北京"}})
    @patch("smart_clean_agent.services.chat_service.save_session")
    @patch("smart_clean_agent.services.chat_service.create_session")
    @patch("smart_clean_agent.services.chat_service.get_user_profile")
    def test_run_chat_creates_session_and_saves_messages(
        self,
        mock_get_user_profile,
        mock_create_session,
        mock_save_session,
        mock_load_user_memory,
        mock_build_user_memory_summary,
        mock_load_report_memory,
        mock_update_user_memory,
        mock_refresh_report_memory,
        mock_build_report_memory_summary,
    ):
        mock_get_user_profile.return_value = {"user_id": "1001", "city": "北京"}
        mock_create_session.return_value = {
            "session_id": "session_001",
            "title": "会话 1",
            "created_at": "2025-01-01T10:00:00",
            "messages": [],
        }
        mock_save_session.side_effect = lambda user_id, session_data: {
            "session_id": session_data["session_id"],
            "title": session_data["title"],
            "created_at": session_data["created_at"],
            "updated_at": "2025-01-01T10:05:00",
            "messages": session_data["messages"],
            "session_summary": "摘要",
            "recent_history": "历史",
        }
        agent = FakeAgent("最终回答")

        result = run_chat(user_id="1001", message="北京今天天气怎么样？", agent=agent)

        self.assertEqual(result["answer"], "最终回答")
        self.assertEqual(result["session_id"], "session_001")
        self.assertEqual(result["session_summary"], "摘要")
        self.assertEqual(result["report_memory_summary"], "趋势摘要")
        self.assertTrue(result["status_events"])
        self.assertEqual(agent.query, "北京今天天气怎么样？")
        self.assertEqual(agent.runtime_context["user_id"], "1001")
        self.assertEqual(agent.runtime_context["city"], "北京")
        saved_messages = mock_save_session.call_args.args[1]["messages"]
        self.assertEqual(saved_messages[0]["role"], "user")
        self.assertEqual(saved_messages[1]["role"], "assistant")
        self.assertNotIn("status_events", result["session_data"])
        mock_update_user_memory.assert_called_once()
        mock_refresh_report_memory.assert_called()

    @patch("smart_clean_agent.services.chat_service.load_session", return_value=None)
    @patch("smart_clean_agent.services.chat_service.get_user_profile", return_value={"user_id": "1001", "city": "北京"})
    def test_run_chat_raises_when_session_not_found(self, mock_get_user_profile, mock_load_session):
        with self.assertRaises(ChatServiceError) as context:
            run_chat(user_id="1001", message="你好", session_id="missing", agent=FakeAgent("ok"))

        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(context.exception.code, "session_not_found")

    @patch("smart_clean_agent.services.chat_service.build_report_memory_summary", return_value="趋势摘要")
    @patch("smart_clean_agent.services.chat_service.refresh_report_memory", return_value={"trend_summary": "趋势摘要"})
    @patch("smart_clean_agent.services.chat_service.update_user_memory")
    @patch("smart_clean_agent.services.chat_service.load_report_memory", return_value={"monthly_records": [{"month": "2025-01"}], "trend_summary": "趋势摘要"})
    @patch("smart_clean_agent.services.chat_service.build_user_memory_summary", return_value="用户记忆")
    @patch("smart_clean_agent.services.chat_service.load_user_memory", return_value={"profile_snapshot": {"city": "北京"}})
    @patch("smart_clean_agent.services.chat_service.save_session")
    @patch("smart_clean_agent.services.chat_service.create_session")
    @patch("smart_clean_agent.services.chat_service.get_user_profile")
    def test_run_report_returns_report_key(
        self,
        mock_get_user_profile,
        mock_create_session,
        mock_save_session,
        mock_load_user_memory,
        mock_build_user_memory_summary,
        mock_load_report_memory,
        mock_update_user_memory,
        mock_refresh_report_memory,
        mock_build_report_memory_summary,
    ):
        mock_get_user_profile.return_value = {"user_id": "1001", "city": "北京"}
        mock_create_session.return_value = {
            "session_id": "session_002",
            "title": "会话 2",
            "created_at": "2025-01-01T10:00:00",
            "messages": [],
        }
        mock_save_session.side_effect = lambda user_id, session_data: {
            "session_id": session_data["session_id"],
            "title": session_data["title"],
            "created_at": session_data["created_at"],
            "updated_at": "2025-01-01T10:05:00",
            "messages": session_data["messages"],
            "session_summary": "摘要",
            "recent_history": "历史",
        }

        result = run_report(user_id="1001", query="生成我的本月使用报告", agent=FakeAgent("报告内容"))

        self.assertEqual(result["report"], "报告内容")
        self.assertEqual(result["report_memory_summary"], "趋势摘要")


if __name__ == "__main__":
    unittest.main()

