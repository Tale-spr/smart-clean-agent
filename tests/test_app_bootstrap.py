import unittest
from unittest.mock import patch, sentinel

from smart_clean_agent.web.bootstrap import build_agent, build_app_context, build_runtime_context, initialize_ui_state, save_current_session


class AppBootstrapTestCase(unittest.TestCase):
    def test_initialize_ui_state_sets_defaults(self):
        session_state = {}

        initialize_ui_state(session_state, ["1001", "1002"])

        self.assertEqual(session_state["message"], [])
        self.assertEqual(session_state["current_status_events"], [])
        self.assertEqual(session_state["latest_status_events"], [])
        self.assertEqual(session_state["selected_user_id"], "1001")

    @patch("smart_clean_agent.web.bootstrap.build_report_memory_summary", return_value="报告记忆")
    @patch("smart_clean_agent.web.bootstrap.load_report_memory", return_value={"monthly_records": [{"month": "2025-03"}], "trend_summary": "报告记忆"})
    @patch("smart_clean_agent.web.bootstrap.build_user_memory_summary", return_value="用户记忆")
    @patch("smart_clean_agent.web.bootstrap.load_user_memory", return_value={"profile_snapshot": {"city": "北京"}})
    def test_build_runtime_context_uses_current_session_state(
        self,
        mock_load_user_memory,
        mock_build_user_memory_summary,
        mock_load_report_memory,
        mock_build_report_memory_summary,
    ):
        session_state = {
            "selected_user_id": "1001",
            "selected_city": "北京",
            "current_session_id": "session_001",
            "message": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，我是客服助手"},
            ],
        }
        status_events = []

        runtime_context = build_runtime_context(
            session_state,
            status_events=status_events,
            status_event_callback=sentinel.status_callback,
        )

        self.assertFalse(runtime_context["report"])
        self.assertEqual(runtime_context["user_id"], "1001")
        self.assertEqual(runtime_context["city"], "北京")
        self.assertEqual(runtime_context["session_id"], "session_001")
        self.assertTrue(runtime_context["session_summary"])
        self.assertTrue(runtime_context["recent_history"])
        self.assertEqual(runtime_context["user_memory_summary"], "用户记忆")
        self.assertEqual(runtime_context["report_memory_summary"], "报告记忆")
        self.assertIs(runtime_context["status_events"], status_events)
        self.assertIs(runtime_context["status_event_callback"], sentinel.status_callback)

    @patch("smart_clean_agent.web.bootstrap.ReactAgent")
    @patch("smart_clean_agent.web.bootstrap.create_agent_tools")
    @patch("smart_clean_agent.web.bootstrap.RagSummarizeService")
    @patch("smart_clean_agent.web.bootstrap.VectorStoreService")
    @patch("smart_clean_agent.web.bootstrap.create_embedding_model")
    @patch("smart_clean_agent.web.bootstrap.create_chat_model")
    def test_build_agent_wires_dependencies_in_bootstrap(
        self,
        mock_create_chat_model,
        mock_create_embedding_model,
        mock_vector_store_service,
        mock_rag_service,
        mock_create_agent_tools,
        mock_react_agent,
    ):
        mock_create_chat_model.return_value = sentinel.chat_model
        mock_create_chat_model.side_effect = [sentinel.chat_model, sentinel.normalization_model]
        mock_create_embedding_model.return_value = sentinel.embedding_model
        mock_vector_store_service.return_value = sentinel.vector_store_service
        mock_rag_service.return_value = sentinel.rag_service
        mock_create_agent_tools.return_value = sentinel.tools
        mock_react_agent.return_value = sentinel.agent

        agent = build_agent()

        self.assertIs(agent, sentinel.agent)
        mock_vector_store_service.assert_called_once_with(embedding_function=sentinel.embedding_model)
        mock_rag_service.assert_called_once_with(
            model=sentinel.chat_model,
            vector_store_service=sentinel.vector_store_service,
        )
        mock_create_agent_tools.assert_called_once_with(sentinel.rag_service)
        self.assertEqual(mock_create_chat_model.call_count, 2)
        mock_create_chat_model.assert_any_call()
        mock_create_chat_model.assert_any_call(role="normalization")
        mock_react_agent.assert_called_once_with(
            model=sentinel.chat_model,
            tools=sentinel.tools,
            normalization_model=sentinel.normalization_model,
        )

    @patch("smart_clean_agent.web.bootstrap.initialize_memory_state")
    @patch("smart_clean_agent.web.bootstrap.ensure_active_session")
    @patch("smart_clean_agent.web.bootstrap.get_user_profile")
    @patch("smart_clean_agent.web.bootstrap.list_user_profiles")
    def test_build_app_context_initializes_selected_profile_and_agent(
        self,
        mock_list_user_profiles,
        mock_get_user_profile,
        mock_ensure_active_session,
        mock_initialize_memory_state,
    ):
        profile = {"user_id": "1001", "city": "北京", "name": "用户A"}
        mock_list_user_profiles.return_value = [profile]
        mock_get_user_profile.return_value = profile
        session_state = {}

        context = build_app_context(session_state)

        self.assertEqual(context.user_profiles, [profile])
        self.assertEqual(context.selected_profile, profile)
        self.assertEqual(session_state["selected_user_id"], "1001")
        self.assertEqual(session_state["selected_city"], "北京")
        mock_ensure_active_session.assert_called_once_with(session_state, "1001")
        mock_initialize_memory_state.assert_called_once_with(session_state, profile)

    @patch("smart_clean_agent.web.bootstrap.refresh_report_memory")
    @patch("smart_clean_agent.web.bootstrap.update_user_memory")
    @patch("smart_clean_agent.web.bootstrap.get_user_profile")
    @patch("smart_clean_agent.web.bootstrap.save_session")
    def test_save_current_session_updates_memories(
        self,
        mock_save_session,
        mock_get_user_profile,
        mock_update_user_memory,
        mock_refresh_report_memory,
    ):
        session_state = {
            "current_session_id": "session_001",
            "current_user_id": "1001",
            "selected_city": "北京",
            "current_session_title": "会话 1",
            "current_session_created_at": "2025-01-01T10:00:00",
            "message": [{"role": "user", "content": "你好"}],
        }
        mock_save_session.return_value = {
            "title": "会话 1",
            "created_at": "2025-01-01T10:00:00",
            "session_summary": "摘要",
            "recent_history": "历史",
        }
        mock_get_user_profile.return_value = {"user_id": "1001", "city": "北京"}

        save_current_session(session_state)

        mock_update_user_memory.assert_called_once()
        mock_refresh_report_memory.assert_called_once_with("1001")
        self.assertEqual(session_state["current_session_summary"], "摘要")
        self.assertEqual(session_state["current_recent_history"], "历史")


if __name__ == "__main__":
    unittest.main()

