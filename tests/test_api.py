import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from smart_clean_agent.api.main import create_app
from smart_clean_agent.services.chat_service import ChatServiceError


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())

    @patch("smart_clean_agent.api.main.get_dependency_issues", return_value=[])
    def test_health_returns_healthy(self, mock_get_dependency_issues):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    @patch("smart_clean_agent.api.main.get_dependency_issues", return_value=["未配置 DASHSCOPE_API_KEY"])
    def test_health_returns_unhealthy(self, mock_get_dependency_issues):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "unhealthy")
        self.assertIn("missing_dependencies", response.json())

    @patch("smart_clean_agent.api.main.run_chat")
    @patch("smart_clean_agent.api.main.get_dependency_issues", return_value=[])
    def test_chat_returns_answer(self, mock_get_dependency_issues, mock_run_chat):
        mock_run_chat.return_value = {
            "user_id": "1001",
            "session_id": "session_001",
            "answer": "你好，我可以帮你查询天气。",
            "status_events": [
                {
                    "event_type": "stage.model",
                    "title": "正在分析问题",
                    "detail": "正在判断是否需要调用工具",
                    "created_at": "2025-01-01T10:00:00",
                    "level": "info",
                }
            ],
            "session_summary": "摘要",
        }

        response = self.client.post("/chat", json={"user_id": "1001", "message": "你好"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "你好，我可以帮你查询天气。")

    @patch("smart_clean_agent.api.main.run_report")
    @patch("smart_clean_agent.api.main.get_dependency_issues", return_value=[])
    def test_report_returns_report(self, mock_get_dependency_issues, mock_run_report):
        mock_run_report.return_value = {
            "user_id": "1001",
            "session_id": "session_002",
            "report": "本月使用报告",
            "status_events": [],
            "report_memory_summary": "覆盖月份: 2025-01、2025-02、2025-03",
        }

        response = self.client.post("/report", json={"user_id": "1001", "query": "生成我的使用报告"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["report"], "本月使用报告")

    @patch("smart_clean_agent.api.main.get_dependency_issues", return_value=["本地向量库不存在"])
    def test_chat_returns_dependency_error_when_not_ready(self, mock_get_dependency_issues):
        response = self.client.post("/chat", json={"user_id": "1001", "message": "你好"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "dependency_not_ready")

    @patch("smart_clean_agent.api.main.get_dependency_issues", return_value=[])
    @patch("smart_clean_agent.api.main.run_chat", side_effect=ChatServiceError("用户资料不存在", code="user_not_found", status_code=404))
    def test_chat_returns_service_error(self, mock_run_chat, mock_get_dependency_issues):
        response = self.client.post("/chat", json={"user_id": "1001", "message": "你好"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "user_not_found")

    def test_chat_returns_422_when_missing_required_fields(self):
        response = self.client.post("/chat", json={"user_id": "1001"})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()

