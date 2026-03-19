import unittest
from unittest.mock import patch

from smart_clean_agent.agent.tools.agent_tools import (
    fetch_external_data,
    fetch_external_history,
    format_external_data,
    get_current_month,
    get_user_id_from_context,
    get_user_location_from_context,
)


class AgentToolsTestCase(unittest.TestCase):
    def test_format_external_data_returns_stable_text(self):
        record = {
            "特征": "65㎡公寓 | 单身 | 木地板",
            "效率": "覆盖率:85%",
            "耗材": "主刷寿命:剩余60天",
            "对比": "优于65%同面积用户",
        }

        result = format_external_data(record, "2025-01")

        self.assertIn("月份: 2025-01", result)
        self.assertIn("特征: 65㎡公寓 | 单身 | 木地板", result)
        self.assertIn("清洁效率: 覆盖率:85%", result)

    def test_fetch_external_data_tool_returns_string(self):
        result = fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})

        self.assertIsInstance(result, str)
        self.assertIn("月份: 2025-01", result)
        self.assertIn("特征:", result)

    @patch("smart_clean_agent.agent.tools.agent_tools.build_report_memory_summary", return_value="覆盖月份: 2025-01、2025-02、2025-03\n清洁效率趋势: 稳定\n耗材趋势: 稳定\n问题变化趋势: 稳定\n总结建议: 继续保持")
    @patch("smart_clean_agent.agent.tools.agent_tools.refresh_report_memory", return_value={"trend_summary": "ok"})
    def test_fetch_external_history_returns_trend_text(self, mock_refresh_report_memory, mock_build_report_memory_summary):
        result = fetch_external_history.invoke({"user_id": "1001", "months": 3})

        self.assertIsInstance(result, str)
        self.assertIn("覆盖月份:", result)
        self.assertIn("清洁效率趋势:", result)
        self.assertIn("总结建议:", result)
        mock_refresh_report_memory.assert_called_once_with("1001", months=3)
        mock_build_report_memory_summary.assert_called_once()

    def test_get_current_month_returns_yyyy_mm(self):
        result = get_current_month.invoke({})

        self.assertRegex(result, r"^\d{4}-\d{2}$")

    def test_get_user_id_from_context_returns_value(self):
        result = get_user_id_from_context({"user_id": "1001", "city": "北京", "report": False})

        self.assertEqual(result, "1001")

    def test_get_user_id_from_context_returns_default_message(self):
        result = get_user_id_from_context({})

        self.assertEqual(result, "当前会话未配置用户ID")

    def test_get_user_location_from_context_returns_value(self):
        result = get_user_location_from_context({"user_id": "1001", "city": "北京", "report": False})

        self.assertEqual(result, "北京")

    def test_get_user_location_from_context_returns_default_message(self):
        result = get_user_location_from_context({})

        self.assertEqual(result, "当前会话未配置用户所在城市")


if __name__ == "__main__":
    unittest.main()

