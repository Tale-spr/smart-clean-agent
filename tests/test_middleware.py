import unittest
from types import SimpleNamespace

from smart_clean_agent.agent.tools.middleware import build_runtime_context_prompt, monitor_tool


class MiddlewareTestCase(unittest.TestCase):
    def test_build_runtime_context_prompt_returns_empty_without_context(self):
        result = build_runtime_context_prompt({})

        self.assertEqual(result, "")

    def test_build_runtime_context_prompt_includes_memory_sections(self):
        result = build_runtime_context_prompt(
            {
                "session_summary": "最近用户关注天气和保养。",
                "recent_history": "用户: 北京今天天气怎么样？\n助手: 北京今天晴。",
                "user_memory_summary": "用户画像: 城市: 北京；户型: 65㎡公寓；地面: 木地板",
                "report_memory_summary": "覆盖月份: 2025-01、2025-02、2025-03",
            },
            is_report=True,
        )

        self.assertIn("最近会话上下文", result)
        self.assertIn("最近用户关注天气和保养。", result)
        self.assertIn("用户: 北京今天天气怎么样？", result)
        self.assertIn("用户长期记忆", result)
        self.assertIn("报告趋势记忆", result)

    def test_monitor_tool_records_rag_stage_event(self):
        context = {"status_events": []}
        request = SimpleNamespace(
            tool_call={"name": "rag_summarize", "args": {"query": "保养"}},
            runtime=SimpleNamespace(context=context),
        )

        result = monitor_tool.wrap_tool_call(request, lambda _: "ok")

        self.assertEqual(result, "ok")
        self.assertEqual(context["status_events"][0]["event_type"], "stage.rag")
        self.assertEqual(context["status_events"][1]["event_type"], "tool.success")

    def test_monitor_tool_records_error_event_on_failure(self):
        context = {"status_events": []}
        request = SimpleNamespace(
            tool_call={"name": "get_weather", "args": {"city": "北京"}},
            runtime=SimpleNamespace(context=context),
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            monitor_tool.wrap_tool_call(request, lambda _: (_ for _ in ()).throw(RuntimeError("boom")))

        self.assertEqual(context["status_events"][0]["event_type"], "stage.tool")
        self.assertEqual(context["status_events"][1]["event_type"], "error.tool")


if __name__ == "__main__":
    unittest.main()

