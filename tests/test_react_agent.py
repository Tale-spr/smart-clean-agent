import unittest

from langchain_core.messages import AIMessage

from smart_clean_agent.agent.react_agent import ReactAgent


class DummyModel:
    def __init__(self, answer: str = "这是最终回答。"):
        self.answer = answer
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.answer)


class DummyTool:
    def __init__(self, name: str, result: str):
        self.name = name
        self.result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.result


class ReactAgentTestCase(unittest.TestCase):
    def _build_agent(
        self,
        *,
        answer: str = "这是最终回答。",
        weather_result: str = "城市: 北京\n天气: 晴",
        rag_result: str = "建议定期更换滚刷。",
        user_id_result: str = "1001",
        current_month_result: str = "2026-03",
        external_data_result: str = "月份: 2026-03\n特征: 高频清扫\n清洁效率: 良好\n耗材: 正常\n对比: 稳定",
        history_result: str = "最近3个月趋势稳定",
    ):
        agent = ReactAgent.__new__(ReactAgent)
        agent.chat_model = DummyModel(answer=answer)
        user_id_tool = DummyTool("get_user_id", user_id_result)
        get_current_month_tool = DummyTool("get_current_month", current_month_result)
        weather_tool = DummyTool("get_weather", weather_result)
        rag_tool = DummyTool("rag_summarize", rag_result)
        external_data_tool = DummyTool("fetch_external_data", external_data_result)
        history_tool = DummyTool("fetch_external_history", history_result)
        agent.tools = [user_id_tool, get_current_month_tool, weather_tool, rag_tool, external_data_tool, history_tool]
        agent.tool_map = {tool.name: tool for tool in agent.tools}
        agent.normal_graph = agent._build_normal_graph()
        agent.report_graph = agent._build_report_graph()
        return agent, weather_tool, rag_tool, get_current_month_tool, user_id_tool, external_data_tool, history_tool

    def _build_runtime_context(self):
        return {
            "report": False,
            "force_report_agent": False,
            "user_id": "1001",
            "city": "北京",
            "session_id": "session_001",
            "session_summary": "",
            "recent_history": "",
            "user_memory_summary": "",
            "report_memory_summary": "",
            "trace_tool_calls": [],
            "react_trace": [],
            "react_step_count": 0,
            "react_stop_reason": "",
            "report_tool_sequence": [],
            "report_sequence_violation": False,
            "status_events": [],
        }

    def test_execute_weather_query_without_city_calls_location_then_weather(self):
        agent, weather_tool, _, _, _, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("今天天气怎么样？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_user_location", "get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "北京"})
        self.assertEqual(runtime_context["react_stop_reason"], "enough_information")
        self.assertEqual(runtime_context["react_step_count"], 2)

    def test_execute_weather_query_with_city_calls_weather_directly(self):
        agent, weather_tool, rag_tool, _, _, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("上海今天温度怎么样？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "上海"})
        self.assertEqual(rag_tool.calls, [])

    def test_execute_environment_query_without_explicit_weather_keywords_uses_weather_chain(self):
        agent, weather_tool, rag_tool, _, _, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()
        runtime_context["city"] = "杭州"

        result = agent.execute("现在我所在城市适不适合高频湿拖？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_user_location", "get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "杭州"})
        self.assertEqual(rag_tool.calls, [])
        self.assertEqual(runtime_context["react_stop_reason"], "enough_information")

    def test_execute_weather_plus_maintenance_query_calls_weather_then_rag(self):
        agent, weather_tool, rag_tool, _, _, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("广州湿度高，扫拖一体机要怎么保养？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_weather", "rag_summarize"])
        self.assertEqual(weather_tool.calls[0], {"city": "广州"})
        self.assertEqual(rag_tool.calls[0], {"query": "广州湿度高 扫拖一体机要怎么保养"})

    def test_execute_knowledge_query_calls_rag_only(self):
        agent, _, rag_tool, _, _, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("滚刷更换后需要注意什么？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["rag_summarize"])
        self.assertEqual(rag_tool.calls[0], {"query": "滚刷更换后需要注意什么"})
        self.assertEqual(runtime_context["react_stop_reason"], "enough_information")

    def test_execute_report_query_uses_explicit_report_graph(self):
        agent, _, _, get_current_month_tool, _, external_data_tool, _ = self._build_agent(
            answer="# 扫地机器人使用情况报告与保养建议\n\n2026-03 使用总结"
        )
        runtime_context = self._build_runtime_context()
        runtime_context["force_report_agent"] = True

        result = agent.execute("生成我的本月使用报告", runtime_context)

        self.assertIn("2026-03", result)
        self.assertTrue(runtime_context["report"])
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_current_month", "fetch_external_data"])
        self.assertEqual(runtime_context["report_tool_sequence"], ["get_current_month", "fetch_external_data"])
        self.assertEqual(get_current_month_tool.calls, [{}])
        self.assertEqual(external_data_tool.calls[0]["month"], "2026-03")

    def test_execute_faq_003_keeps_water_flow_focus_term_in_prompt(self):
        agent, _, rag_tool, _, _, _, _ = self._build_agent(
            rag_result="木地板家庭拖地时建议控制出水量，避免地板受潮。"
        )
        runtime_context = self._build_runtime_context()

        result = agent.execute("木地板家庭选扫拖机器人要注意什么？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(rag_tool.calls[0], {"query": "木地板家庭选扫拖机器人要注意什么"})
        prompt_text = agent.chat_model.messages[-1].content
        self.assertIn("出水量", prompt_text)

    def test_execute_report_query_with_explicit_month_and_trend_calls_history(self):
        agent, _, _, _, _, external_data_tool, history_tool = self._build_agent(
            answer="# 扫地机器人使用情况报告与保养建议\n\n2025-06 趋势总结"
        )
        runtime_context = self._build_runtime_context()
        runtime_context["force_report_agent"] = True

        result = agent.execute("请给我生成2025-06的趋势报告并给建议", runtime_context)

        self.assertIn("2025-06", result)
        self.assertEqual(runtime_context["trace_tool_calls"], ["fetch_external_data", "fetch_external_history", "rag_summarize"])
        self.assertEqual(external_data_tool.calls[0], {"user_id": "1001", "month": "2025-06"})
        self.assertEqual(history_tool.calls[0], {"user_id": "1001", "months": 3})


if __name__ == "__main__":
    unittest.main()
