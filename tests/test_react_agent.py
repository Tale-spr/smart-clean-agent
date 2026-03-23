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


class DummyReportAgent:
    def stream(self, input_dict, stream_mode="values", context=None):
        yield {"messages": [AIMessage(content="报告内容")]}


class ReactAgentTestCase(unittest.TestCase):
    def _build_agent(self, *, weather_result: str = "城市: 北京\n天气: 晴", rag_result: str = "建议定期更换滚刷。"):
        agent = ReactAgent.__new__(ReactAgent)
        agent.chat_model = DummyModel()
        get_current_month_tool = DummyTool("get_current_month", "2026-03")
        weather_tool = DummyTool("get_weather", weather_result)
        rag_tool = DummyTool("rag_summarize", rag_result)
        agent.tools = [get_current_month_tool, weather_tool, rag_tool]
        agent.tool_map = {tool.name: tool for tool in agent.tools}
        agent.report_agent = DummyReportAgent()
        agent.agent = None
        agent.normal_graph = agent._build_normal_graph()
        return agent, weather_tool, rag_tool, get_current_month_tool

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
        agent, weather_tool, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("今天天气怎么样？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_user_location", "get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "北京"})
        self.assertEqual(runtime_context["react_stop_reason"], "enough_information")
        self.assertEqual(runtime_context["react_step_count"], 2)

    def test_execute_weather_query_with_city_calls_weather_directly(self):
        agent, weather_tool, rag_tool, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("上海今天温度怎么样？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "上海"})
        self.assertEqual(rag_tool.calls, [])

    def test_execute_environment_query_without_explicit_weather_keywords_uses_weather_chain(self):
        agent, weather_tool, rag_tool, _ = self._build_agent()
        runtime_context = self._build_runtime_context()
        runtime_context["city"] = "杭州"

        result = agent.execute("现在我所在城市适不适合高频湿拖？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_user_location", "get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "杭州"})
        self.assertEqual(rag_tool.calls, [])
        self.assertEqual(runtime_context["react_stop_reason"], "enough_information")

    def test_execute_weather_plus_maintenance_query_calls_weather_then_rag(self):
        agent, weather_tool, rag_tool, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("广州湿度高，扫拖一体机要怎么保养？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_weather", "rag_summarize"])
        self.assertEqual(weather_tool.calls[0], {"city": "广州"})
        self.assertEqual(rag_tool.calls[0], {"query": "广州湿度高 扫拖一体机要怎么保养"})

    def test_execute_knowledge_query_calls_rag_only(self):
        agent, _, rag_tool, _ = self._build_agent()
        runtime_context = self._build_runtime_context()

        result = agent.execute("滚刷更换后需要注意什么？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["rag_summarize"])
        self.assertEqual(rag_tool.calls[0], {"query": "滚刷更换后需要注意什么"})
        self.assertEqual(runtime_context["react_stop_reason"], "enough_information")

    def test_execute_report_query_uses_report_agent_path(self):
        agent, _, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()
        runtime_context["force_report_agent"] = True

        result = agent.execute("生成我的本月使用报告", runtime_context)

        self.assertEqual(result, "报告内容")
        self.assertTrue(runtime_context["report"])
        self.assertEqual(
            runtime_context["trace_tool_calls"][:3],
            ["get_user_id", "get_current_month", "fill_context_for_report"],
        )
        self.assertEqual(
            runtime_context["report_tool_sequence"][:3],
            ["get_user_id", "get_current_month", "fill_context_for_report"],
        )

    def test_execute_faq_003_keeps_water_flow_focus_term_in_prompt(self):
        agent, _, rag_tool, _ = self._build_agent(
            rag_result="木地板家庭拖地时建议控制出水量，避免地板受潮。"
        )
        runtime_context = self._build_runtime_context()

        result = agent.execute("木地板家庭选扫拖机器人要注意什么？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(rag_tool.calls[0], {"query": "木地板家庭选扫拖机器人要注意什么"})
        prompt_text = agent.chat_model.messages[-1].content
        self.assertIn("出水量", prompt_text)


if __name__ == "__main__":
    unittest.main()
