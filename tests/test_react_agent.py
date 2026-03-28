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
        normalization_answer: str | None = None,
    ):
        agent = ReactAgent.__new__(ReactAgent)
        agent.chat_model = DummyModel(answer=answer)
        agent.normalization_model = DummyModel(
            answer=normalization_answer
            or '{"normalized_query":"今天天气怎么样","intent":"weather","needs_weather":true,"needs_knowledge":false,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["city","weather"],"reason":"天气查询","confidence":"high"}'
        )
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
            "user_memory_payload": {},
            "retrieved_user_memory_summary": "",
            "retrieved_user_memory_fields": [],
            "memory_retrieval_reason": "",
            "is_new_session_first_turn": False,
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
        agent, weather_tool, rag_tool, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"上海今天温度怎么样","intent":"weather","needs_weather":true,"needs_knowledge":false,"city":"上海","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["weather"],"reason":"显式城市天气查询","confidence":"high"}'
        )
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
        agent.normalization_model.answer = '{"normalized_query":"当前城市是否适合高频湿拖","intent":"weather","needs_weather":true,"needs_knowledge":false,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["city","weather"],"reason":"环境适配问题","confidence":"high"}'

        result = agent.execute("现在我所在城市适不适合高频湿拖？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_user_location", "get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "杭州"})
        self.assertEqual(rag_tool.calls, [])
        self.assertEqual(runtime_context["react_stop_reason"], "enough_information")

    def test_execute_weather_plus_maintenance_query_calls_weather_then_rag(self):
        agent, weather_tool, rag_tool, _, _, _, _ = self._build_agent()
        runtime_context = self._build_runtime_context()
        agent.normalization_model.answer = '{"normalized_query":"广州湿度高，扫拖一体机要怎么保养","intent":"combined","needs_weather":true,"needs_knowledge":true,"city":"广州","user_weather_premise_type":"humid","user_weather_premise_text":"用户明确提到当前环境潮湿或湿度较高","missing_slots":["weather","knowledge"],"reason":"天气与保养建议组合问题","confidence":"high"}'

        result = agent.execute("广州湿度高，扫拖一体机要怎么保养？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_weather", "rag_summarize"])
        self.assertEqual(weather_tool.calls[0], {"city": "广州"})
        self.assertEqual(rag_tool.calls[0], {"query": "广州湿度高 扫拖一体机要怎么保养"})

    def test_execute_knowledge_query_calls_rag_only(self):
        agent, _, rag_tool, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"滚刷更换后需要注意什么","intent":"knowledge","needs_weather":false,"needs_knowledge":true,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["knowledge"],"reason":"知识型问题","confidence":"high"}'
        )
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
        agent.normalization_model.answer = '{"normalized_query":"木地板家庭选扫拖机器人要注意什么","intent":"knowledge","needs_weather":false,"needs_knowledge":true,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["knowledge"],"reason":"知识型问题","confidence":"high"}'

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

    def test_execute_environment_query_with_rain_premise_returns_guarded_answer(self):
        agent, weather_tool, _, _, _, _, _ = self._build_agent(
            weather_result="城市: 深圳\n天气: 多云\n温度: 7\n湿度: 26"
        )
        runtime_context = self._build_runtime_context()
        runtime_context["city"] = "深圳"
        agent.normalization_model.answer = '{"normalized_query":"当前城市最近下雨，机器人清洁要注意什么","intent":"weather","needs_weather":true,"needs_knowledge":false,"city":"","user_weather_premise_type":"rain","user_weather_premise_text":"用户明确提到最近下雨或处于雨天场景","missing_slots":["city","weather"],"reason":"雨天环境问题","confidence":"high"}'

        result = agent.execute("我所在的城市最近下雨，机器人清洁要注意什么？", runtime_context)

        self.assertIn("防滑", result)
        self.assertIn("防潮", result)
        self.assertNotIn("结论：", result)
        self.assertNotIn("依据：", result)
        self.assertNotIn("建议：", result)
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_user_location", "get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "深圳"})

    def test_execute_report_query_without_monthly_record_returns_limited_report(self):
        agent, _, _, get_current_month_tool, _, external_data_tool, _ = self._build_agent(
            answer="这段回答不应被直接使用。",
            external_data_result="",
        )
        runtime_context = self._build_runtime_context()
        runtime_context["force_report_agent"] = True

        result = agent.execute("生成我的本月使用报告", runtime_context)

        self.assertIn("未获取到足够完整的本月使用记录", result)
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_current_month", "fetch_external_data"])
        self.assertEqual(get_current_month_tool.calls, [{}])
        self.assertEqual(external_data_tool.calls[0]["month"], "2026-03")

    def test_normalize_query_combined_intent_turns_ambiguous_dragging_question_into_weather_and_knowledge(self):
        agent, weather_tool, rag_tool, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"当前天气是否适合拖地","intent":"combined","needs_weather":true,"needs_knowledge":true,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["city","weather","knowledge"],"reason":"天气与拖地建议组合问题","confidence":"high"}'
        )
        runtime_context = self._build_runtime_context()
        runtime_context["city"] = "北京"

        result = agent.execute("我这边这种天适不适合拖地？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_user_location", "get_weather", "rag_summarize"])
        self.assertEqual(weather_tool.calls[0], {"city": "北京"})
        self.assertEqual(rag_tool.calls[0], {"query": "当前天气是否适合拖地"})
        self.assertFalse(runtime_context["react_trace"] == [])
        self.assertIn("normalize", [item["phase"] for item in runtime_context["react_trace"]])

    def test_normalize_query_invalid_json_falls_back_to_heuristics(self):
        agent, weather_tool, _, _, _, _, _ = self._build_agent(normalization_answer="not json")
        runtime_context = self._build_runtime_context()

        result = agent.execute("上海今天温度怎么样？", runtime_context)

        self.assertEqual(result, "这是最终回答。")
        self.assertEqual(runtime_context["trace_tool_calls"], ["get_weather"])
        self.assertEqual(weather_tool.calls[0], {"city": "上海"})
        normalize_traces = [item for item in runtime_context["react_trace"] if item["phase"] == "normalize"]
        self.assertTrue(normalize_traces)
        self.assertIn("fallback=true", normalize_traces[0]["content"])

    def test_new_session_smalltalk_does_not_retrieve_long_term_memory(self):
        agent, _, _, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"你好","intent":"direct","needs_weather":false,"needs_knowledge":false,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":[],"reason":"寒暄","confidence":"high"}'
        )
        runtime_context = self._build_runtime_context()
        runtime_context["is_new_session_first_turn"] = True
        runtime_context["user_memory_payload"] = {
            "profile_snapshot": {"city": "上海", "house_type": "70㎡公寓", "floor_type": "瓷砖"},
            "preferences": ["湿拖偏好"],
            "environment": ["瓷砖", "小户型"],
            "cleaning_habits": ["高频清扫"],
            "pain_points": ["滤网维护"],
            "recent_focuses": ["拖地出水量"],
        }

        result = agent.execute("你好", runtime_context)

        self.assertIn("你好", result)
        self.assertEqual(runtime_context["retrieved_user_memory_summary"], "")
        self.assertEqual(runtime_context["retrieved_user_memory_fields"], [])
        self.assertEqual(runtime_context["memory_retrieval_reason"], "skip_for_smalltalk_or_weather")
        self.assertIsNone(agent.chat_model.messages)

    def test_capability_query_returns_natural_intro_without_model_generation(self):
        agent, _, _, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"你能帮我做什么","intent":"direct","needs_weather":false,"needs_knowledge":false,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":[],"reason":"能力询问","confidence":"high"}'
        )
        runtime_context = self._build_runtime_context()

        result = agent.execute("你能帮我做什么？", runtime_context)

        self.assertIn("我可以帮你", result)
        self.assertNotIn("结论：", result)
        self.assertIsNone(agent.chat_model.messages)

    def test_weather_query_does_not_retrieve_long_term_memory(self):
        agent, _, _, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"北京今天天气怎么样","intent":"weather","needs_weather":true,"needs_knowledge":false,"city":"北京","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["weather"],"reason":"天气查询","confidence":"high"}'
        )
        runtime_context = self._build_runtime_context()
        runtime_context["user_memory_payload"] = {
            "profile_snapshot": {"city": "北京", "house_type": "70㎡公寓", "floor_type": "木地板"},
            "preferences": ["湿拖偏好"],
            "environment": ["木地板", "养宠"],
            "cleaning_habits": ["高频清扫"],
            "pain_points": ["漏扫问题"],
            "recent_focuses": ["拖地出水量"],
        }

        agent.execute("北京今天天气怎么样？", runtime_context)

        self.assertEqual(runtime_context["retrieved_user_memory_summary"], "")
        self.assertEqual(runtime_context["retrieved_user_memory_fields"], [])
        self.assertEqual(runtime_context["memory_retrieval_reason"], "skip_for_smalltalk_or_weather")
        prompt_text = agent.chat_model.messages[-1].content
        self.assertIn("自然、简洁、直接", prompt_text)
        self.assertNotIn("结论 + 依据 + 建议", prompt_text)
        self.assertIn("不要使用“结论：”“依据：”“建议：”", prompt_text)

    def test_environment_query_retrieves_environment_and_preferences(self):
        agent, _, rag_tool, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"这种天气适不适合拖地","intent":"combined","needs_weather":true,"needs_knowledge":true,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["city","weather","knowledge"],"reason":"环境适配问题","confidence":"high"}'
        )
        runtime_context = self._build_runtime_context()
        runtime_context["user_memory_payload"] = {
            "profile_snapshot": {"city": "北京", "house_type": "70㎡公寓", "floor_type": "木地板"},
            "preferences": ["湿拖偏好"],
            "environment": ["木地板", "养宠"],
            "cleaning_habits": ["高频清扫"],
            "pain_points": ["漏扫问题"],
            "recent_focuses": ["拖地出水量"],
        }

        agent.execute("这种天气适不适合拖地？", runtime_context)

        self.assertIn("environment", runtime_context["retrieved_user_memory_fields"])
        self.assertIn("preferences", runtime_context["retrieved_user_memory_fields"])
        self.assertIn("木地板", runtime_context["retrieved_user_memory_summary"])
        self.assertIn("湿拖偏好", runtime_context["retrieved_user_memory_summary"])
        prompt_text = agent.chat_model.messages[-1].content
        self.assertIn("相关环境特征", prompt_text)
        self.assertIn("木地板", prompt_text)
        self.assertIn("自然、专业、像真实客服交流", prompt_text)
        self.assertNotIn("结论 + 依据 + 建议", prompt_text)
        self.assertEqual(rag_tool.calls[0], {"query": "这种天气适不适合拖地"})

    def test_troubleshooting_query_retrieves_pain_points(self):
        agent, _, rag_tool, _, _, _, _ = self._build_agent(
            normalization_answer='{"normalized_query":"最近总漏扫怎么办","intent":"knowledge","needs_weather":false,"needs_knowledge":true,"city":"","user_weather_premise_type":"","user_weather_premise_text":"","missing_slots":["knowledge"],"reason":"故障排查问题","confidence":"high"}'
        )
        runtime_context = self._build_runtime_context()
        runtime_context["user_memory_payload"] = {
            "profile_snapshot": {"city": "北京", "house_type": "70㎡公寓", "floor_type": "木地板"},
            "preferences": ["湿拖偏好"],
            "environment": ["木地板", "养宠"],
            "cleaning_habits": ["高频清扫"],
            "pain_points": ["漏扫问题", "滤网维护"],
            "recent_focuses": ["最近总漏扫怎么办", "滤网清洗"],
        }

        agent.execute("最近总漏扫怎么办？", runtime_context)

        self.assertIn("pain_points", runtime_context["retrieved_user_memory_fields"])
        self.assertIn("recent_focuses", runtime_context["retrieved_user_memory_fields"])
        self.assertIn("漏扫问题", runtime_context["retrieved_user_memory_summary"])
        self.assertEqual(rag_tool.calls[0], {"query": "最近总漏扫怎么办"})


if __name__ == "__main__":
    unittest.main()
