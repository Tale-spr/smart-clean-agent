import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from smart_clean_agent.agent.react_agent import ReactAgent


class FakeStreamingAgent:
    def stream(self, input_dict, stream_mode, context):
        yield {"messages": [HumanMessage(content="如何高效设置清洁计划")]}
        yield {"messages": [AIMessage(content="可以按房间和时间段设置清洁计划。")]}
        yield {"messages": [AIMessage(content="可以按房间和时间段设置清洁计划。建议工作日早晨清扫客厅。")]}


class ReactAgentTestCase(unittest.TestCase):
    def test_execute_stream_only_yields_assistant_delta(self):
        agent = ReactAgent.__new__(ReactAgent)
        agent.agent = FakeStreamingAgent()

        result = "".join(
            agent.execute_stream(
                "如何高效设置清洁计划",
                {
                    "report": False,
                    "user_id": "1001",
                    "city": "北京",
                    "session_id": "session_001",
                    "session_summary": "",
                    "recent_history": "",
                    "user_memory_summary": "",
                    "report_memory_summary": "",
                },
            )
        )

        self.assertNotIn("如何高效设置清洁计划", result)
        self.assertEqual(result, "可以按房间和时间段设置清洁计划。建议工作日早晨清扫客厅。")


if __name__ == "__main__":
    unittest.main()

