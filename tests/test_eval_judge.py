import unittest
from unittest.mock import Mock, patch

from smart_clean_agent.evaluation.judge import JudgeEvaluator, _extract_json_payload, _validate_score
from smart_clean_agent.evaluation.service import EvalCase, EvalPoint, EvalTrace


class EvalJudgeTestCase(unittest.TestCase):
    def test_extract_json_payload_supports_fenced_json(self):
        payload = _extract_json_payload(
            """```json
            {"correctness_score": 5, "passed": true}
            ```"""
        )

        self.assertEqual(payload["correctness_score"], 5)
        self.assertTrue(payload["passed"])

    def test_validate_score_supports_float_and_string_value(self):
        self.assertEqual(_validate_score(4.0, "correctness_score"), 4)
        self.assertEqual(_validate_score("4", "correctness_score"), 4)
        self.assertEqual(_validate_score("4.5", "correctness_score"), 5)
        self.assertEqual(_validate_score("4分", "correctness_score"), 4)

    def test_validate_score_rejects_invalid_value(self):
        with self.assertRaises(ValueError):
            _validate_score(6, "correctness_score")

    @patch("smart_clean_agent.evaluation.judge.create_chat_model")
    def test_judge_evaluate_returns_structured_result(self, mock_create_chat_model):
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(
            content=(
                '{"correctness_score": 4, "completeness_score": 3, "groundedness_score": 5, '
                '"tool_usage_score": 4, "report_quality_score": 0, "passed": true, "reason": "回答基本正确"}'
            )
        )
        mock_create_chat_model.return_value = mock_model

        judge = JudgeEvaluator(model_name="qwen-plus")
        mock_create_chat_model.assert_called_once_with(model_name="qwen-plus", role="judge")
        result = judge.evaluate(
            EvalCase(
                case_id="faq_001",
                query="测试问题",
                category="faq",
                expected_route="normal",
                required_tools=["rag_summarize"],
                optional_tools=[],
                required_points=[EvalPoint("rp_01", "导航", ["导航"])],
                optional_points=[],
                expected_retrieval_mode="required",
            ),
            "这是导航相关的回答",
            EvalTrace(
                tool_calls=["rag_summarize"],
                retrieved_docs=[{"source": "a.txt", "snippet": "导航资料"}],
                retrieval_hit=True,
                execution_mode="normal",
            ),
        )

        self.assertTrue(result.enabled)
        self.assertEqual(result.correctness_score, 4)
        self.assertTrue(result.passed)
        self.assertEqual(result.reason, "回答基本正确")
        invoked_messages = mock_model.invoke.call_args.args[0]
        self.assertIn("tool_observations_summary", invoked_messages[-1].content)

    @patch("smart_clean_agent.evaluation.judge.create_chat_model")
    def test_judge_evaluate_normalizes_float_and_string_scores(self, mock_create_chat_model):
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(
            content=(
                '{"correctness_score": "4", "completeness_score": 3.0, "groundedness_score": "4.5", '
                '"tool_usage_score": "5分", "report_quality_score": 0, "passed": "true", "reason": "结构化输出"}'
            )
        )
        mock_create_chat_model.return_value = mock_model

        judge = JudgeEvaluator(model_name="qwen-plus")
        mock_create_chat_model.assert_called_once_with(model_name="qwen-plus", role="judge")
        result = judge.evaluate(
            EvalCase(
                case_id="faq_001",
                query="测试问题",
                category="faq",
                expected_route="normal",
                required_tools=["rag_summarize"],
                optional_tools=[],
                required_points=[EvalPoint("rp_01", "导航", ["导航"])],
                optional_points=[],
                expected_retrieval_mode="required",
            ),
            "这是导航相关的回答",
            EvalTrace(
                tool_calls=["rag_summarize"],
                retrieved_docs=[{"source": "a.txt", "snippet": "导航资料"}],
                retrieval_hit=True,
                execution_mode="normal",
            ),
        )

        self.assertEqual(result.correctness_score, 4)
        self.assertEqual(result.completeness_score, 3)
        self.assertEqual(result.groundedness_score, 5)
        self.assertEqual(result.tool_usage_score, 5)
        self.assertTrue(result.passed)

    @patch("smart_clean_agent.evaluation.judge.create_chat_model")
    def test_judge_evaluate_includes_tool_evidence(self, mock_create_chat_model):
        mock_model = Mock()
        mock_model.invoke.return_value = Mock(
            content=(
                '{"correctness_score": 5, "completeness_score": 5, "groundedness_score": 5, '
                '"tool_usage_score": 5, "report_quality_score": 0, "passed": true, "reason": "包含工具证据"}'
            )
        )
        mock_create_chat_model.return_value = mock_model

        judge = JudgeEvaluator(model_name="qwen-plus")
        judge.evaluate(
            EvalCase(
                case_id="env_003",
                query="上海这种潮湿天气要不要降低出水量？",
                category="environment_fit",
                expected_route="normal",
                required_tools=["get_weather"],
                optional_tools=["rag_summarize"],
                required_points=[EvalPoint("rp_01", "潮湿", ["潮湿"])],
                optional_points=[],
                expected_retrieval_mode="optional",
            ),
            "建议先低档出水量。",
            EvalTrace(
                tool_calls=["get_weather"],
                tool_evidence=[{"tool_name": "get_weather", "summary": "上海 湿度 78%"}],
                execution_mode="normal",
            ),
        )

        invoked_messages = mock_model.invoke.call_args.args[0]
        self.assertIn("上海 湿度 78%", invoked_messages[-1].content)


if __name__ == "__main__":
    unittest.main()
