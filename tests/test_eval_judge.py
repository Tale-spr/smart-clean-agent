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


if __name__ == "__main__":
    unittest.main()
