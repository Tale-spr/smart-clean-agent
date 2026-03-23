import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from smart_clean_agent.evaluation import run
from smart_clean_agent.evaluation.service import EvalCase, JudgeBasedSummary, RuleBasedSummary


class EvalRunTestCase(unittest.TestCase):
    def test_build_eval_runtime_context_routes_report_case_to_report_mode(self):
        case = EvalCase(
            case_id="report_001",
            query="生成报告",
            category="report_generation",
            expected_route="report",
            required_tools=["get_user_id", "get_current_month", "fill_context_for_report", "fetch_external_data"],
            optional_tools=["fetch_external_history"],
            required_points=[],
            optional_points=[],
            expected_retrieval_mode="optional",
        )

        runtime_context = run.build_eval_runtime_context(case)

        self.assertTrue(runtime_context["report"])
        self.assertTrue(runtime_context["force_report_agent"])

    def test_build_eval_runtime_context_routes_normal_case_to_normal_mode(self):
        case = EvalCase(
            case_id="faq_001",
            query="普通问题",
            category="faq",
            expected_route="normal",
            required_tools=["rag_summarize"],
            optional_tools=[],
            required_points=[],
            optional_points=[],
            expected_retrieval_mode="required",
        )

        runtime_context = run.build_eval_runtime_context(case)

        self.assertFalse(runtime_context["report"])
        self.assertFalse(runtime_context["force_report_agent"])

    def test_main_returns_zero_on_success_without_judge(self):
        fake_cases = [object()]
        fake_results = [object()]
        fake_rule_summary = RuleBasedSummary(
            total_cases=1,
            route_correct_rate=1.0,
            required_tool_pass_rate=1.0,
            tool_sequence_valid_rate=1.0,
            retrieval_mode_valid_rate=1.0,
            content_pass_rate=1.0,
            required_point_hit_rate_avg=1.0,
            optional_point_hit_rate_avg=1.0,
            report_generation_success_rate=1.0,
        )
        fake_judge_summary = JudgeBasedSummary(enabled=False)

        with patch("smart_clean_agent.evaluation.run.load_eval_cases", return_value=fake_cases):
            with patch("smart_clean_agent.evaluation.run.AgentEvaluationExecutor") as mock_executor_cls:
                executor = Mock()
                mock_executor_cls.return_value = executor
                with patch(
                    "smart_clean_agent.evaluation.run.run_evaluation",
                    return_value=(fake_results, fake_rule_summary, fake_judge_summary),
                ):
                    with patch(
                        "smart_clean_agent.evaluation.run.write_evaluation_outputs",
                        return_value=(Path("a.json"), Path("a.csv")),
                    ):
                        result = run.main()

        self.assertEqual(result, 0)
        mock_executor_cls.assert_called_once_with(with_judge=False, judge_model_name=None)

    def test_main_returns_zero_on_success_with_judge(self):
        fake_rule_summary = RuleBasedSummary(
            total_cases=1,
            route_correct_rate=1.0,
            required_tool_pass_rate=1.0,
            tool_sequence_valid_rate=1.0,
            retrieval_mode_valid_rate=1.0,
            content_pass_rate=1.0,
            required_point_hit_rate_avg=1.0,
            optional_point_hit_rate_avg=1.0,
            report_generation_success_rate=1.0,
        )
        fake_judge_summary = JudgeBasedSummary(
            enabled=True,
            total_cases=1,
            judge_pass_rate=1.0,
            avg_correctness_score=5.0,
            avg_completeness_score=4.0,
            avg_groundedness_score=5.0,
            avg_tool_usage_score=5.0,
            avg_report_quality_score=0.0,
        )

        with patch("smart_clean_agent.evaluation.run.load_eval_cases", return_value=[object()]):
            with patch("smart_clean_agent.evaluation.run.AgentEvaluationExecutor") as mock_executor_cls:
                executor = Mock()
                executor.judge_case = Mock()
                mock_executor_cls.return_value = executor
                with patch(
                    "smart_clean_agent.evaluation.run.run_evaluation",
                    return_value=([object()], fake_rule_summary, fake_judge_summary),
                ):
                    with patch(
                        "smart_clean_agent.evaluation.run.write_evaluation_outputs",
                        return_value=(Path("a.json"), Path("a.csv")),
                    ):
                        result = run.main(argv=["--with-judge", "--judge-model", "qwen-plus"])

        self.assertEqual(result, 0)
        mock_executor_cls.assert_called_once_with(with_judge=True, judge_model_name="qwen-plus")

    def test_main_returns_one_on_failure(self):
        with patch("smart_clean_agent.evaluation.run.load_eval_cases", side_effect=ValueError("bad dataset")):
            result = run.main()

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
