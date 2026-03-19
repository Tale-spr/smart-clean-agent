import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from smart_clean_agent.evaluation import run
from smart_clean_agent.evaluation.service import EvalSummary


class EvalRunTestCase(unittest.TestCase):
    def test_main_returns_zero_on_success(self):
        fake_cases = [object()]
        fake_results = [object()]
        fake_summary = EvalSummary(
            total_cases=1,
            answer_keyword_hit_rate=1.0,
            tool_call_success_rate=1.0,
            retrieval_hit_rate=1.0,
            report_generation_success_rate=1.0,
            cases_with_expected_tool=1,
            cases_with_expected_retrieval=1,
            report_generation_cases=1,
        )

        with patch("smart_clean_agent.evaluation.run.load_eval_cases", return_value=fake_cases):
            with patch("smart_clean_agent.evaluation.run.AgentEvaluationExecutor") as mock_executor_cls:
                executor = Mock()
                mock_executor_cls.return_value = executor
                with patch("smart_clean_agent.evaluation.run.run_evaluation", return_value=(fake_results, fake_summary)):
                    with patch("smart_clean_agent.evaluation.run.write_evaluation_outputs", return_value=(Path("a.json"), Path("a.csv"))):
                        result = run.main()

        self.assertEqual(result, 0)
        mock_executor_cls.assert_called_once()

    def test_main_returns_one_on_failure(self):
        with patch("smart_clean_agent.evaluation.run.load_eval_cases", side_effect=ValueError("bad dataset")):
            result = run.main()

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()

