import unittest

from smart_clean_agent.evaluation.compare import compare_results


class EvalCompareTestCase(unittest.TestCase):
    def test_compare_results_reports_summary_diff_and_case_changes(self):
        before_payload = {
            "generated_at": "before",
            "rule_based_summary": {"content_pass_rate": 0.5, "route_correct_rate": 1.0},
            "judge_based_summary": {"enabled": False},
            "results": [
                {
                    "case_id": "case_001",
                    "rule_based": {
                        "route_correct": True,
                        "required_tools_present": True,
                        "tool_sequence_valid": True,
                        "retrieval_mode_valid": True,
                        "required_point_hit_rate": 0.0,
                    },
                    "judge_based": {"enabled": False},
                }
            ],
        }
        after_payload = {
            "generated_at": "after",
            "rule_based_summary": {"content_pass_rate": 1.0, "route_correct_rate": 1.0},
            "judge_based_summary": {"enabled": False},
            "results": [
                {
                    "case_id": "case_001",
                    "rule_based": {
                        "route_correct": True,
                        "required_tools_present": True,
                        "tool_sequence_valid": True,
                        "retrieval_mode_valid": True,
                        "required_point_hit_rate": 1.0,
                    },
                    "judge_based": {"enabled": False},
                }
            ],
        }

        comparison = compare_results(before_payload, after_payload)

        self.assertEqual(comparison["rule_based_summary_diff"]["content_pass_rate"], 0.5)
        self.assertEqual(len(comparison["improved_cases"]), 1)
        self.assertEqual(comparison["improved_cases"][0]["case_id"], "case_001")
        self.assertEqual(comparison["regressed_cases"], [])


if __name__ == "__main__":
    unittest.main()
