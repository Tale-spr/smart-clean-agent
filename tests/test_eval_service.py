import json
import shutil
import unittest
import uuid
from pathlib import Path

from smart_clean_agent.evaluation.service import (
    EvalCase,
    EvalSummary,
    build_eval_trace,
    evaluate_case,
    load_eval_cases,
    run_evaluation,
    summarize_results,
    write_evaluation_outputs,
)


class EvalServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"eval_service_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def _write_dataset(self, content: str) -> Path:
        dataset_path = self.base_dir / "eval_cases.csv"
        dataset_path.write_text(content, encoding="utf-8")
        return dataset_path

    def test_load_eval_cases_parses_rows(self):
        dataset_path = self._write_dataset(
            "case_id,query,category,expected_keywords,expected_tool,expected_retrieval_hit,user_id,city\n"
            "case_001,测试问题,faq,关键词A|关键词B,rag_summarize,true,1002,上海\n"
        )

        cases = load_eval_cases(dataset_path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "case_001")
        self.assertEqual(cases[0].expected_keywords, ["关键词A", "关键词B"])
        self.assertTrue(cases[0].expected_retrieval_hit)
        self.assertEqual(cases[0].user_id, "1002")
        self.assertEqual(cases[0].city, "上海")

    def test_load_eval_cases_raises_when_missing_fields(self):
        dataset_path = self._write_dataset("query,category\n测试问题,faq\n")

        with self.assertRaises(ValueError):
            load_eval_cases(dataset_path)

    def test_load_eval_cases_raises_when_empty(self):
        dataset_path = self._write_dataset(
            "case_id,query,category,expected_keywords,expected_tool,expected_retrieval_hit\n"
        )

        with self.assertRaises(ValueError):
            load_eval_cases(dataset_path)

    def test_load_eval_cases_raises_when_category_invalid(self):
        dataset_path = self._write_dataset(
            "case_id,query,category,expected_keywords,expected_tool,expected_retrieval_hit\n"
            "case_001,测试问题,unknown,关键词,rag_summarize,true\n"
        )

        with self.assertRaises(ValueError):
            load_eval_cases(dataset_path)

    def test_evaluate_case_and_summary(self):
        cases = [
            EvalCase(
                case_id="case_001",
                query="问题1",
                category="faq",
                expected_keywords=["导航"],
                expected_tool="rag_summarize",
                expected_retrieval_hit=True,
            ),
            EvalCase(
                case_id="case_002",
                query="问题2",
                category="report_generation",
                expected_keywords=["报告", "建议"],
                expected_tool="fill_context_for_report",
                expected_retrieval_hit=False,
            ),
        ]

        responses = {
            "case_001": ("这里有导航建议", build_eval_trace(["rag_summarize"], [{"source": "a.txt"}])),
            "case_002": ("这是报告，但没有具体结论", build_eval_trace(["fill_context_for_report"], [])),
        }

        results, summary = run_evaluation(cases, lambda case: responses[case.case_id])

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].answer_keyword_hit)
        self.assertFalse(results[1].answer_keyword_hit)
        self.assertAlmostEqual(summary.answer_keyword_hit_rate, 0.5)
        self.assertAlmostEqual(summary.tool_call_success_rate, 1.0)
        self.assertAlmostEqual(summary.retrieval_hit_rate, 1.0)
        self.assertAlmostEqual(summary.report_generation_success_rate, 0.0)

    def test_summarize_results_raises_when_empty(self):
        with self.assertRaises(ValueError):
            summarize_results([])

    def test_write_evaluation_outputs_creates_json_and_csv(self):
        result = evaluate_case(
            EvalCase(
                case_id="case_001",
                query="测试问题",
                category="faq",
                expected_keywords=["关键词"],
                expected_tool="rag_summarize",
                expected_retrieval_hit=True,
            ),
            lambda case: ("包含关键词的回答", build_eval_trace(["rag_summarize"], [{"source": "data.txt"}])),
        )
        summary = EvalSummary(
            total_cases=1,
            answer_keyword_hit_rate=1.0,
            tool_call_success_rate=1.0,
            retrieval_hit_rate=1.0,
            report_generation_success_rate=0.0,
            cases_with_expected_tool=1,
            cases_with_expected_retrieval=1,
            report_generation_cases=0,
        )

        json_path, csv_path = write_evaluation_outputs([result], summary, self.base_dir)

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["total_cases"], 1)
        self.assertEqual(payload["results"][0]["case_id"], "case_001")


if __name__ == "__main__":
    unittest.main()

