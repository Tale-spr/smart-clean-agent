import json
import shutil
import unittest
import uuid
from pathlib import Path

from smart_clean_agent.evaluation.service import (
    EvalCase,
    EvalPoint,
    EvalTrace,
    JudgeBasedSummary,
    JudgeResult,
    NormalSummary,
    ReportSummary,
    RuleBasedSummary,
    build_eval_trace,
    evaluate_case,
    load_eval_cases,
    run_evaluation,
    validate_report_tool_dependency,
    validate_tool_sequence,
    write_evaluation_outputs,
)


class EvalServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"eval_service_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def _write_dataset(self, records: list[dict]) -> Path:
        dataset_path = self.base_dir / "eval_cases.jsonl"
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return dataset_path

    def test_load_eval_cases_parses_jsonl_and_points(self):
        dataset_path = self._write_dataset(
            [
                {
                    "case_id": "faq_001",
                    "query": "测试问题",
                    "category": "faq",
                    "user_id": "1002",
                    "city": "上海",
                    "expected_route": "normal",
                    "required_tools": [],
                    "optional_tools": ["get_weather"],
                    "required_points": [
                        {"point_id": "rp_01", "label": "关键词A", "aliases": ["关键词A", "别名A"]},
                        {"label": "关键词B", "aliases": ["别名B"]},
                    ],
                    "optional_points": [{"label": "可选信息", "aliases": ["可选信息", "补充说明"]}],
                    "expected_retrieval_mode": "required",
                    "notes": "测试样例",
                    "forbidden_tools": [],
                    "allow_no_tool": True,
                }
            ]
        )

        cases = load_eval_cases(dataset_path)

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.case_id, "faq_001")
        self.assertEqual(case.expected_route, "normal")
        self.assertEqual(case.required_tools, [])
        self.assertEqual(case.optional_tools, ["get_weather"])
        self.assertEqual(case.required_points[0].aliases, ["关键词A", "别名A"])
        self.assertEqual(case.required_points[1].aliases[0], "关键词B")
        self.assertEqual(case.optional_points[0].label, "可选信息")
        self.assertEqual(case.expected_retrieval_mode, "required")
        self.assertEqual(case.notes, "测试样例")

    def test_load_eval_cases_rejects_invalid_tools(self):
        dataset_path = self._write_dataset(
            [
                {
                    "case_id": "bad_001",
                    "query": "测试问题",
                    "category": "faq",
                    "expected_route": "normal",
                    "required_tools": ["fetch_external_data"],
                    "optional_tools": [],
                    "required_points": [{"label": "要点", "aliases": ["要点"]}],
                    "optional_points": [],
                    "expected_retrieval_mode": "required",
                }
            ]
        )

        with self.assertRaises(ValueError):
            load_eval_cases(dataset_path)

    def test_load_eval_cases_rejects_invalid_retrieval_mode(self):
        dataset_path = self._write_dataset(
            [
                {
                    "case_id": "bad_002",
                    "query": "测试问题",
                    "category": "faq",
                    "expected_route": "normal",
                    "required_tools": ["rag_summarize"],
                    "optional_tools": [],
                    "required_points": [{"label": "要点", "aliases": ["要点"]}],
                    "optional_points": [],
                    "expected_retrieval_mode": "sometimes",
                }
            ]
        )

        with self.assertRaises(ValueError):
            load_eval_cases(dataset_path)

    def test_run_evaluation_returns_rule_and_judge_summaries(self):
        cases = [
            EvalCase(
                case_id="faq_001",
                query="问题1",
                category="faq",
                expected_route="normal",
                required_tools=[],
                optional_tools=["rag_summarize"],
                required_points=[EvalPoint("rp_01", "导航", ["导航", "路线规划"])],
                optional_points=[EvalPoint("op_01", "建议", ["建议"])],
                expected_retrieval_mode="required",
                allow_no_tool=True,
            ),
            EvalCase(
                case_id="report_001",
                query="生成2025-03报告",
                category="report_generation",
                expected_route="report",
                required_tools=["get_user_id", "get_current_month", "fill_context_for_report", "fetch_external_data"],
                optional_tools=["fetch_external_history"],
                required_points=[
                    EvalPoint("rp_01", "报告", ["报告", "总结"]),
                    EvalPoint("rp_02", "建议", ["建议", "优化建议"]),
                ],
                optional_points=[],
                expected_retrieval_mode="optional",
                target_month="2025-03",
            ),
        ]

        responses = {
            "faq_001": (
                "这里给出导航路线规划建议",
                build_eval_trace(
                    ["rag_summarize"],
                    [{"source": "a.txt", "snippet": "导航资料"}],
                    step_count=1,
                    stop_reason="enough_information",
                    execution_mode="normal",
                ),
            ),
            "report_001": (
                "这是2025年3月的一份总结，没有后续动作。",
                build_eval_trace(
                    ["get_user_id", "get_current_month", "fill_context_for_report", "fetch_external_data"],
                    [],
                    step_count=2,
                    stop_reason="enough_information",
                    execution_mode="report",
                ),
            ),
        }

        def judge(case, answer, trace):
            return JudgeResult(
                enabled=True,
                correctness_score=4,
                completeness_score=3,
                groundedness_score=4,
                tool_usage_score=5,
                report_quality_score=4 if case.category == "report_generation" else 0,
                passed=case.case_id == "faq_001",
                reason="测试 judge",
            )

        results, rule_summary, judge_summary, normal_summary, report_summary = run_evaluation(
            cases,
            lambda case: responses[case.case_id],
            judge=judge,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].rule_based.content_pass)
        self.assertAlmostEqual(results[0].rule_based.required_point_hit_rate, 1.0)
        self.assertTrue(results[1].rule_based.route_correct)
        self.assertTrue(results[1].rule_based.required_tools_present)
        self.assertTrue(results[1].rule_based.tool_sequence_valid)
        self.assertFalse(results[1].rule_based.content_pass)
        self.assertIsInstance(rule_summary, RuleBasedSummary)
        self.assertAlmostEqual(rule_summary.route_correct_rate, 1.0)
        self.assertAlmostEqual(rule_summary.required_tool_pass_rate, 1.0)
        self.assertAlmostEqual(rule_summary.content_pass_rate, 0.5)
        self.assertAlmostEqual(rule_summary.report_generation_success_rate, 0.0)
        self.assertIsInstance(normal_summary, NormalSummary)
        self.assertAlmostEqual(normal_summary.tool_usage_valid_rate, 1.0)
        self.assertIsInstance(report_summary, ReportSummary)
        self.assertAlmostEqual(report_summary.tool_dependency_valid_rate, 1.0)
        self.assertAlmostEqual(report_summary.time_consistency_valid_rate, 1.0)
        self.assertIsInstance(judge_summary, JudgeBasedSummary)
        self.assertTrue(judge_summary.enabled)
        self.assertAlmostEqual(judge_summary.judge_pass_rate, 0.5)

    def test_evaluate_case_identifies_unexpected_tools_and_missing_points(self):
        case = EvalCase(
            case_id="env_001",
            query="天气会影响拖地吗",
            category="environment_fit",
            expected_route="normal",
            required_tools=["get_weather"],
            optional_tools=["rag_summarize"],
            required_points=[EvalPoint("rp_01", "天气", ["天气"]), EvalPoint("rp_02", "拖地", ["拖地"])],
            optional_points=[],
            expected_retrieval_mode="forbidden",
        )

        result = evaluate_case(
            case,
            lambda _: (
                "建议你留意天气因素。",
                EvalTrace(
                    tool_calls=["get_weather", "fetch_external_data"],
                    retrieved_docs=[],
                    retrieval_hit=False,
                    step_count=1,
                    stop_reason="enough_information",
                    tool_sequence_valid=True,
                    execution_mode="normal",
                ),
            ),
        )

        self.assertTrue(result.rule_based.required_tools_present)
        self.assertTrue(result.rule_based.unexpected_tools_present)
        self.assertEqual(result.rule_based.unexpected_tools, ["fetch_external_data"])
        self.assertEqual(result.rule_based.missing_required_points, ["拖地"])
        self.assertFalse(result.rule_based.content_pass)
        self.assertFalse(result.rule_based.tool_usage_valid)

    def test_environment_case_allows_missing_optional_location_tool(self):
        case = EvalCase(
            case_id="env_002",
            query="我这边天气干燥，对扫地机器人使用有什么影响？",
            category="environment_fit",
            expected_route="normal",
            required_tools=["get_weather"],
            optional_tools=["get_user_location", "rag_summarize"],
            required_points=[EvalPoint("rp_01", "干燥环境", ["干燥"]), EvalPoint("rp_02", "使用影响", ["滤网"])],
            optional_points=[],
            expected_retrieval_mode="optional",
            user_id="1001",
            city="北京",
        )

        result = evaluate_case(
            case,
            lambda _: (
                "干燥环境下要注意滤网清理。",
                EvalTrace(
                    tool_calls=["get_weather"],
                    retrieved_docs=[],
                    retrieval_hit=False,
                    step_count=1,
                    stop_reason="enough_information",
                    tool_sequence_valid=True,
                    execution_mode="normal",
                ),
            ),
        )

        self.assertTrue(result.rule_based.required_tools_present)
        self.assertTrue(result.rule_based.content_pass)
        self.assertTrue(result.rule_based.tool_usage_valid)

    def test_evaluate_case_degrades_gracefully_when_judge_fails(self):
        case = EvalCase(
            case_id="faq_001",
            query="测试问题",
            category="faq",
            expected_route="normal",
            required_tools=[],
            optional_tools=["rag_summarize"],
            required_points=[EvalPoint("rp_01", "关键词", ["关键词"])],
            optional_points=[],
            expected_retrieval_mode="required",
            allow_no_tool=True,
        )

        result = evaluate_case(
            case,
            lambda _: (
                "包含关键词的回答",
                build_eval_trace(
                    ["rag_summarize"],
                    [{"source": "doc.txt", "snippet": "关键词资料"}],
                    execution_mode="normal",
                ),
            ),
            judge=lambda *_: (_ for _ in ()).throw(ValueError("groundedness_score 必须为 0-5 的整数")),
        )

        self.assertTrue(result.rule_based.content_pass)
        self.assertTrue(result.judge_based.enabled)
        self.assertFalse(result.judge_based.passed)
        self.assertIn("Judge 执行失败", result.judge_based.reason)

    def test_normal_case_allows_no_tool_when_case_enables_it(self):
        case = EvalCase(
            case_id="faq_009",
            query="你好",
            category="faq",
            expected_route="normal",
            required_tools=[],
            optional_tools=["rag_summarize"],
            required_points=[EvalPoint("rp_01", "你好", ["你好"])],
            optional_points=[],
            expected_retrieval_mode="optional",
            allow_no_tool=True,
        )

        result = evaluate_case(
            case,
            lambda _: (
                "你好，有什么可以帮你？",
                build_eval_trace([], [], execution_mode="normal"),
            ),
        )

        self.assertTrue(result.rule_based.tool_usage_valid)
        self.assertEqual(result.rule_based.unnecessary_tool_calls, [])

    def test_normal_case_flags_repeated_tool_calls(self):
        case = EvalCase(
            case_id="faq_010",
            query="重复调用测试",
            category="faq",
            expected_route="normal",
            required_tools=[],
            optional_tools=["rag_summarize"],
            required_points=[EvalPoint("rp_01", "测试", ["测试"])],
            optional_points=[],
            expected_retrieval_mode="optional",
            allow_no_tool=True,
        )

        result = evaluate_case(
            case,
            lambda _: (
                "测试",
                EvalTrace(
                    tool_calls=["rag_summarize", "rag_summarize"],
                    retrieved_docs=[],
                    retrieval_hit=False,
                    execution_mode="normal",
                ),
            ),
        )

        self.assertEqual(result.rule_based.repeated_tool_calls, ["rag_summarize"])
        self.assertFalse(result.rule_based.tool_usage_valid)

    def test_validate_tool_sequence_checks_weather_order(self):
        self.assertFalse(validate_tool_sequence(["get_weather", "get_user_location"]))
        self.assertTrue(validate_tool_sequence(["get_user_location", "get_weather"]))
        self.assertFalse(validate_tool_sequence(["rag_summarize", "rag_summarize"]))

    def test_validate_report_tool_dependency_checks_required_tools_and_dependency_order(self):
        required_tools = ["get_user_id", "get_current_month", "fill_context_for_report", "fetch_external_data"]
        self.assertTrue(
            validate_report_tool_dependency(
                ["get_user_id", "get_current_month", "fill_context_for_report", "fetch_external_data", "rag_summarize"],
                required_tools,
            )
        )
        self.assertFalse(
            validate_report_tool_dependency(
                ["get_user_id", "fill_context_for_report", "fetch_external_data"],
                required_tools,
            )
        )
        self.assertFalse(
            validate_report_tool_dependency(
                ["fetch_external_data", "fill_context_for_report", "get_user_id", "get_current_month"],
                required_tools,
            )
        )

    def test_report_case_flags_future_month_trend_as_time_inconsistent(self):
        case = EvalCase(
            case_id="report_009",
            query="请生成我2025-06的报告",
            category="report_generation",
            expected_route="report",
            required_tools=["get_user_id", "get_current_month", "fill_context_for_report", "fetch_external_data"],
            optional_tools=["fetch_external_history", "rag_summarize"],
            required_points=[EvalPoint("rp_01", "报告", ["报告"])],
            optional_points=[],
            expected_retrieval_mode="optional",
            target_month="2025-06",
        )

        result = evaluate_case(
            case,
            lambda _: (
                "这是2025年6月报告，同时参考了2025年10月-12月趋势。",
                build_eval_trace(
                    ["get_user_id", "get_current_month", "fill_context_for_report", "fetch_external_data", "fetch_external_history"],
                    [],
                    execution_mode="report",
                ),
            ),
        )

        self.assertFalse(result.rule_based.time_consistency_valid)
        self.assertTrue(result.rule_based.tool_dependency_valid)

    def test_write_evaluation_outputs_creates_new_json_shape(self):
        result = evaluate_case(
            EvalCase(
                case_id="faq_001",
                query="测试问题",
                category="faq",
                expected_route="normal",
                required_tools=[],
                optional_tools=["rag_summarize"],
                required_points=[EvalPoint("rp_01", "关键词", ["关键词"])],
                optional_points=[],
                expected_retrieval_mode="required",
                allow_no_tool=True,
            ),
            lambda case: (
                "包含关键词的回答",
                build_eval_trace(
                    ["rag_summarize"],
                    [{"source": "data.txt", "snippet": "关键词资料"}],
                    execution_mode=case.expected_route,
                ),
            ),
        )
        rule_summary = RuleBasedSummary(
            total_cases=1,
            route_correct_rate=1.0,
            required_tool_pass_rate=1.0,
            tool_sequence_valid_rate=1.0,
            retrieval_mode_valid_rate=1.0,
            content_pass_rate=1.0,
            required_point_hit_rate_avg=1.0,
            optional_point_hit_rate_avg=1.0,
            report_generation_success_rate=0.0,
        )
        judge_summary = JudgeBasedSummary(enabled=False)
        normal_summary = NormalSummary(
            total_cases=1,
            content_pass_rate=1.0,
            tool_usage_valid_rate=1.0,
            unexpected_tool_rate=0.0,
            required_point_hit_rate_avg=1.0,
            judge_pass_rate=None,
        )
        report_summary = ReportSummary(
            total_cases=0,
            required_tools_present_rate=0.0,
            tool_dependency_valid_rate=0.0,
            time_consistency_valid_rate=0.0,
            content_pass_rate=0.0,
            avg_groundedness_score=None,
            avg_report_quality_score=None,
        )

        json_path, csv_path = write_evaluation_outputs(
            [result],
            rule_summary,
            judge_summary,
            normal_summary,
            report_summary,
            self.base_dir,
        )

        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("rule_based_summary", payload)
        self.assertIn("judge_based_summary", payload)
        self.assertIn("normal_summary", payload)
        self.assertIn("report_summary", payload)
        self.assertIn("rule_based", payload["results"][0])
        self.assertEqual(payload["results"][0]["expected_route"], "normal")


if __name__ == "__main__":
    unittest.main()
