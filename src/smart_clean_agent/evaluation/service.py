import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_EVAL_DATASET = Path("data/eval/eval_cases.jsonl")
DEFAULT_RESULTS_DIR = Path("data/eval/results")
REQUIRED_CASE_FIELDS = {
    "case_id",
    "query",
    "category",
    "expected_route",
    "required_tools",
    "optional_tools",
    "required_points",
    "optional_points",
    "expected_retrieval_mode",
}
ALLOWED_CATEGORIES = {"faq", "troubleshooting", "environment_fit", "report_generation"}
NORMAL_CATEGORIES = {"faq", "troubleshooting", "environment_fit"}
ALLOWED_ROUTES = {"normal", "report"}
ALLOWED_RETRIEVAL_MODES = {"required", "optional", "forbidden"}
ALLOWED_TOOLS_BY_ROUTE = {
    "normal": {"get_user_location", "get_weather", "rag_summarize"},
    "report": {
        "get_user_id",
        "get_current_month",
        "fetch_external_data",
        "fetch_external_history",
        "rag_summarize",
    },
}
VALID_STOP_REASONS = {"", "enough_information", "max_steps_reached", "tool_failed", "unsupported_request", "consistency_failed"}
YEAR_MONTH_PATTERN = re.compile(r"(?P<year>\d{4})\s*(?:年|/|-)\s*(?P<month>\d{1,2})\s*月?")
YEAR_MONTH_RANGE_PATTERN = re.compile(
    r"(?P<year>\d{4})\s*(?:年|/|-)\s*(?P<start>\d{1,2})\s*月?\s*[—\-–~至到]+\s*(?P<end>\d{1,2})\s*月"
)


@dataclass
class EvalPoint:
    point_id: str
    label: str
    aliases: list[str]


@dataclass
class EvalCase:
    case_id: str
    query: str
    category: str
    expected_route: str
    required_tools: list[str]
    optional_tools: list[str]
    required_points: list[EvalPoint]
    optional_points: list[EvalPoint]
    expected_retrieval_mode: str
    user_id: str = "1001"
    city: str = "北京"
    notes: str = ""
    forbidden_tools: list[str] | None = None
    allow_no_tool: bool = False
    target_month: str = ""
    allowed_trend_window: list[str] | None = None


@dataclass
class EvalTrace:
    tool_calls: list[str] = field(default_factory=list)
    retrieved_docs: list[dict[str, str]] = field(default_factory=list)
    tool_evidence: list[dict[str, str]] = field(default_factory=list)
    retrieval_hit: bool = False
    step_count: int = 0
    stop_reason: str = ""
    tool_sequence_valid: bool = True
    execution_mode: str = "normal"


@dataclass
class RuleBasedResult:
    route_correct: bool
    required_tools_present: bool
    missing_required_tools: list[str]
    unexpected_tools_present: bool
    unexpected_tools: list[str]
    retrieval_mode_valid: bool
    required_point_hit_rate: float
    optional_point_hit_rate: float
    missing_required_points: list[str]
    content_pass: bool
    step_count: int
    step_limit_respected: bool
    stop_reason: str
    stop_reason_valid: bool
    tool_sequence_valid: bool
    tool_usage_valid: bool
    unnecessary_tool_calls: list[str]
    repeated_tool_calls: list[str]
    tool_dependency_valid: bool
    time_consistency_valid: bool


@dataclass
class JudgeResult:
    enabled: bool = False
    correctness_score: int | None = None
    completeness_score: int | None = None
    groundedness_score: int | None = None
    tool_usage_score: int | None = None
    report_quality_score: int | None = None
    passed: bool | None = None
    reason: str = ""


@dataclass
class EvalResult:
    case_id: str
    query: str
    category: str
    expected_route: str
    answer: str
    actual_tools: list[str]
    execution_mode: str
    user_id: str
    city: str
    rule_based: RuleBasedResult
    judge_based: JudgeResult


@dataclass
class RuleBasedSummary:
    total_cases: int
    route_correct_rate: float
    required_tool_pass_rate: float
    tool_sequence_valid_rate: float
    retrieval_mode_valid_rate: float
    content_pass_rate: float
    required_point_hit_rate_avg: float
    optional_point_hit_rate_avg: float
    report_generation_success_rate: float


@dataclass
class NormalSummary:
    total_cases: int
    content_pass_rate: float
    tool_usage_valid_rate: float
    unexpected_tool_rate: float
    required_point_hit_rate_avg: float
    judge_pass_rate: float | None = None


@dataclass
class ReportSummary:
    total_cases: int
    required_tools_present_rate: float
    tool_dependency_valid_rate: float
    time_consistency_valid_rate: float
    content_pass_rate: float
    avg_groundedness_score: float | None = None
    avg_report_quality_score: float | None = None


@dataclass
class JudgeBasedSummary:
    enabled: bool
    total_cases: int = 0
    judge_pass_rate: float | None = None
    avg_correctness_score: float | None = None
    avg_completeness_score: float | None = None
    avg_groundedness_score: float | None = None
    avg_tool_usage_score: float | None = None
    avg_report_quality_score: float | None = None


def parse_eval_points(raw_points: list[dict] | None, prefix: str) -> list[EvalPoint]:
    points: list[EvalPoint] = []
    for index, item in enumerate(raw_points or [], start=1):
        if not isinstance(item, dict):
            raise ValueError("point 配置必须为对象")
        label = str(item.get("label") or "").strip()
        if not label:
            raise ValueError("point 缺少 label")
        point_id = str(item.get("point_id") or f"{prefix}_{index:02d}").strip()
        aliases = [str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()]
        if label not in aliases:
            aliases.insert(0, label)
        points.append(EvalPoint(point_id=point_id, label=label, aliases=aliases))
    return points


def _normalize_year_month(value: str) -> str:
    normalized = value.strip()
    match = YEAR_MONTH_PATTERN.search(normalized)
    if match:
        return f"{match.group('year')}-{int(match.group('month')):02d}"
    if re.fullmatch(r"\d{4}-\d{2}", normalized):
        return normalized
    return ""


def _extract_year_months(text: str) -> list[str]:
    months: list[str] = []
    seen: set[str] = set()
    for match in YEAR_MONTH_RANGE_PATTERN.finditer(text or ""):
        start = f"{match.group('year')}-{int(match.group('start')):02d}"
        end = f"{match.group('year')}-{int(match.group('end')):02d}"
        for month in (start, end):
            if month not in seen:
                seen.add(month)
                months.append(month)
    for match in YEAR_MONTH_PATTERN.finditer(text or ""):
        month = f"{match.group('year')}-{int(match.group('month')):02d}"
        if month not in seen:
            seen.add(month)
            months.append(month)
    return months


def load_eval_cases(jsonl_path: str | Path | None = None) -> list[EvalCase]:
    dataset_path = Path(jsonl_path or DEFAULT_EVAL_DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测数据文件不存在: {dataset_path}")

    cases: list[EvalCase] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            normalized = line.strip()
            if not normalized:
                continue

            try:
                raw_case = json.loads(normalized)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第{index}条评测数据不是合法 JSON: {str(exc)}") from exc

            fieldnames = set(raw_case.keys())
            if not REQUIRED_CASE_FIELDS.issubset(fieldnames):
                raise ValueError(f"第{index}条评测数据缺少必要字段")

            category = str(raw_case.get("category") or "").strip()
            if category not in ALLOWED_CATEGORIES:
                raise ValueError(f"第{index}条评测数据的 category 非法: {category}")

            query = str(raw_case.get("query") or "").strip()
            if not query:
                raise ValueError(f"第{index}条评测数据缺少 query")

            expected_route = str(raw_case.get("expected_route") or "").strip()
            if expected_route not in ALLOWED_ROUTES:
                raise ValueError(f"第{index}条评测数据的 expected_route 非法: {expected_route}")

            expected_retrieval_mode = str(raw_case.get("expected_retrieval_mode") or "").strip()
            if expected_retrieval_mode not in ALLOWED_RETRIEVAL_MODES:
                raise ValueError(f"第{index}条评测数据的 expected_retrieval_mode 非法: {expected_retrieval_mode}")

            required_tools = [str(tool).strip() for tool in raw_case.get("required_tools", []) if str(tool).strip()]
            optional_tools = [str(tool).strip() for tool in raw_case.get("optional_tools", []) if str(tool).strip()]
            forbidden_tools = [str(tool).strip() for tool in raw_case.get("forbidden_tools", []) if str(tool).strip()]
            allowed_tools = ALLOWED_TOOLS_BY_ROUTE[expected_route]
            invalid_tools = [tool for tool in required_tools + optional_tools + forbidden_tools if tool not in allowed_tools]
            if invalid_tools:
                raise ValueError(f"第{index}条评测数据包含非法工具: {', '.join(invalid_tools)}")
            if set(required_tools) & set(optional_tools):
                raise ValueError(f"第{index}条评测数据的 required_tools 与 optional_tools 不应重复")

            allow_no_tool = bool(raw_case.get("allow_no_tool", category in {"faq", "troubleshooting"}))
            target_month = _normalize_year_month(str(raw_case.get("target_month") or ""))
            if not target_month and expected_route == "report":
                target_month = _normalize_year_month(query)
            allowed_trend_window = [
                normalized_month
                for raw_month in raw_case.get("allowed_trend_window", [])
                if (normalized_month := _normalize_year_month(str(raw_month)))
            ]

            cases.append(
                EvalCase(
                    case_id=str(raw_case.get("case_id") or f"case_{index:03d}").strip(),
                    query=query,
                    category=category,
                    expected_route=expected_route,
                    required_tools=required_tools,
                    optional_tools=optional_tools,
                    required_points=parse_eval_points(raw_case.get("required_points", []), prefix="required"),
                    optional_points=parse_eval_points(raw_case.get("optional_points", []), prefix="optional"),
                    expected_retrieval_mode=expected_retrieval_mode,
                    user_id=str(raw_case.get("user_id") or "1001").strip() or "1001",
                    city=str(raw_case.get("city") or "北京").strip() or "北京",
                    notes=str(raw_case.get("notes") or "").strip(),
                    forbidden_tools=forbidden_tools,
                    allow_no_tool=allow_no_tool,
                    target_month=target_month,
                    allowed_trend_window=allowed_trend_window,
                )
            )

    if not cases:
        raise ValueError("评测数据集不能为空")
    return cases


def validate_tool_sequence(tool_calls: list[str]) -> bool:
    if len(tool_calls) > 5:
        return False
    for previous, current in zip(tool_calls, tool_calls[1:]):
        if previous == current:
            return False
    if "get_weather" in tool_calls and "get_user_location" in tool_calls:
        return tool_calls.index("get_user_location") < tool_calls.index("get_weather")
    return True


def validate_report_tool_dependency(tool_calls: list[str], required_tools: list[str]) -> bool:
    if any(tool not in tool_calls for tool in required_tools):
        return False

    def is_before(before: str, after: str) -> bool:
        return before not in tool_calls or after not in tool_calls or tool_calls.index(before) < tool_calls.index(after)

    return (
        is_before("get_user_id", "fetch_external_data")
        and is_before("get_current_month", "fetch_external_data")
        and is_before("get_user_id", "fetch_external_history")
        and is_before("get_current_month", "fetch_external_history")
        and is_before("fetch_external_data", "fetch_external_history")
    )


def build_eval_trace(
    tool_calls: list[str] | None = None,
    retrieved_docs: list[dict[str, str]] | None = None,
    tool_evidence: list[dict[str, str]] | None = None,
    step_count: int = 0,
    stop_reason: str = "",
    execution_mode: str = "normal",
) -> EvalTrace:
    safe_tool_calls = list(tool_calls or [])
    safe_retrieved_docs = list(retrieved_docs or [])
    safe_tool_evidence = list(tool_evidence or [])
    return EvalTrace(
        tool_calls=safe_tool_calls,
        retrieved_docs=safe_retrieved_docs,
        tool_evidence=safe_tool_evidence,
        retrieval_hit=bool(safe_retrieved_docs),
        step_count=step_count,
        stop_reason=stop_reason,
        tool_sequence_valid=validate_tool_sequence(safe_tool_calls),
        execution_mode=execution_mode,
    )


def _match_point(answer: str, point: EvalPoint) -> bool:
    normalized_answer = (answer or "").lower()
    return any(alias.lower() in normalized_answer for alias in point.aliases)


def _evaluate_points(points: list[EvalPoint], answer: str) -> tuple[float, list[str]]:
    if not points:
        return 1.0, []
    missing = [point.label for point in points if not _match_point(answer, point)]
    hit_rate = (len(points) - len(missing)) / len(points)
    return hit_rate, missing


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _unexpected_tools(case: EvalCase, tool_calls: list[str]) -> list[str]:
    allowed_tools = ALLOWED_TOOLS_BY_ROUTE.get(case.expected_route, set())
    forbidden = set(case.forbidden_tools or [])
    return _deduplicate_preserve_order([tool for tool in tool_calls if tool not in allowed_tools or tool in forbidden])


def _unnecessary_tool_calls(case: EvalCase, tool_calls: list[str]) -> list[str]:
    recommended = set(case.required_tools) | set(case.optional_tools)
    forbidden = set(case.forbidden_tools or [])
    unnecessary = [tool for tool in tool_calls if tool not in recommended and tool not in forbidden]
    return _deduplicate_preserve_order(unnecessary)


def _repeated_tool_calls(tool_calls: list[str]) -> list[str]:
    duplicates: list[str] = []
    counts: dict[str, int] = {}
    for tool in tool_calls:
        counts[tool] = counts.get(tool, 0) + 1
        if counts[tool] == 2:
            duplicates.append(tool)
    return duplicates


def _retrieval_mode_valid(expected_mode: str, retrieval_hit: bool) -> bool:
    if expected_mode == "required":
        return retrieval_hit
    if expected_mode == "forbidden":
        return not retrieval_hit
    return True


def _time_consistency_valid(case: EvalCase, answer: str) -> bool:
    if case.category != "report_generation":
        return True

    target_month = case.target_month or _normalize_year_month(case.query)
    if not target_month:
        return True

    mentioned_months = _extract_year_months(_extract_report_analysis_scope(answer))
    if not mentioned_months:
        return False
    if target_month not in mentioned_months:
        return False

    allowed_trend_window = set(case.allowed_trend_window or [])
    if allowed_trend_window:
        allowed = allowed_trend_window | {target_month}
        return all(month in allowed for month in mentioned_months)

    return all(month <= target_month for month in mentioned_months)


def _extract_report_analysis_scope(answer: str) -> str:
    normalized = answer or ""
    lines = normalized.splitlines(keepends=True)
    keywords = ("建议", "提升方向", "长期使用", "保养与使用")
    offset = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0:
            offset += len(line)
            continue
        if stripped.startswith("#") and any(keyword in stripped for keyword in keywords):
            return normalized[:offset]
        offset += len(line)
    return normalized


def _build_rule_based_result(case: EvalCase, trace: EvalTrace, answer: str) -> RuleBasedResult:
    route_correct = trace.execution_mode == case.expected_route
    missing_required_tools = [tool for tool in case.required_tools if tool not in trace.tool_calls]
    required_tools_present = not missing_required_tools
    unexpected_tools = _unexpected_tools(case, trace.tool_calls)
    unexpected_tools_present = bool(unexpected_tools)
    unnecessary_tool_calls = _unnecessary_tool_calls(case, trace.tool_calls)
    repeated_tool_calls = _repeated_tool_calls(trace.tool_calls)
    retrieval_mode_valid = _retrieval_mode_valid(case.expected_retrieval_mode, trace.retrieval_hit)
    required_point_hit_rate, missing_required_points = _evaluate_points(case.required_points, answer)
    optional_point_hit_rate, _ = _evaluate_points(case.optional_points, answer)
    content_pass = required_point_hit_rate == 1.0
    step_limit_respected = trace.step_count <= 5
    stop_reason_valid = trace.stop_reason in VALID_STOP_REASONS
    tool_sequence_valid = trace.tool_sequence_valid
    tool_dependency_valid = True
    time_consistency_valid = True

    if case.expected_route == "report":
        tool_dependency_valid = validate_report_tool_dependency(trace.tool_calls, case.required_tools)
        time_consistency_valid = _time_consistency_valid(case, answer)

    no_tool_invalid = not trace.tool_calls and not case.allow_no_tool
    tool_usage_valid = (
        required_tools_present
        and not unexpected_tools_present
        and not repeated_tool_calls
        and not unnecessary_tool_calls
        and not no_tool_invalid
    )

    return RuleBasedResult(
        route_correct=route_correct,
        required_tools_present=required_tools_present,
        missing_required_tools=missing_required_tools,
        unexpected_tools_present=unexpected_tools_present,
        unexpected_tools=unexpected_tools,
        retrieval_mode_valid=retrieval_mode_valid,
        required_point_hit_rate=required_point_hit_rate,
        optional_point_hit_rate=optional_point_hit_rate,
        missing_required_points=missing_required_points,
        content_pass=content_pass,
        step_count=trace.step_count,
        step_limit_respected=step_limit_respected,
        stop_reason=trace.stop_reason,
        stop_reason_valid=stop_reason_valid,
        tool_sequence_valid=tool_sequence_valid,
        tool_usage_valid=tool_usage_valid,
        unnecessary_tool_calls=unnecessary_tool_calls,
        repeated_tool_calls=repeated_tool_calls,
        tool_dependency_valid=tool_dependency_valid,
        time_consistency_valid=time_consistency_valid,
    )


def evaluate_case(
    case: EvalCase,
    executor: Callable[[EvalCase], tuple[str, EvalTrace]],
    judge: Callable[[EvalCase, str, EvalTrace], JudgeResult] | None = None,
) -> EvalResult:
    answer, trace = executor(case)
    rule_based = _build_rule_based_result(case, trace, answer)
    if judge:
        try:
            judge_based = judge(case, answer, trace)
        except Exception as exc:
            judge_based = JudgeResult(
                enabled=True,
                passed=False,
                reason=f"Judge 执行失败: {str(exc)}",
            )
    else:
        judge_based = JudgeResult(enabled=False)
    return EvalResult(
        case_id=case.case_id,
        query=case.query,
        category=case.category,
        expected_route=case.expected_route,
        answer=answer or "",
        actual_tools=list(trace.tool_calls),
        execution_mode=trace.execution_mode,
        user_id=case.user_id,
        city=case.city,
        rule_based=rule_based,
        judge_based=judge_based,
    )


def summarize_rule_results(results: Iterable[EvalResult]) -> RuleBasedSummary:
    result_list = list(results)
    total_cases = len(result_list)
    if total_cases == 0:
        raise ValueError("评测结果不能为空")

    report_cases = [result for result in result_list if result.category == "report_generation"]
    return RuleBasedSummary(
        total_cases=total_cases,
        route_correct_rate=sum(result.rule_based.route_correct for result in result_list) / total_cases,
        required_tool_pass_rate=sum(result.rule_based.required_tools_present for result in result_list) / total_cases,
        tool_sequence_valid_rate=sum(result.rule_based.tool_sequence_valid for result in result_list) / total_cases,
        retrieval_mode_valid_rate=sum(result.rule_based.retrieval_mode_valid for result in result_list) / total_cases,
        content_pass_rate=sum(result.rule_based.content_pass for result in result_list) / total_cases,
        required_point_hit_rate_avg=sum(result.rule_based.required_point_hit_rate for result in result_list) / total_cases,
        optional_point_hit_rate_avg=sum(result.rule_based.optional_point_hit_rate for result in result_list) / total_cases,
        report_generation_success_rate=(
            sum(
                result.rule_based.route_correct
                and result.rule_based.required_tools_present
                and result.rule_based.tool_dependency_valid
                and result.rule_based.time_consistency_valid
                and result.rule_based.content_pass
                and bool(result.answer.strip())
                for result in report_cases
            ) / len(report_cases)
            if report_cases
            else 0.0
        ),
    )


def summarize_normal_results(results: Iterable[EvalResult]) -> NormalSummary:
    result_list = [result for result in results if result.category in NORMAL_CATEGORIES]
    if not result_list:
        return NormalSummary(
            total_cases=0,
            content_pass_rate=0.0,
            tool_usage_valid_rate=0.0,
            unexpected_tool_rate=0.0,
            required_point_hit_rate_avg=0.0,
            judge_pass_rate=None,
        )

    judge_enabled_cases = [result for result in result_list if result.judge_based.enabled]
    judge_pass_rate = None
    if judge_enabled_cases:
        judge_pass_rate = sum(bool(result.judge_based.passed) for result in judge_enabled_cases) / len(judge_enabled_cases)

    return NormalSummary(
        total_cases=len(result_list),
        content_pass_rate=sum(result.rule_based.content_pass for result in result_list) / len(result_list),
        tool_usage_valid_rate=sum(result.rule_based.tool_usage_valid for result in result_list) / len(result_list),
        unexpected_tool_rate=sum(result.rule_based.unexpected_tools_present for result in result_list) / len(result_list),
        required_point_hit_rate_avg=sum(result.rule_based.required_point_hit_rate for result in result_list) / len(result_list),
        judge_pass_rate=judge_pass_rate,
    )


def summarize_judge_results(results: Iterable[EvalResult]) -> JudgeBasedSummary:
    result_list = [result for result in results if result.judge_based.enabled]
    if not result_list:
        return JudgeBasedSummary(enabled=False)

    def _avg(values: list[int | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    report_scores = [result.judge_based.report_quality_score for result in result_list if result.category == "report_generation"]
    return JudgeBasedSummary(
        enabled=True,
        total_cases=len(result_list),
        judge_pass_rate=sum(bool(result.judge_based.passed) for result in result_list) / len(result_list),
        avg_correctness_score=_avg([result.judge_based.correctness_score for result in result_list]),
        avg_completeness_score=_avg([result.judge_based.completeness_score for result in result_list]),
        avg_groundedness_score=_avg([result.judge_based.groundedness_score for result in result_list]),
        avg_tool_usage_score=_avg([result.judge_based.tool_usage_score for result in result_list]),
        avg_report_quality_score=_avg(report_scores),
    )


def summarize_report_results(results: Iterable[EvalResult]) -> ReportSummary:
    result_list = [result for result in results if result.category == "report_generation"]
    if not result_list:
        return ReportSummary(
            total_cases=0,
            required_tools_present_rate=0.0,
            tool_dependency_valid_rate=0.0,
            time_consistency_valid_rate=0.0,
            content_pass_rate=0.0,
            avg_groundedness_score=None,
            avg_report_quality_score=None,
        )

    judge_enabled_cases = [result for result in result_list if result.judge_based.enabled]

    def _avg(values: list[int | None]) -> float | None:
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    return ReportSummary(
        total_cases=len(result_list),
        required_tools_present_rate=sum(result.rule_based.required_tools_present for result in result_list) / len(result_list),
        tool_dependency_valid_rate=sum(result.rule_based.tool_dependency_valid for result in result_list) / len(result_list),
        time_consistency_valid_rate=sum(result.rule_based.time_consistency_valid for result in result_list) / len(result_list),
        content_pass_rate=sum(result.rule_based.content_pass for result in result_list) / len(result_list),
        avg_groundedness_score=_avg([result.judge_based.groundedness_score for result in judge_enabled_cases]),
        avg_report_quality_score=_avg([result.judge_based.report_quality_score for result in judge_enabled_cases]),
    )


def run_evaluation(
    cases: list[EvalCase],
    executor: Callable[[EvalCase], tuple[str, EvalTrace]],
    judge: Callable[[EvalCase, str, EvalTrace], JudgeResult] | None = None,
) -> tuple[list[EvalResult], RuleBasedSummary, JudgeBasedSummary, NormalSummary, ReportSummary]:
    results = [evaluate_case(case, executor, judge=judge) for case in cases]
    return (
        results,
        summarize_rule_results(results),
        summarize_judge_results(results),
        summarize_normal_results(results),
        summarize_report_results(results),
    )


def _serialize_result(result: EvalResult) -> dict[str, object]:
    return _round_floats(asdict(result))


def _round_floats(value, digits: int = 2):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_floats(item, digits=digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item, digits=digits) for item in value]
    return value


def write_evaluation_outputs(
    results: list[EvalResult],
    rule_summary: RuleBasedSummary,
    judge_summary: JudgeBasedSummary,
    normal_summary: NormalSummary,
    report_summary: ReportSummary,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir or DEFAULT_RESULTS_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"eval_results_{timestamp}.json"
    csv_path = output_path / f"eval_results_{timestamp}.csv"

    json_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rule_based_summary": _round_floats(asdict(rule_summary)),
        "judge_based_summary": _round_floats(asdict(judge_summary)),
        "normal_summary": _round_floats(asdict(normal_summary)),
        "report_summary": _round_floats(asdict(report_summary)),
        "results": [_serialize_result(result) for result in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "category",
            "query",
            "expected_route",
            "execution_mode",
            "route_correct",
            "required_tools_present",
            "missing_required_tools",
            "unexpected_tools",
            "retrieval_mode_valid",
            "required_point_hit_rate",
            "optional_point_hit_rate",
            "missing_required_points",
            "content_pass",
            "tool_sequence_valid",
            "tool_usage_valid",
            "unnecessary_tool_calls",
            "repeated_tool_calls",
            "tool_dependency_valid",
            "time_consistency_valid",
            "step_count",
            "stop_reason",
            "judge_enabled",
            "judge_passed",
            "correctness_score",
            "completeness_score",
            "groundedness_score",
            "tool_usage_score",
            "report_quality_score",
            "answer",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "category": result.category,
                    "query": result.query,
                    "expected_route": result.expected_route,
                    "execution_mode": result.execution_mode,
                    "route_correct": result.rule_based.route_correct,
                    "required_tools_present": result.rule_based.required_tools_present,
                    "missing_required_tools": "|".join(result.rule_based.missing_required_tools),
                    "unexpected_tools": "|".join(result.rule_based.unexpected_tools),
                    "retrieval_mode_valid": result.rule_based.retrieval_mode_valid,
                    "required_point_hit_rate": round(result.rule_based.required_point_hit_rate, 2),
                    "optional_point_hit_rate": round(result.rule_based.optional_point_hit_rate, 2),
                    "missing_required_points": "|".join(result.rule_based.missing_required_points),
                    "content_pass": result.rule_based.content_pass,
                    "tool_sequence_valid": result.rule_based.tool_sequence_valid,
                    "tool_usage_valid": result.rule_based.tool_usage_valid,
                    "unnecessary_tool_calls": "|".join(result.rule_based.unnecessary_tool_calls),
                    "repeated_tool_calls": "|".join(result.rule_based.repeated_tool_calls),
                    "tool_dependency_valid": result.rule_based.tool_dependency_valid,
                    "time_consistency_valid": result.rule_based.time_consistency_valid,
                    "step_count": result.rule_based.step_count,
                    "stop_reason": result.rule_based.stop_reason,
                    "judge_enabled": result.judge_based.enabled,
                    "judge_passed": result.judge_based.passed,
                    "correctness_score": result.judge_based.correctness_score,
                    "completeness_score": result.judge_based.completeness_score,
                    "groundedness_score": result.judge_based.groundedness_score,
                    "tool_usage_score": result.judge_based.tool_usage_score,
                    "report_quality_score": result.judge_based.report_quality_score,
                    "answer": result.answer,
                }
            )

    return json_path, csv_path
