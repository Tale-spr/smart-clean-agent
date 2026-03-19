import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_EVAL_DATASET = Path("data/eval/eval_cases.csv")
DEFAULT_RESULTS_DIR = Path("data/eval/results")
REQUIRED_CASE_FIELDS = {"query", "category", "expected_keywords", "expected_tool", "expected_retrieval_hit"}
ALLOWED_CATEGORIES = {"faq", "troubleshooting", "environment_fit", "report_generation"}
TRUE_VALUES = {"true", "1", "yes", "y", "是"}
FALSE_VALUES = {"false", "0", "no", "n", "否", ""}


@dataclass
class EvalCase:
    case_id: str
    query: str
    category: str
    expected_keywords: list[str]
    expected_tool: str
    expected_retrieval_hit: bool
    user_id: str = "1001"
    city: str = "北京"


@dataclass
class EvalTrace:
    tool_calls: list[str]
    retrieved_docs: list[dict[str, str]]
    retrieval_hit: bool


@dataclass
class EvalResult:
    case_id: str
    query: str
    category: str
    answer: str
    answer_keyword_hit: bool
    missing_keywords: list[str]
    expected_tool: str
    actual_tools: list[str]
    tool_call_checked: bool
    tool_call_success: bool
    expected_retrieval_hit: bool
    actual_retrieval_hit: bool
    retrieval_expectation_met: bool
    report_generation_success: bool
    user_id: str
    city: str


@dataclass
class EvalSummary:
    total_cases: int
    answer_keyword_hit_rate: float
    tool_call_success_rate: float
    retrieval_hit_rate: float
    report_generation_success_rate: float
    cases_with_expected_tool: int
    cases_with_expected_retrieval: int
    report_generation_cases: int



def parse_bool_flag(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"非法布尔值: {value}")



def parse_keywords(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]



def load_eval_cases(csv_path: str | Path | None = None) -> list[EvalCase]:
    dataset_path = Path(csv_path or DEFAULT_EVAL_DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(f"评测数据文件不存在: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not REQUIRED_CASE_FIELDS.issubset(fieldnames):
            raise ValueError("评测数据文件缺少必要字段")

        cases: list[EvalCase] = []
        for index, row in enumerate(reader, start=1):
            category = (row.get("category") or "").strip()
            if category not in ALLOWED_CATEGORIES:
                raise ValueError(f"第{index}条评测数据的category非法: {category}")

            query = (row.get("query") or "").strip()
            if not query:
                raise ValueError(f"第{index}条评测数据缺少query")

            case_id = (row.get("case_id") or f"case_{index:03d}").strip()
            user_id = (row.get("user_id") or "1001").strip() or "1001"
            city = (row.get("city") or "北京").strip() or "北京"

            cases.append(
                EvalCase(
                    case_id=case_id,
                    query=query,
                    category=category,
                    expected_keywords=parse_keywords(row.get("expected_keywords", "")),
                    expected_tool=(row.get("expected_tool") or "").strip(),
                    expected_retrieval_hit=parse_bool_flag(row.get("expected_retrieval_hit", "")),
                    user_id=user_id,
                    city=city,
                )
            )

    if not cases:
        raise ValueError("评测数据集不能为空")

    return cases



def build_eval_trace(tool_calls: list[str] | None = None, retrieved_docs: list[dict[str, str]] | None = None) -> EvalTrace:
    safe_tool_calls = list(tool_calls or [])
    safe_retrieved_docs = list(retrieved_docs or [])
    return EvalTrace(
        tool_calls=safe_tool_calls,
        retrieved_docs=safe_retrieved_docs,
        retrieval_hit=bool(safe_retrieved_docs),
    )



def evaluate_case(
    case: EvalCase,
    executor: Callable[[EvalCase], tuple[str, EvalTrace]],
) -> EvalResult:
    answer, trace = executor(case)
    normalized_answer = answer or ""
    answer_lower = normalized_answer.lower()
    missing_keywords = [keyword for keyword in case.expected_keywords if keyword.lower() not in answer_lower]
    answer_keyword_hit = not missing_keywords
    tool_call_checked = bool(case.expected_tool)
    tool_call_success = case.expected_tool in trace.tool_calls if tool_call_checked else True
    retrieval_expectation_met = trace.retrieval_hit == case.expected_retrieval_hit
    report_generation_success = (
        case.category == "report_generation"
        and bool(normalized_answer.strip())
        and answer_keyword_hit
        and tool_call_success
    )

    return EvalResult(
        case_id=case.case_id,
        query=case.query,
        category=case.category,
        answer=normalized_answer,
        answer_keyword_hit=answer_keyword_hit,
        missing_keywords=missing_keywords,
        expected_tool=case.expected_tool,
        actual_tools=trace.tool_calls,
        tool_call_checked=tool_call_checked,
        tool_call_success=tool_call_success,
        expected_retrieval_hit=case.expected_retrieval_hit,
        actual_retrieval_hit=trace.retrieval_hit,
        retrieval_expectation_met=retrieval_expectation_met,
        report_generation_success=report_generation_success,
        user_id=case.user_id,
        city=case.city,
    )



def summarize_results(results: Iterable[EvalResult]) -> EvalSummary:
    result_list = list(results)
    total_cases = len(result_list)
    if total_cases == 0:
        raise ValueError("评测结果不能为空")

    cases_with_expected_tool = sum(1 for result in result_list if result.tool_call_checked)
    cases_with_expected_retrieval = sum(1 for result in result_list if result.expected_retrieval_hit)
    report_generation_cases = sum(1 for result in result_list if result.category == "report_generation")

    answer_keyword_hit_rate = sum(1 for result in result_list if result.answer_keyword_hit) / total_cases
    tool_call_success_rate = (
        sum(1 for result in result_list if result.tool_call_checked and result.tool_call_success) / cases_with_expected_tool
        if cases_with_expected_tool
        else 0.0
    )
    retrieval_hit_rate = (
        sum(1 for result in result_list if result.expected_retrieval_hit and result.actual_retrieval_hit) / cases_with_expected_retrieval
        if cases_with_expected_retrieval
        else 0.0
    )
    report_generation_success_rate = (
        sum(1 for result in result_list if result.report_generation_success) / report_generation_cases
        if report_generation_cases
        else 0.0
    )

    return EvalSummary(
        total_cases=total_cases,
        answer_keyword_hit_rate=answer_keyword_hit_rate,
        tool_call_success_rate=tool_call_success_rate,
        retrieval_hit_rate=retrieval_hit_rate,
        report_generation_success_rate=report_generation_success_rate,
        cases_with_expected_tool=cases_with_expected_tool,
        cases_with_expected_retrieval=cases_with_expected_retrieval,
        report_generation_cases=report_generation_cases,
    )



def run_evaluation(
    cases: list[EvalCase],
    executor: Callable[[EvalCase], tuple[str, EvalTrace]],
) -> tuple[list[EvalResult], EvalSummary]:
    results = [evaluate_case(case, executor) for case in cases]
    return results, summarize_results(results)



def _serialize_result(result: EvalResult) -> dict[str, object]:
    payload = asdict(result)
    payload["actual_tools"] = list(result.actual_tools)
    payload["missing_keywords"] = list(result.missing_keywords)
    return payload



def write_evaluation_outputs(
    results: list[EvalResult],
    summary: EvalSummary,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_dir or DEFAULT_RESULTS_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"eval_results_{timestamp}.json"
    csv_path = output_path / f"eval_results_{timestamp}.csv"

    json_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": asdict(summary),
        "results": [_serialize_result(result) for result in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "category",
            "query",
            "answer_keyword_hit",
            "missing_keywords",
            "expected_tool",
            "actual_tools",
            "tool_call_checked",
            "tool_call_success",
            "expected_retrieval_hit",
            "actual_retrieval_hit",
            "retrieval_expectation_met",
            "report_generation_success",
            "user_id",
            "city",
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
                    "answer_keyword_hit": result.answer_keyword_hit,
                    "missing_keywords": "|".join(result.missing_keywords),
                    "expected_tool": result.expected_tool,
                    "actual_tools": "|".join(result.actual_tools),
                    "tool_call_checked": result.tool_call_checked,
                    "tool_call_success": result.tool_call_success,
                    "expected_retrieval_hit": result.expected_retrieval_hit,
                    "actual_retrieval_hit": result.actual_retrieval_hit,
                    "retrieval_expectation_met": result.retrieval_expectation_met,
                    "report_generation_success": result.report_generation_success,
                    "user_id": result.user_id,
                    "city": result.city,
                    "answer": result.answer,
                }
            )

    return json_path, csv_path

