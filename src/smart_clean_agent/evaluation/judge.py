import json
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from dataclasses import asdict

from langchain_core.messages import HumanMessage, SystemMessage

from smart_clean_agent.evaluation.service import EvalCase, EvalTrace, JudgeResult
from smart_clean_agent.model.factory import create_chat_model

JUDGE_SYSTEM_PROMPT = """你是一个严谨的离线评测裁判。请根据用户问题、答案、期望要点、工具链路和证据摘要，对答案质量进行评分。

只输出一个 JSON 对象，不要输出任何额外文字。字段必须包含：
- correctness_score: 0-5 整数
- completeness_score: 0-5 整数
- groundedness_score: 0-5 整数
- tool_usage_score: 0-5 整数
- report_quality_score: 0-5 整数，非报告类填 0
- passed: 布尔值
- reason: 简短中文原因
"""


def _extract_json_payload(content: str) -> dict:
    normalized = (content or "").strip()
    if normalized.startswith("```"):
        normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
        normalized = re.sub(r"\s*```$", "", normalized)
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Judge 输出中未找到 JSON 对象")
    return json.loads(normalized[start : end + 1])


def _validate_score(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须为 0-5 的整数")

    numeric_value: Decimal | None = None
    if isinstance(value, int):
        numeric_value = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} 必须为 0-5 的整数")
        numeric_value = Decimal(str(value))
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} 必须为 0-5 的整数")
        matched = re.search(r"-?\d+(?:\.\d+)?", normalized)
        if not matched:
            raise ValueError(f"{field_name} 必须为 0-5 的整数")
        try:
            numeric_value = Decimal(matched.group(0))
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} 必须为 0-5 的整数") from exc
    else:
        raise ValueError(f"{field_name} 必须为 0-5 的整数")

    if numeric_value < 0 or numeric_value > 5:
        raise ValueError(f"{field_name} 必须为 0-5 的整数")
    return int(numeric_value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _normalize_passed(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "通过", "是"}:
            return True
        if normalized in {"false", "0", "no", "n", "不通过", "否"}:
            return False
    return bool(value)


class JudgeEvaluator:
    def __init__(self, model_name: str | None = None):
        self.model = create_chat_model(model_name=model_name, role="judge")

    def evaluate(self, case: EvalCase, answer: str, trace: EvalTrace) -> JudgeResult:
        retrieved_docs_summary = [
            {
                "source": item.get("source", ""),
                "snippet": item.get("snippet", ""),
            }
            for item in trace.retrieved_docs[:3]
        ]
        human_payload = {
            "query": case.query,
            "category": case.category,
            "answer": answer,
            "required_points": [asdict(point) for point in case.required_points],
            "optional_points": [asdict(point) for point in case.optional_points],
            "actual_tools": trace.tool_calls,
            "execution_mode": trace.execution_mode,
            "tool_observations_summary": trace.tool_evidence[:5],
            "retrieved_docs_summary": retrieved_docs_summary,
        }
        response = self.model.invoke(
            [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(human_payload, ensure_ascii=False, indent=2)),
            ]
        )
        payload = _extract_json_payload(getattr(response, "content", ""))
        return JudgeResult(
            enabled=True,
            correctness_score=_validate_score(payload.get("correctness_score"), "correctness_score"),
            completeness_score=_validate_score(payload.get("completeness_score"), "completeness_score"),
            groundedness_score=_validate_score(payload.get("groundedness_score"), "groundedness_score"),
            tool_usage_score=_validate_score(payload.get("tool_usage_score"), "tool_usage_score"),
            report_quality_score=_validate_score(payload.get("report_quality_score"), "report_quality_score"),
            passed=_normalize_passed(payload.get("passed")),
            reason=str(payload.get("reason") or "").strip(),
        )
