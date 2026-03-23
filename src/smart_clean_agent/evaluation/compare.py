import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比两个 evaluation 结果文件")
    parser.add_argument("before", help="旧结果文件路径")
    parser.add_argument("after", help="新结果文件路径")
    parser.add_argument("--output", help="可选的输出 JSON 路径")
    return parser.parse_args(argv)


def _load_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _diff_summary(before: dict, after: dict) -> dict:
    diff: dict[str, float | None] = {}
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key)
        after_value = after.get(key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            diff[key] = after_value - before_value
        elif before_value != after_value:
            diff[key] = None
    return diff


def _rule_score(result: dict) -> float:
    rule = result.get("rule_based", {})
    return (
        float(bool(rule.get("route_correct")))
        + float(bool(rule.get("required_tools_present")))
        + float(bool(rule.get("tool_sequence_valid")))
        + float(bool(rule.get("retrieval_mode_valid")))
        + float(rule.get("required_point_hit_rate", 0.0))
    )


def _judge_score(result: dict) -> float:
    judge = result.get("judge_based", {})
    if not judge.get("enabled"):
        return 0.0
    return float(judge.get("correctness_score") or 0) + float(judge.get("completeness_score") or 0)


def compare_results(before_payload: dict, after_payload: dict) -> dict:
    before_results = {item["case_id"]: item for item in before_payload.get("results", [])}
    after_results = {item["case_id"]: item for item in after_payload.get("results", [])}
    improved_cases: list[dict[str, object]] = []
    regressed_cases: list[dict[str, object]] = []

    for case_id in sorted(set(before_results) & set(after_results)):
        before_score = _rule_score(before_results[case_id]) + _judge_score(before_results[case_id])
        after_score = _rule_score(after_results[case_id]) + _judge_score(after_results[case_id])
        if after_score > before_score:
            improved_cases.append({"case_id": case_id, "before_score": before_score, "after_score": after_score})
        elif after_score < before_score:
            regressed_cases.append({"case_id": case_id, "before_score": before_score, "after_score": after_score})

    return {
        "before_file": before_payload.get("generated_at", ""),
        "after_file": after_payload.get("generated_at", ""),
        "rule_based_summary_diff": _diff_summary(
            before_payload.get("rule_based_summary", {}),
            after_payload.get("rule_based_summary", {}),
        ),
        "judge_based_summary_diff": _diff_summary(
            before_payload.get("judge_based_summary", {}),
            after_payload.get("judge_based_summary", {}),
        ),
        "improved_cases": improved_cases,
        "regressed_cases": regressed_cases,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        comparison = compare_results(_load_payload(args.before), _load_payload(args.after))
        serialized = json.dumps(comparison, ensure_ascii=False, indent=2)
        print(serialized)
        if args.output:
            Path(args.output).write_text(serialized, encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"评测对比失败: {str(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
