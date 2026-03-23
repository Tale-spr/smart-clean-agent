import argparse
import csv
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将旧版 CSV 评测集转换为 JSONL")
    parser.add_argument("--input", dest="input_path", default="data/eval/eval_cases.csv")
    parser.add_argument("--output", dest="output_path", default="data/eval/eval_cases.jsonl")
    return parser.parse_args(argv)


def _point(label: str, index: int) -> dict:
    normalized = (label or "").strip()
    return {
        "point_id": f"required_{index:02d}",
        "label": normalized,
        "aliases": [normalized],
    }


def convert_csv_to_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    source = Path(input_path)
    target = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"旧版 CSV 数据集不存在: {source}")

    rows: list[str] = []
    with open(source, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (row.get("category") or "").strip()
            query = (row.get("query") or "").strip()
            expected_tool = (row.get("expected_tool") or "").strip()
            required_points = [
                _point(label, index)
                for index, label in enumerate(
                    [item.strip() for item in (row.get("expected_keywords") or "").split("|") if item.strip()],
                    start=1,
                )
            ]
            record = {
                "case_id": (row.get("case_id") or "").strip(),
                "query": query,
                "category": category,
                "user_id": (row.get("user_id") or "1001").strip() or "1001",
                "city": (row.get("city") or "北京").strip() or "北京",
                "expected_route": "report" if category == "report_generation" else "normal",
                "required_tools": [expected_tool] if expected_tool else [],
                "optional_tools": [],
                "required_points": required_points,
                "optional_points": [],
                "expected_retrieval_mode": "required" if (row.get("expected_retrieval_hit") or "").strip().lower() in {"true", "1", "yes", "y", "是"} else "forbidden",
                "notes": "由旧版 CSV 自动迁移生成",
            }
            rows.append(json.dumps(record, ensure_ascii=False))

    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        count = convert_csv_to_jsonl(args.input_path, args.output_path)
        print(f"已转换 {count} 条评测数据到 {args.output_path}")
        return 0
    except Exception as exc:
        print(f"迁移评测数据失败: {str(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
