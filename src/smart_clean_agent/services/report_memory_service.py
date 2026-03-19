import csv
import json
import re
from datetime import datetime
from pathlib import Path

from smart_clean_agent.utils.config_handler import agent_conf
from smart_clean_agent.utils.path_tool import get_abs_path

REPORT_MEMORY_DEFAULT_DIR = "data/memory/reports"
DEFAULT_TREND_MONTH_WINDOW = 3
REPORT_RECORD_FIELDS = {"用户ID", "特征", "清洁效率", "耗材", "对比", "时间"}
ISSUE_LABELS = {
    "漏扫": "漏扫",
    "失败": "失败",
    "缠绕": "缠绕",
    "低于": "表现偏低",
    "未使用": "功能未使用",
    "磨损": "磨损",
    "衰减": "电池衰减",
    "泼洒": "污渍/猫砂处理",
    "故障": "故障",
    "回充": "回充",
}



def get_report_memory_dir(base_dir: str | None = None) -> Path:
    memory_dir = base_dir or get_abs_path(agent_conf.get("report_memory_dir", REPORT_MEMORY_DEFAULT_DIR))
    path = Path(memory_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path



def get_report_memory_path(user_id: str, base_dir: str | None = None) -> Path:
    return get_report_memory_dir(base_dir) / f"{user_id}.json"



def build_empty_report_memory(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "monthly_records": [],
        "trend_summary": "",
        "last_updated_at": "",
    }



def load_report_memory(user_id: str, base_dir: str | None = None) -> dict:
    memory_path = get_report_memory_path(user_id, base_dir)
    if not memory_path.exists():
        return build_empty_report_memory(user_id)

    with open(memory_path, "r", encoding="utf-8") as f:
        memory = json.load(f)

    payload = build_empty_report_memory(user_id)
    payload.update(memory)
    payload["user_id"] = user_id
    return payload



def save_report_memory(user_id: str, memory: dict, base_dir: str | None = None) -> dict:
    memory_path = get_report_memory_path(user_id, base_dir)
    payload = build_empty_report_memory(user_id)
    payload.update(memory)
    payload["user_id"] = user_id
    if not payload.get("last_updated_at"):
        payload["last_updated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload



def refresh_report_memory(
    user_id: str,
    months: int | None = None,
    csv_path: str | None = None,
    base_dir: str | None = None,
) -> dict:
    window = months or int(agent_conf.get("report_memory_month_window", DEFAULT_TREND_MONTH_WINDOW))
    user_records = load_user_monthly_records(user_id, csv_path)
    recent_records = user_records[-window:] if len(user_records) > window else user_records
    trend_summary = build_external_history_summary(recent_records)
    memory = {
        "user_id": user_id,
        "monthly_records": recent_records,
        "trend_summary": trend_summary,
        "last_updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return save_report_memory(user_id, memory, base_dir)



def build_report_memory_summary(memory: dict) -> str:
    return (memory.get("trend_summary") or "").strip()



def load_user_monthly_records(user_id: str, csv_path: str | None = None) -> list[dict[str, str]]:
    records = load_external_records(csv_path)
    user_records = []
    for row in records:
        if row["用户ID"] != user_id:
            continue
        user_records.append(
            {
                "month": row["时间"],
                "feature": row["特征"],
                "efficiency": row["清洁效率"],
                "consumables": row["耗材"],
                "comparison": row["对比"],
            }
        )

    return sorted(user_records, key=lambda item: item["month"])



def load_external_records(csv_path: str | None = None) -> list[dict[str, str]]:
    record_path = csv_path or get_abs_path(agent_conf["external_data_path"])
    path = Path(record_path)
    if not path.exists():
        raise FileNotFoundError(f"外部数据文件{path}不存在")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not REPORT_RECORD_FIELDS.issubset(fieldnames):
            raise ValueError("外部数据文件缺少必要字段")
        return [dict(row) for row in reader]



def build_external_history_summary(records: list[dict[str, str]]) -> str:
    if not records:
        return ""

    months_text = "、".join(record["month"] for record in records)
    efficiency_summary = summarize_efficiency_trend(records)
    consumables_summary = summarize_consumables_trend(records)
    issue_summary = summarize_issue_trend(records)
    suggestion_summary = summarize_trend_suggestion(records)

    return (
        f"覆盖月份: {months_text}\n"
        f"清洁效率趋势: {efficiency_summary}\n"
        f"耗材趋势: {consumables_summary}\n"
        f"问题变化趋势: {issue_summary}\n"
        f"总结建议: {suggestion_summary}"
    )



def summarize_efficiency_trend(records: list[dict[str, str]]) -> str:
    monthly_highlights = [f"{record['month']}:{first_content_line(record['efficiency'])}" for record in records]
    values = [extract_first_percentage(record["efficiency"]) for record in records]
    valid_values = [value for value in values if value is not None]
    if len(valid_values) >= 2:
        start = valid_values[0]
        end = valid_values[-1]
        if end > start:
            trend = "整体回升"
        elif end < start:
            trend = "整体下降"
        else:
            trend = "基本稳定"
        return f"{trend}；" + "；".join(monthly_highlights)
    return "；".join(monthly_highlights)



def summarize_consumables_trend(records: list[dict[str, str]]) -> str:
    monthly_highlights = [f"{record['month']}:{first_content_line(record['consumables'])}" for record in records]
    scores = [score_consumables(record["consumables"]) for record in records]
    if len(scores) >= 2:
        if scores[-1] > scores[0]:
            trend = "耗材压力上升"
        elif scores[-1] < scores[0]:
            trend = "耗材压力缓解"
        else:
            trend = "耗材状态基本稳定"
        return f"{trend}；" + "；".join(monthly_highlights)
    return "；".join(monthly_highlights)



def summarize_issue_trend(records: list[dict[str, str]]) -> str:
    monthly_issue_counts = [count_issue_indicators(record) for record in records]
    issue_labels = []
    for record in records:
        issue_labels.extend(extract_issue_labels(record))
    labels = "、".join(_dedupe_keep_order(issue_labels)[:3]) or "暂无明显高频问题"

    if len(monthly_issue_counts) >= 2:
        if monthly_issue_counts[-1] > monthly_issue_counts[0]:
            trend = "问题数量增加"
        elif monthly_issue_counts[-1] < monthly_issue_counts[0]:
            trend = "问题数量减少"
        else:
            trend = "问题数量基本稳定"
        return f"{trend}；主要关注: {labels}"
    return f"主要关注: {labels}"



def summarize_trend_suggestion(records: list[dict[str, str]]) -> str:
    latest_record = records[-1]
    issues = extract_issue_labels(latest_record)
    suggestions: list[str] = []

    if score_consumables(latest_record["consumables"]) >= 3:
        suggestions.append("优先检查并更换关键耗材，避免影响后续清洁效果")
    if "缠绕" in issues:
        suggestions.append("重点清理主刷和边刷缠绕，必要时更换防缠绕配件")
    if "回充" in issues:
        suggestions.append("检查充电座位置与回充传感器，减少回充失败")
    if "漏扫" in issues:
        suggestions.append("复查家具底部和边角区域，必要时开启沿边或局部补扫")
    if not suggestions:
        suggestions.append("保持当前清洁频率，并按月复查耗材状态和清洁效率")

    return "；".join(_dedupe_keep_order(suggestions)[:2])



def extract_first_percentage(text: str) -> int | None:
    matched = re.search(r"(\d+)%", text or "")
    if matched is None:
        return None
    return int(matched.group(1))



def score_consumables(text: str) -> int:
    content = text or ""
    if any(keyword in content for keyword in ["急需", "紧急", "待更换", "剩余10%", "剩余15%", "剩余20%", "剩余20天", "剩余25天"]):
        return 3
    if any(keyword in content for keyword in ["中度", "剩余30%", "剩余35%", "剩余40%", "剩余30天", "剩余35天", "剩余40天"]):
        return 2
    return 1



def count_issue_indicators(record: dict[str, str]) -> int:
    text = "\n".join([record.get("efficiency", ""), record.get("consumables", ""), record.get("comparison", "")])
    return sum(1 for keyword in ISSUE_LABELS if keyword in text)



def extract_issue_labels(record: dict[str, str]) -> list[str]:
    text = "\n".join([record.get("efficiency", ""), record.get("consumables", ""), record.get("comparison", "")])
    matched: list[str] = []
    for keyword, label in ISSUE_LABELS.items():
        if keyword in text:
            matched.append(label)
    return _dedupe_keep_order(matched)



def first_content_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "暂无记录"



def _dedupe_keep_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        value = (item or "").strip()
        if value and value not in deduped:
            deduped.append(value)
    return deduped

