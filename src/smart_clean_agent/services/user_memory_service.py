import json
import re
from datetime import datetime
from pathlib import Path

from smart_clean_agent.services.conversation_memory_service import normalize_message_text
from smart_clean_agent.utils.config_handler import agent_conf
from smart_clean_agent.utils.path_tool import get_abs_path

USER_MEMORY_DEFAULT_DIR = "data/memory/users"
MAX_RECENT_FOCUSES = 8
MAX_RECENT_MESSAGES = 3

PREFERENCE_KEYWORDS = {
    "湿拖": "湿拖偏好",
    "拖地": "湿拖偏好",
    "静音": "静音偏好",
    "安静": "静音偏好",
}

CLEANING_HABIT_KEYWORDS = {
    "定时": "定时清扫",
    "预约": "定时清扫",
    "每天": "高频清扫",
    "每日": "高频清扫",
    "高频": "高频清扫",
    "沿边": "沿边清扫",
}

ENVIRONMENT_KEYWORDS = {
    "木地板": "木地板",
    "瓷砖": "瓷砖",
    "地毯": "地毯",
    "大理石": "大理石",
    "混合地面": "混合地面",
    "养宠": "养宠",
    "宠物": "养宠",
    "猫": "养宠",
    "狗": "养宠",
    "小户型": "小户型",
    "大户型": "大户型",
    "多层": "多层",
    "老人": "老人",
    "儿童": "儿童",
}

PAIN_POINT_KEYWORDS = {
    "避障": "避障问题",
    "回充": "回充问题",
    "漏扫": "漏扫问题",
    "主刷": "主刷缠绕",
    "缠绕": "主刷缠绕",
    "滤网": "滤网维护",
    "水箱": "水箱问题",
    "地图": "地图错乱",
    "wifi": "联网问题",
    "wi-fi": "联网问题",
}

USER_MEMORY_KEYS = [
    "user_id",
    "profile_snapshot",
    "preferences",
    "environment",
    "cleaning_habits",
    "pain_points",
    "recent_focuses",
    "last_updated_at",
]



def get_user_memory_dir(base_dir: str | None = None) -> Path:
    memory_dir = base_dir or get_abs_path(agent_conf.get("user_memory_dir", USER_MEMORY_DEFAULT_DIR))
    path = Path(memory_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path



def get_user_memory_path(user_id: str, base_dir: str | None = None) -> Path:
    return get_user_memory_dir(base_dir) / f"{user_id}.json"



def build_profile_snapshot(profile: dict[str, str] | None) -> dict[str, str]:
    source = profile or {}
    return {
        "user_id": (source.get("user_id") or "").strip(),
        "city": (source.get("city") or "").strip(),
        "name": (source.get("name") or "").strip(),
        "house_type": (source.get("house_type") or "").strip(),
        "floor_type": (source.get("floor_type") or "").strip(),
    }



def build_empty_user_memory(user_id: str, profile_snapshot: dict[str, str] | None = None) -> dict:
    return {
        "user_id": user_id,
        "profile_snapshot": profile_snapshot or build_profile_snapshot({"user_id": user_id}),
        "preferences": [],
        "environment": [],
        "cleaning_habits": [],
        "pain_points": [],
        "recent_focuses": [],
        "last_updated_at": "",
    }



def load_user_memory(user_id: str, base_dir: str | None = None) -> dict:
    memory_path = get_user_memory_path(user_id, base_dir)
    if not memory_path.exists():
        return build_empty_user_memory(user_id)

    with open(memory_path, "r", encoding="utf-8") as f:
        memory = json.load(f)

    payload = build_empty_user_memory(user_id)
    payload.update(memory)
    payload["user_id"] = user_id
    return payload



def save_user_memory(user_id: str, memory: dict, base_dir: str | None = None) -> dict:
    memory_path = get_user_memory_path(user_id, base_dir)
    payload = build_empty_user_memory(user_id)
    payload.update(memory)
    payload["user_id"] = user_id
    payload["profile_snapshot"] = build_profile_snapshot(payload.get("profile_snapshot"))
    for key in ["preferences", "environment", "cleaning_habits", "pain_points", "recent_focuses"]:
        payload[key] = _dedupe_keep_order(payload.get(key, []), limit=MAX_RECENT_FOCUSES if key == "recent_focuses" else None)

    if not payload.get("last_updated_at"):
        payload["last_updated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload



def update_user_memory(
    user_id: str,
    profile: dict[str, str] | None,
    messages: list[dict[str, str]],
    base_dir: str | None = None,
) -> dict:
    snapshot = build_profile_snapshot(profile)
    memory = load_user_memory(user_id, base_dir)
    extracted = extract_user_memory_signals(messages)
    static_environment = extract_environment_from_profile(snapshot)

    memory["profile_snapshot"] = snapshot
    memory["preferences"] = _dedupe_keep_order(memory.get("preferences", []) + extracted["preferences"])
    memory["environment"] = _dedupe_keep_order(memory.get("environment", []) + static_environment + extracted["environment"])
    memory["cleaning_habits"] = _dedupe_keep_order(memory.get("cleaning_habits", []) + extracted["cleaning_habits"])
    memory["pain_points"] = _dedupe_keep_order(memory.get("pain_points", []) + extracted["pain_points"])
    memory["recent_focuses"] = _dedupe_keep_order(memory.get("recent_focuses", []) + extracted["recent_focuses"], limit=MAX_RECENT_FOCUSES)
    memory["last_updated_at"] = datetime.now().isoformat(timespec="seconds")
    return save_user_memory(user_id, memory, base_dir)



def extract_environment_from_profile(profile_snapshot: dict[str, str]) -> list[str]:
    signals: list[str] = []
    floor_type = profile_snapshot.get("floor_type", "")
    house_type = profile_snapshot.get("house_type", "")

    if floor_type:
        signals.append(floor_type)
    if "别墅" in house_type:
        signals.extend(["大户型", "多层"])
    if "多层" in house_type:
        signals.append("多层")
    if "老人" in house_type:
        signals.append("老人")
    if "儿童" in house_type:
        signals.append("儿童")

    area = extract_area_value(house_type)
    if area is not None:
        if area <= 70:
            signals.append("小户型")
        if area >= 120:
            signals.append("大户型")

    return _dedupe_keep_order(signals)



def extract_area_value(text: str) -> int | None:
    matched = re.search(r"(\d+)\s*㎡", text or "")
    if matched is None:
        return None
    return int(matched.group(1))



def extract_user_memory_signals(messages: list[dict[str, str]]) -> dict[str, list[str]]:
    preferences: list[str] = []
    environment: list[str] = []
    cleaning_habits: list[str] = []
    pain_points: list[str] = []
    recent_focuses: list[str] = []
    user_messages = [message for message in messages if (message.get("role") or "").strip() == "user"]

    for message in user_messages:
        content = (message.get("content") or "").strip()
        if not content:
            continue
        lowered = content.lower()
        matched_labels: list[str] = []

        matched_labels.extend(_match_keyword_labels(content, ENVIRONMENT_KEYWORDS))
        environment.extend(_match_keyword_labels(content, ENVIRONMENT_KEYWORDS))
        preferences.extend(_match_keyword_labels(content, PREFERENCE_KEYWORDS))
        cleaning_habits.extend(_match_keyword_labels(content, CLEANING_HABIT_KEYWORDS))

        pain_matches = _match_keyword_labels(lowered, PAIN_POINT_KEYWORDS)
        pain_points.extend(pain_matches)

        matched_labels.extend(_match_keyword_labels(content, PREFERENCE_KEYWORDS))
        matched_labels.extend(_match_keyword_labels(content, CLEANING_HABIT_KEYWORDS))
        matched_labels.extend(pain_matches)

        if matched_labels:
            recent_focuses.extend(matched_labels)
        else:
            recent_focuses.append(normalize_message_text(content, max_length=30))

    recent_focus_snippets = [
        normalize_message_text((message.get("content") or ""), max_length=30)
        for message in user_messages[-MAX_RECENT_MESSAGES:]
        if (message.get("content") or "").strip()
    ]
    recent_focuses.extend(recent_focus_snippets)

    return {
        "preferences": _dedupe_keep_order(preferences),
        "environment": _dedupe_keep_order(environment),
        "cleaning_habits": _dedupe_keep_order(cleaning_habits),
        "pain_points": _dedupe_keep_order(pain_points),
        "recent_focuses": _dedupe_keep_order(recent_focuses, limit=MAX_RECENT_FOCUSES),
    }



def _match_keyword_labels(text: str, mapping: dict[str, str]) -> list[str]:
    matched: list[str] = []
    for keyword, label in mapping.items():
        if keyword.lower() in text.lower():
            matched.append(label)
    return matched



def _dedupe_keep_order(items: list[str], limit: int | None = None) -> list[str]:
    values = [item.strip() for item in items if item and item.strip()]
    if limit is not None:
        values = values[-limit:]

    deduped: list[str] = []
    for item in values:
        if item not in deduped:
            deduped.append(item)
    return deduped



def build_user_memory_summary(memory: dict) -> str:
    profile_snapshot = memory.get("profile_snapshot") or {}
    preferences = memory.get("preferences") or []
    environment = memory.get("environment") or []
    cleaning_habits = memory.get("cleaning_habits") or []
    pain_points = memory.get("pain_points") or []
    recent_focuses = memory.get("recent_focuses") or []

    lines: list[str] = []
    city = profile_snapshot.get("city", "")
    house_type = profile_snapshot.get("house_type", "")
    floor_type = profile_snapshot.get("floor_type", "")
    if city or house_type or floor_type:
        profile_parts = []
        if city:
            profile_parts.append(f"城市: {city}")
        if house_type:
            profile_parts.append(f"户型: {house_type}")
        if floor_type:
            profile_parts.append(f"地面: {floor_type}")
        lines.append("用户画像: " + "；".join(profile_parts))
    if preferences:
        lines.append("长期偏好: " + "、".join(preferences))
    if environment:
        lines.append("环境特征: " + "、".join(environment))
    if cleaning_habits:
        lines.append("清洁习惯: " + "、".join(cleaning_habits))
    if pain_points:
        lines.append("常见问题: " + "、".join(pain_points))
    if recent_focuses:
        lines.append("最近关注: " + "；".join(recent_focuses[-MAX_RECENT_MESSAGES:]))

    return "\n".join(lines)

