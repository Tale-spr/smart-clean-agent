import json
from datetime import datetime
from pathlib import Path

from smart_clean_agent.services.conversation_memory_service import build_recent_history, summarize_messages
from smart_clean_agent.utils.config_handler import agent_conf
from smart_clean_agent.utils.path_tool import get_abs_path



def get_session_store_dir(base_dir: str | None = None) -> Path:
    session_dir = base_dir or get_abs_path(agent_conf["session_store_dir"])
    path = Path(session_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path



def get_user_session_dir(user_id: str, base_dir: str | None = None) -> Path:
    session_dir = get_session_store_dir(base_dir) / user_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir



def generate_session_metadata() -> tuple[str, str, str]:
    now = datetime.now()
    session_id = now.strftime("%Y%m%d_%H%M%S_%f")
    display_time = now.strftime("%Y-%m-%d %H:%M:%S")
    iso_time = now.isoformat(timespec="seconds")
    return session_id, display_time, iso_time



def build_session_data(
    user_id: str,
    session_id: str,
    title: str,
    created_at: str,
    updated_at: str,
    messages: list[dict[str, str]] | None = None,
    session_summary: str = "",
    recent_history: str = "",
) -> dict:
    return {
        "session_id": session_id,
        "user_id": user_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": messages or [],
        "session_summary": session_summary,
        "recent_history": recent_history,
    }



def get_session_path(user_id: str, session_id: str, base_dir: str | None = None) -> Path:
    return get_user_session_dir(user_id, base_dir) / f"{session_id}.json"



def save_session(user_id: str, session_data: dict, base_dir: str | None = None) -> dict:
    session_id = session_data["session_id"]
    session_path = get_session_path(user_id, session_id, base_dir)
    current_time = datetime.now().isoformat(timespec="seconds")
    messages = session_data.get("messages", [])

    session_payload = {
        "session_id": session_id,
        "user_id": user_id,
        "title": session_data.get("title") or session_id,
        "created_at": session_data.get("created_at") or current_time,
        "updated_at": current_time,
        "messages": messages,
        "session_summary": summarize_messages(messages),
        "recent_history": build_recent_history(messages),
    }

    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_payload, f, ensure_ascii=False, indent=2)

    return session_payload



def load_session(user_id: str, session_id: str, base_dir: str | None = None) -> dict | None:
    session_path = get_session_path(user_id, session_id, base_dir)
    if not session_path.exists():
        return None

    with open(session_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    session_data.setdefault("session_summary", summarize_messages(session_data.get("messages", [])))
    session_data.setdefault("recent_history", build_recent_history(session_data.get("messages", [])))
    return session_data



def list_sessions(user_id: str, base_dir: str | None = None) -> list[dict]:
    user_dir = get_user_session_dir(user_id, base_dir)
    sessions: list[dict] = []
    for session_file in user_dir.glob("*.json"):
        loaded = load_session(user_id, session_file.stem, base_dir)
        if loaded is not None:
            sessions.append(loaded)

    return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)



def create_session(user_id: str, base_dir: str | None = None, title: str | None = None) -> dict:
    session_id, display_time, iso_time = generate_session_metadata()
    session_data = build_session_data(
        user_id=user_id,
        session_id=session_id,
        title=title or f"会话 {display_time}",
        created_at=iso_time,
        updated_at=iso_time,
        messages=[],
    )
    return save_session(user_id, session_data, base_dir)



def delete_session(user_id: str, session_id: str, base_dir: str | None = None) -> bool:
    session_path = get_session_path(user_id, session_id, base_dir)
    if not session_path.exists():
        return False

    session_path.unlink()
    return True



def get_latest_session(user_id: str, base_dir: str | None = None) -> dict | None:
    sessions = list_sessions(user_id, base_dir)
    return sessions[0] if sessions else None

