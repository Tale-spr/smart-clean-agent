from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, MutableMapping

from smart_clean_agent.agent.react_agent import ReactAgent
from smart_clean_agent.agent.runtime_context import AgentRuntimeContext
from smart_clean_agent.agent.tools.agent_tools import create_agent_tools
from smart_clean_agent.model.factory import create_chat_model, create_embedding_model
from smart_clean_agent.rag.rag_service import RagSummarizeService
from smart_clean_agent.rag.vector_store import VectorStoreService
from smart_clean_agent.services.conversation_memory_service import build_recent_history, summarize_messages
from smart_clean_agent.services.dependency_service import validate_runtime_dependencies
from smart_clean_agent.services.report_memory_service import build_report_memory_summary, load_report_memory, refresh_report_memory
from smart_clean_agent.services.session_service import create_session, get_latest_session, load_session, save_session
from smart_clean_agent.services.user_memory_service import build_profile_snapshot, build_user_memory_summary, load_user_memory, update_user_memory
from smart_clean_agent.services.user_profile_service import get_user_profile, list_user_profiles
from smart_clean_agent.utils.logger_handler import logger


@dataclass
class AppContext:
    user_profiles: list[dict[str, str]]
    selected_profile: dict[str, str]



def validate_online_dependencies() -> None:
    validate_runtime_dependencies()



def build_agent() -> ReactAgent:
    chat_model = create_chat_model()
    embedding_model = create_embedding_model()
    vector_store_service = VectorStoreService(embedding_function=embedding_model)
    rag_service = RagSummarizeService(
        model=chat_model,
        vector_store_service=vector_store_service,
    )
    tools = create_agent_tools(rag_service)
    return ReactAgent(model=chat_model, tools=tools)



def get_or_create_agent(session_state: MutableMapping[str, Any]) -> ReactAgent:
    agent = session_state.get("agent")
    if agent is None:
        agent = build_agent()
        session_state["agent"] = agent
    return agent



def initialize_ui_state(session_state: MutableMapping[str, Any], user_options: list[str]) -> None:
    if "message" not in session_state:
        session_state["message"] = []
    if "current_status_events" not in session_state:
        session_state["current_status_events"] = []
    if "latest_status_events" not in session_state:
        session_state["latest_status_events"] = []

    if "selected_user_id" not in session_state or session_state["selected_user_id"] not in user_options:
        session_state["selected_user_id"] = user_options[0]



def sync_session_state(session_state: MutableMapping[str, Any], user_id: str, session_data: dict) -> None:
    session_state["current_user_id"] = user_id
    session_state["current_session_id"] = session_data["session_id"]
    session_state["current_session_title"] = session_data["title"]
    session_state["current_session_created_at"] = session_data["created_at"]
    session_state["current_session_summary"] = session_data.get("session_summary", "")
    session_state["current_recent_history"] = session_data.get("recent_history", "")
    session_state["message"] = session_data["messages"]



def ensure_active_session(session_state: MutableMapping[str, Any], user_id: str) -> None:
    latest_session = get_latest_session(user_id)
    if latest_session is None:
        latest_session = create_session(user_id)

    sync_session_state(session_state, user_id, latest_session)



def initialize_memory_state(session_state: MutableMapping[str, Any], selected_profile: dict[str, str]) -> None:
    user_id = selected_profile["user_id"]
    messages = session_state.get("message", [])

    try:
        user_memory = load_user_memory(user_id)
        if user_memory.get("profile_snapshot") != build_profile_snapshot(selected_profile):
            update_user_memory(user_id, selected_profile, messages)
    except Exception as exc:
        logger.warning(f"[Memory]初始化用户长期记忆失败: {str(exc)}")

    try:
        report_memory = load_report_memory(user_id)
        if not report_memory.get("monthly_records"):
            refresh_report_memory(user_id)
    except Exception as exc:
        logger.warning(f"[Memory]初始化报告记忆失败: {str(exc)}")



def save_current_session(session_state: MutableMapping[str, Any]) -> None:
    current_session_id = session_state.get("current_session_id")
    current_user_id = session_state.get("current_user_id")
    if not current_session_id or not current_user_id:
        return

    session_data = {
        "session_id": current_session_id,
        "title": session_state.get("current_session_title", current_session_id),
        "created_at": session_state.get("current_session_created_at"),
        "messages": session_state.get("message", []),
    }
    saved_session = save_session(current_user_id, session_data)
    session_state["current_session_title"] = saved_session["title"]
    session_state["current_session_created_at"] = saved_session["created_at"]
    session_state["current_session_summary"] = saved_session.get("session_summary", "")
    session_state["current_recent_history"] = saved_session.get("recent_history", "")

    try:
        profile = get_user_profile(current_user_id) or {
            "user_id": current_user_id,
            "city": session_state.get("selected_city", ""),
        }
        update_user_memory(current_user_id, profile, session_state.get("message", []))
        refresh_report_memory(current_user_id)
    except Exception as exc:
        logger.warning(f"[Memory]会话保存后更新记忆失败: {str(exc)}")



def build_runtime_context(
    session_state: MutableMapping[str, Any],
    status_events: list[dict[str, str]] | None = None,
    status_event_callback: Callable[[dict[str, str]], Any] | None = None,
) -> AgentRuntimeContext:
    user_id = session_state["selected_user_id"]
    messages = session_state.get("message", [])
    user_memory_summary = ""
    report_memory_summary = ""

    try:
        user_memory = load_user_memory(user_id)
        if not user_memory.get("profile_snapshot"):
            profile = get_user_profile(user_id) or {"user_id": user_id, "city": session_state.get("selected_city", "")}
            user_memory["profile_snapshot"] = build_profile_snapshot(profile)
        user_memory_summary = build_user_memory_summary(user_memory)
    except Exception as exc:
        logger.warning(f"[Memory]构建用户长期记忆上下文失败: {str(exc)}")

    try:
        report_memory = load_report_memory(user_id)
        if not report_memory.get("monthly_records"):
            report_memory = refresh_report_memory(user_id)
        report_memory_summary = build_report_memory_summary(report_memory)
    except Exception as exc:
        logger.warning(f"[Memory]构建报告记忆上下文失败: {str(exc)}")

    runtime_context: AgentRuntimeContext = {
        "report": False,
        "user_id": user_id,
        "city": session_state["selected_city"],
        "session_id": session_state.get("current_session_id", ""),
        "session_summary": summarize_messages(messages),
        "recent_history": build_recent_history(messages),
        "user_memory_summary": user_memory_summary,
        "report_memory_summary": report_memory_summary,
    }
    if status_events is not None:
        runtime_context["status_events"] = status_events
    if status_event_callback is not None:
        runtime_context["status_event_callback"] = status_event_callback
    return runtime_context



def build_app_context(session_state: MutableMapping[str, Any]) -> AppContext:
    user_profiles = list_user_profiles()
    if not user_profiles:
        raise ValueError("未找到可用用户资料")

    user_options = [profile["user_id"] for profile in user_profiles]
    initialize_ui_state(session_state, user_options)

    selected_user_id = session_state["selected_user_id"]
    selected_profile = get_user_profile(selected_user_id)
    if not selected_profile:
        raise ValueError("当前用户资料不存在")

    session_state["selected_city"] = selected_profile["city"]

    if session_state.get("current_user_id") != selected_user_id:
        ensure_active_session(session_state, selected_user_id)
    elif not session_state.get("current_session_id"):
        ensure_active_session(session_state, selected_user_id)
    elif load_session(selected_user_id, session_state["current_session_id"]) is None:
        ensure_active_session(session_state, selected_user_id)

    initialize_memory_state(session_state, selected_profile)
    return AppContext(user_profiles=user_profiles, selected_profile=selected_profile)

