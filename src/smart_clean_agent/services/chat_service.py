from collections.abc import Callable
from functools import lru_cache
from typing import Any

from smart_clean_agent.agent.react_agent import ReactAgent
from smart_clean_agent.agent.runtime_context import AgentRuntimeContext
from smart_clean_agent.web.bootstrap import build_agent
from smart_clean_agent.services.conversation_memory_service import build_recent_history, summarize_messages
from smart_clean_agent.services.report_memory_service import build_report_memory_summary, load_report_memory, refresh_report_memory
from smart_clean_agent.services.session_service import create_session, load_session, save_session
from smart_clean_agent.services.status_event_service import get_visible_status_events, record_status_event
from smart_clean_agent.services.user_memory_service import build_profile_snapshot, build_user_memory_summary, load_user_memory, update_user_memory
from smart_clean_agent.services.user_profile_service import get_user_profile
from smart_clean_agent.utils.logger_handler import logger


class ChatServiceError(Exception):
    def __init__(self, message: str, code: str = "chat_service_error", status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@lru_cache(maxsize=1)
def get_chat_agent() -> ReactAgent:
    return build_agent()


def run_chat(
    user_id: str,
    message: str,
    session_id: str | None = None,
    agent: ReactAgent | None = None,
    status_event_callback: Callable[[dict[str, str]], Any] | None = None,
) -> dict[str, Any]:
    return _run_agent_interaction(
        user_id=user_id,
        query=message,
        session_id=session_id,
        response_key="answer",
        agent=agent,
        status_event_callback=status_event_callback,
    )


def run_report(
    user_id: str,
    query: str,
    session_id: str | None = None,
    agent: ReactAgent | None = None,
    status_event_callback: Callable[[dict[str, str]], Any] | None = None,
) -> dict[str, Any]:
    return _run_agent_interaction(
        user_id=user_id,
        query=query,
        session_id=session_id,
        response_key="report",
        agent=agent,
        status_event_callback=status_event_callback,
        force_report_agent=True,
    )


def _run_agent_interaction(
    user_id: str,
    query: str,
    session_id: str | None,
    response_key: str,
    agent: ReactAgent | None,
    status_event_callback: Callable[[dict[str, str]], Any] | None,
    force_report_agent: bool = False,
) -> dict[str, Any]:
    normalized_user_id = (user_id or "").strip()
    normalized_query = (query or "").strip()
    if not normalized_user_id:
        raise ChatServiceError("user_id 不能为空", code="invalid_user_id", status_code=422)
    if not normalized_query:
        raise ChatServiceError("message/query 不能为空", code="invalid_query", status_code=422)

    profile = get_user_profile(normalized_user_id)
    if profile is None:
        raise ChatServiceError("用户资料不存在", code="user_not_found", status_code=404)

    session_data = _load_target_session(normalized_user_id, session_id)
    updated_messages = list(session_data.get("messages", []))
    updated_messages.append({"role": "user", "content": normalized_query})
    session_data["messages"] = updated_messages

    status_events: list[dict[str, str]] = []
    callback = status_event_callback
    pre_runtime_context: dict[str, Any] = {
        "user_id": normalized_user_id,
        "session_id": session_data["session_id"],
        "status_events": status_events,
    }
    if callback is not None:
        pre_runtime_context["status_event_callback"] = callback

    record_status_event(
        pre_runtime_context,
        event_type="stage.memory",
        title="正在整理历史记忆",
        detail="正在整理当前会话摘要、长期记忆与趋势记忆",
    )

    runtime_context = _build_runtime_context(
        profile=profile,
        session_data=session_data,
        status_events=status_events,
        status_event_callback=callback,
        force_report_agent=force_report_agent,
    )

    runtime_agent = agent or get_chat_agent()
    try:
        response_text = runtime_agent.execute(normalized_query, runtime_context)
    except Exception as exc:
        logger.error(f"[ChatService]用户{normalized_user_id}请求执行失败: {str(exc)}", exc_info=True)
        raise ChatServiceError("Agent 执行失败", code="agent_execution_failed", status_code=500) from exc

    session_data["messages"].append({"role": "assistant", "content": response_text})
    saved_session = save_session(normalized_user_id, session_data)
    report_memory_summary = _refresh_memories(normalized_user_id, profile, saved_session["messages"])

    return {
        "user_id": normalized_user_id,
        "session_id": saved_session["session_id"],
        response_key: response_text,
        "status_events": get_visible_status_events(status_events),
        "session_summary": saved_session.get("session_summary", ""),
        "report_memory_summary": report_memory_summary,
        "session_data": saved_session,
    }


def _load_target_session(user_id: str, session_id: str | None) -> dict[str, Any]:
    normalized_session_id = (session_id or "").strip()
    if normalized_session_id:
        session_data = load_session(user_id, normalized_session_id)
        if session_data is None:
            raise ChatServiceError("指定会话不存在", code="session_not_found", status_code=404)
        return session_data

    return create_session(user_id)


def _build_runtime_context(
    profile: dict[str, str],
    session_data: dict[str, Any],
    status_events: list[dict[str, str]],
    status_event_callback: Callable[[dict[str, str]], Any] | None,
    force_report_agent: bool,
) -> AgentRuntimeContext:
    messages = session_data.get("messages", [])
    user_id = profile["user_id"]
    user_memory_summary = ""
    report_memory_summary = ""

    try:
        user_memory = load_user_memory(user_id)
        if not user_memory.get("profile_snapshot"):
            user_memory["profile_snapshot"] = build_profile_snapshot(profile)
        user_memory_summary = build_user_memory_summary(user_memory)
    except Exception as exc:
        logger.warning(f"[ChatService]构建用户长期记忆上下文失败: {str(exc)}")

    try:
        report_memory = load_report_memory(user_id)
        if not report_memory.get("monthly_records"):
            report_memory = refresh_report_memory(user_id)
        report_memory_summary = build_report_memory_summary(report_memory)
    except Exception as exc:
        logger.warning(f"[ChatService]构建报告记忆上下文失败: {str(exc)}")

    runtime_context: AgentRuntimeContext = {
        "report": False,
        "force_report_agent": force_report_agent,
        "execution_mode": "",
        "user_id": user_id,
        "city": profile["city"],
        "session_id": session_data["session_id"],
        "session_summary": summarize_messages(messages),
        "recent_history": build_recent_history(messages),
        "user_memory_summary": user_memory_summary,
        "report_memory_summary": report_memory_summary,
        "trace_tool_calls": [],
        "react_trace": [],
        "react_step_count": 0,
        "react_stop_reason": "",
        "report_tool_sequence": [],
        "report_sequence_violation": False,
        "status_events": status_events,
    }
    if status_event_callback is not None:
        runtime_context["status_event_callback"] = status_event_callback

    return runtime_context


def _refresh_memories(user_id: str, profile: dict[str, str], messages: list[dict[str, str]]) -> str:
    try:
        update_user_memory(user_id, profile, messages)
        report_memory = refresh_report_memory(user_id)
        return build_report_memory_summary(report_memory)
    except Exception as exc:
        logger.warning(f"[ChatService]会话保存后更新记忆失败: {str(exc)}")
        return ""

