from collections.abc import Callable
from typing import Any, NotRequired, TypedDict


class AgentRuntimeContext(TypedDict):
    report: bool
    user_id: str
    city: str
    force_report_agent: NotRequired[bool]
    execution_mode: NotRequired[str]
    report_current_month: NotRequired[str]
    session_id: NotRequired[str]
    session_summary: str
    recent_history: str
    user_memory_summary: str
    user_memory_payload: dict[str, Any]
    retrieved_user_memory_summary: str
    retrieved_user_memory_fields: list[str]
    memory_retrieval_reason: str
    is_new_session_first_turn: bool
    report_memory_summary: str
    trace_tool_calls: NotRequired[list[str]]
    react_trace: NotRequired[list[dict[str, str]]]
    react_stop_reason: NotRequired[str]
    react_step_count: NotRequired[int]
    report_tool_sequence: NotRequired[list[str]]
    report_sequence_violation: NotRequired[bool]
    status_events: NotRequired[list[dict[str, str]]]
    status_event_callback: NotRequired[Callable[[dict[str, str]], Any]]

