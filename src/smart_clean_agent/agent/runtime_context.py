from collections.abc import Callable
from typing import Any, NotRequired, TypedDict


class AgentRuntimeContext(TypedDict):
    report: bool
    user_id: str
    city: str
    session_id: NotRequired[str]
    session_summary: str
    recent_history: str
    user_memory_summary: str
    report_memory_summary: str
    trace_tool_calls: NotRequired[list[str]]
    status_events: NotRequired[list[dict[str, str]]]
    status_event_callback: NotRequired[Callable[[dict[str, str]], Any]]

