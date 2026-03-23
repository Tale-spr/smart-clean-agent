from collections.abc import Callable
from typing import Any, NotRequired, TypedDict


class AgentRuntimeContext(TypedDict):
    report: bool
    user_id: str
    city: str
    force_report_agent: NotRequired[bool]
    execution_mode: NotRequired[str]
    session_id: NotRequired[str]
    session_summary: str
    recent_history: str
    user_memory_summary: str
    report_memory_summary: str
    trace_tool_calls: NotRequired[list[str]]
    react_trace: NotRequired[list[dict[str, str]]]
    react_stop_reason: NotRequired[str]
    react_step_count: NotRequired[int]
    report_tool_sequence: NotRequired[list[str]]
    report_sequence_violation: NotRequired[bool]
    status_events: NotRequired[list[dict[str, str]]]
    status_event_callback: NotRequired[Callable[[dict[str, str]], Any]]

