from collections.abc import Callable
from datetime import datetime
from typing import Any

from smart_clean_agent.utils.logger_handler import logger


StatusEvent = dict[str, str]
VISIBLE_EVENT_PREFIXES = ("stage.", "error.")


def create_status_event(
    event_type: str,
    title: str,
    detail: str = "",
    level: str = "info",
) -> StatusEvent:
    return {
        "event_type": event_type,
        "title": title,
        "detail": detail,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "level": level,
    }


def record_status_event(
    context: dict[str, Any] | None,
    event_type: str,
    title: str,
    detail: str = "",
    level: str = "info",
) -> StatusEvent:
    event = create_status_event(event_type=event_type, title=title, detail=detail, level=level)
    user_id = (context or {}).get("user_id", "-")
    session_id = (context or {}).get("session_id", "-")
    logger.info(
        f"[状态事件]{event_type} | user_id={user_id} | session_id={session_id} | {title} | {detail}"
    )

    if not context:
        return event

    status_events = context.get("status_events")
    if isinstance(status_events, list):
        status_events.append(event)

    callback = context.get("status_event_callback")
    if isinstance(callback, Callable):
        callback(event)

    return event


def get_visible_status_events(events: list[StatusEvent]) -> list[StatusEvent]:
    return [event for event in events if (event.get("event_type") or "").startswith(VISIBLE_EVENT_PREFIXES)]

