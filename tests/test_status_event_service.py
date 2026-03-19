import unittest
from unittest.mock import Mock

from smart_clean_agent.services.status_event_service import get_visible_status_events, record_status_event


class StatusEventServiceTestCase(unittest.TestCase):
    def test_record_status_event_appends_to_context_and_invokes_callback(self):
        callback = Mock()
        events: list[dict[str, str]] = []
        context = {
            "user_id": "1001",
            "session_id": "session_001",
            "status_events": events,
            "status_event_callback": callback,
        }

        event = record_status_event(
            context,
            event_type="stage.model",
            title="正在分析问题",
            detail="正在判断是否需要调用工具",
        )

        self.assertEqual(events, [event])
        callback.assert_called_once_with(event)

    def test_get_visible_status_events_filters_internal_events(self):
        events = [
            {"event_type": "stage.memory", "title": "正在整理历史记忆", "detail": "", "created_at": "", "level": "info"},
            {"event_type": "tool.success", "title": "工具调用完成", "detail": "", "created_at": "", "level": "info"},
            {"event_type": "error.tool", "title": "工具调用失败", "detail": "", "created_at": "", "level": "error"},
        ]

        visible_events = get_visible_status_events(events)

        self.assertEqual([event["event_type"] for event in visible_events], ["stage.memory", "error.tool"])


if __name__ == "__main__":
    unittest.main()

