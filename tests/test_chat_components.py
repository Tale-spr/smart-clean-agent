import unittest

from smart_clean_agent.ui.chat_components import build_process_notes, build_process_panel_html


class ChatComponentsTestCase(unittest.TestCase):
    def test_build_process_notes_maps_events_to_user_friendly_lines(self):
        events = [
            {"event_type": "stage.memory", "detail": "", "title": "", "created_at": "", "level": "info"},
            {"event_type": "stage.tool", "detail": "正在调用 get_weather", "title": "", "created_at": "", "level": "info"},
            {"event_type": "stage.final", "detail": "", "title": "", "created_at": "", "level": "info"},
        ]

        notes = build_process_notes(events)

        self.assertEqual(
            notes,
            [
                "我先结合你最近的会话、长期偏好和趋势记录看一下。",
                "我在查询你所在城市的天气情况。",
                "我已经整理好结论，马上给你正式答复。",
            ],
        )

    def test_build_process_notes_deduplicates_same_note(self):
        events = [
            {"event_type": "stage.model", "detail": "", "title": "", "created_at": "", "level": "info"},
            {"event_type": "stage.model", "detail": "", "title": "", "created_at": "", "level": "info"},
            {"event_type": "stage.tool", "detail": "正在调用 get_user_location", "title": "", "created_at": "", "level": "info"},
            {"event_type": "stage.tool", "detail": "正在调用 get_user_location", "title": "", "created_at": "", "level": "info"},
        ]

        notes = build_process_notes(events)

        self.assertEqual(
            notes,
            [
                "我在判断这次需要补哪些信息，才能给你更准确的答复。",
                "我在确认你当前所在的城市。",
            ],
        )

    def test_build_process_panel_html_contains_spinner_when_processing(self):
        html = build_process_panel_html(["我在查询天气。"], is_processing=True)

        self.assertIn("process-note-spinner", html)
        self.assertIn("我正在整理信息，请稍等片刻", html)
        self.assertIn("1. 我在查询天气。", html)

    def test_build_process_panel_html_hides_spinner_when_finished(self):
        html = build_process_panel_html(["我已经整理好结论。"], is_processing=False)

        self.assertNotIn("我正在整理信息，请稍等片刻", html)
        self.assertIn("本轮处理过程已完成", html)
        self.assertIn("1. 我已经整理好结论。", html)


if __name__ == "__main__":
    unittest.main()

