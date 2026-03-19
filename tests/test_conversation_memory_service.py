import unittest

from smart_clean_agent.services.conversation_memory_service import build_recent_history, summarize_messages


class ConversationMemoryServiceTestCase(unittest.TestCase):
    def test_build_recent_history_formats_recent_messages(self):
        messages = [
            {"role": "user", "content": "今天天气怎么样？"},
            {"role": "assistant", "content": "北京今天晴。"},
            {"role": "user", "content": "适合扫地吗？"},
        ]

        history = build_recent_history(messages)

        self.assertIn("用户: 今天天气怎么样？", history)
        self.assertIn("助手: 北京今天晴。", history)
        self.assertIn("用户: 适合扫地吗？", history)

    def test_summarize_messages_builds_compact_summary(self):
        messages = [
            {"role": "user", "content": "我想看一下本月使用报告"},
            {"role": "assistant", "content": "可以先帮你查询当前月份和使用记录。"},
            {"role": "user", "content": "顺便告诉我最近怎么保养"},
        ]

        summary = summarize_messages(messages)

        self.assertIn("最近用户关注", summary)
        self.assertIn("最近已提供的信息", summary)


if __name__ == "__main__":
    unittest.main()

