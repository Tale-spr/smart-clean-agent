import shutil
import unittest
import uuid
from pathlib import Path

from smart_clean_agent.services.user_memory_service import build_user_memory_summary, load_user_memory, update_user_memory


class UserMemoryServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"user_memory_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.profile = {
            "user_id": "1001",
            "city": "北京",
            "name": "用户A",
            "house_type": "65㎡公寓",
            "floor_type": "木地板",
        }

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_load_user_memory_returns_empty_shell_for_new_user(self):
        memory = load_user_memory("1001", str(self.base_dir))

        self.assertEqual(memory["user_id"], "1001")
        self.assertEqual(memory["preferences"], [])
        self.assertEqual(memory["recent_focuses"], [])

    def test_update_user_memory_extracts_profile_and_message_signals(self):
        messages = [
            {"role": "user", "content": "我想多用湿拖，而且最近总是漏扫沙发底和回充失败。"},
            {"role": "assistant", "content": "收到，我先帮你分析原因。"},
            {"role": "user", "content": "我更希望定时清扫，最好静音一点，我家还有宠物。"},
        ]

        memory = update_user_memory("1001", self.profile, messages, str(self.base_dir))

        self.assertEqual(memory["profile_snapshot"]["floor_type"], "木地板")
        self.assertIn("小户型", memory["environment"])
        self.assertIn("木地板", memory["environment"])
        self.assertIn("养宠", memory["environment"])
        self.assertIn("湿拖偏好", memory["preferences"])
        self.assertIn("静音偏好", memory["preferences"])
        self.assertIn("定时清扫", memory["cleaning_habits"])
        self.assertIn("漏扫问题", memory["pain_points"])
        self.assertIn("回充问题", memory["pain_points"])
        self.assertTrue(memory["recent_focuses"])

    def test_update_user_memory_does_not_duplicate_items(self):
        messages = [{"role": "user", "content": "我希望定时清扫，而且最近漏扫很明显。"}]

        first = update_user_memory("1001", self.profile, messages, str(self.base_dir))
        second = update_user_memory("1001", self.profile, messages, str(self.base_dir))

        self.assertEqual(first["cleaning_habits"], second["cleaning_habits"])
        self.assertEqual(first["pain_points"], second["pain_points"])

    def test_build_user_memory_summary_returns_stable_text(self):
        memory = update_user_memory(
            "1001",
            self.profile,
            [{"role": "user", "content": "我想要静音一些，也总觉得地图会错乱。"}],
            str(self.base_dir),
        )

        summary = build_user_memory_summary(memory)

        self.assertIn("用户画像:", summary)
        self.assertIn("长期偏好:", summary)
        self.assertIn("常见问题:", summary)


if __name__ == "__main__":
    unittest.main()

