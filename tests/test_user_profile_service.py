import shutil
import unittest
import uuid
from pathlib import Path

from smart_clean_agent.services.user_profile_service import (
    _load_profiles_cached,
    get_user_profile,
    load_user_profiles,
    save_user_profiles,
    upsert_user_profile,
)


class UserProfileServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path("tests") / ".tmp"
        self.temp_dir.mkdir(exist_ok=True)
        self.csv_path = self.temp_dir / f"user_profiles_{uuid.uuid4().hex}.csv"
        self.csv_path.write_text(
            "user_id,city,name,house_type,floor_type\n"
            "1001,北京,用户A,65㎡公寓,木地板\n"
            "1002,上海,用户B,70㎡公寓,瓷砖\n",
            encoding="utf-8",
        )
        _load_profiles_cached.cache_clear()

    def tearDown(self):
        if self.csv_path.exists():
            self.csv_path.unlink()
        _load_profiles_cached.cache_clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_user_profiles_reads_valid_csv(self):
        profiles = load_user_profiles(str(self.csv_path))

        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0]["user_id"], "1001")
        self.assertEqual(profiles[0]["city"], "北京")

    def test_load_user_profiles_requires_user_id_and_city(self):
        invalid_path = self.temp_dir / f"invalid_profiles_{uuid.uuid4().hex}.csv"
        invalid_path.write_text("user_id,name\n1001,用户A\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            load_user_profiles(str(invalid_path))

    def test_save_user_profiles_overwrites_file(self):
        save_user_profiles(
            [{"user_id": "2001", "city": "深圳", "name": "用户C", "house_type": "两居", "floor_type": "木地板"}],
            str(self.csv_path),
        )

        profiles = load_user_profiles(str(self.csv_path))
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["user_id"], "2001")

    def test_upsert_user_profile_updates_existing_profile(self):
        updated_profile = upsert_user_profile(
            {
                "user_id": "1001",
                "city": "天津",
                "name": "用户A",
                "house_type": "65㎡公寓",
                "floor_type": "木地板",
            },
            str(self.csv_path),
            original_user_id="1001",
        )

        self.assertEqual(updated_profile["city"], "天津")
        self.assertEqual(get_user_profile("1001", str(self.csv_path))["city"], "天津")

    def test_upsert_user_profile_adds_new_profile(self):
        upsert_user_profile(
            {
                "user_id": "3001",
                "city": "广州",
                "name": "用户D",
                "house_type": "三居",
                "floor_type": "瓷砖",
            },
            str(self.csv_path),
        )

        self.assertIsNotNone(get_user_profile("3001", str(self.csv_path)))

    def test_upsert_user_profile_rejects_duplicate_user_id(self):
        with self.assertRaises(ValueError):
            upsert_user_profile(
                {
                    "user_id": "1002",
                    "city": "杭州",
                    "name": "用户A",
                    "house_type": "65㎡公寓",
                    "floor_type": "木地板",
                },
                str(self.csv_path),
                original_user_id="1001",
            )


if __name__ == "__main__":
    unittest.main()

