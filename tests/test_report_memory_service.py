import shutil
import unittest
import uuid
from pathlib import Path

from smart_clean_agent.services.report_memory_service import build_report_memory_summary, load_report_memory, refresh_report_memory


class ReportMemoryServiceTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"report_memory_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.base_dir / "records.csv"
        self.csv_path.write_text(
            '"用户ID","特征","清洁效率","耗材","对比","时间"\n'
            '"1001","65㎡公寓 | 单身 | 木地板","覆盖率:85%","主刷寿命:剩余60天","优于65%同面积用户","2025-01"\n'
            '"1001","65㎡公寓 | 单身 | 木地板","覆盖率:88%","主刷寿命:剩余40天","优于75%同面积用户","2025-02"\n'
            '"1001","65㎡公寓 | 单身 | 木地板","覆盖率:90%","主刷寿命:剩余20天","优于85%同面积用户","2025-03"\n'
            '"1002","70㎡公寓 | 情侣 | 瓷砖","覆盖率:80%","边刷寿命:剩余50天","低于同类10%","2025-03"\n',
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_load_report_memory_returns_empty_shell_for_new_user(self):
        memory = load_report_memory("1001", str(self.base_dir))

        self.assertEqual(memory["user_id"], "1001")
        self.assertEqual(memory["monthly_records"], [])
        self.assertEqual(memory["trend_summary"], "")

    def test_refresh_report_memory_aggregates_recent_months(self):
        memory = refresh_report_memory("1001", months=3, csv_path=str(self.csv_path), base_dir=str(self.base_dir))

        self.assertEqual(len(memory["monthly_records"]), 3)
        self.assertEqual(memory["monthly_records"][-1]["month"], "2025-03")
        self.assertIn("覆盖月份:", memory["trend_summary"])
        self.assertIn("清洁效率趋势:", memory["trend_summary"])

    def test_refresh_report_memory_gracefully_handles_short_history(self):
        memory = refresh_report_memory("1002", months=3, csv_path=str(self.csv_path), base_dir=str(self.base_dir))

        self.assertEqual(len(memory["monthly_records"]), 1)
        self.assertIn("2025-03", memory["trend_summary"])

    def test_build_report_memory_summary_returns_stable_text(self):
        memory = refresh_report_memory("1001", months=2, csv_path=str(self.csv_path), base_dir=str(self.base_dir))

        summary = build_report_memory_summary(memory)

        self.assertIn("覆盖月份:", summary)
        self.assertIn("耗材趋势:", summary)
        self.assertIn("总结建议:", summary)


if __name__ == "__main__":
    unittest.main()

