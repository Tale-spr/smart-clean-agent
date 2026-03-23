import json
import shutil
import unittest
import uuid
from pathlib import Path

from smart_clean_agent.evaluation.migrate_dataset import convert_csv_to_jsonl


class EvalMigrateDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"eval_migrate_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_convert_csv_to_jsonl_generates_expected_shape(self):
        csv_path = self.base_dir / "eval_cases.csv"
        csv_path.write_text(
            "case_id,query,category,expected_keywords,expected_tool,expected_retrieval_hit,user_id,city\n"
            "faq_001,测试问题,faq,关键词A|关键词B,rag_summarize,true,1002,上海\n",
            encoding="utf-8",
        )
        jsonl_path = self.base_dir / "eval_cases.jsonl"

        count = convert_csv_to_jsonl(csv_path, jsonl_path)

        self.assertEqual(count, 1)
        payload = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
        self.assertEqual(payload["expected_route"], "normal")
        self.assertEqual(payload["required_tools"], ["rag_summarize"])
        self.assertEqual(payload["expected_retrieval_mode"], "required")
        self.assertEqual(payload["required_points"][0]["label"], "关键词A")


if __name__ == "__main__":
    unittest.main()
