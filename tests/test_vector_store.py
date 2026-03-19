import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from smart_clean_agent.rag.vector_store import VectorStoreService, ensure_vector_store_ready, get_knowledge_source_files, get_vector_store_sqlite_path, vector_store_exists


class VectorStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path("tests") / ".tmp" / f"vector_store_{uuid.uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_vector_store_exists_returns_false_when_sqlite_missing(self):
        self.assertFalse(vector_store_exists(str(self.base_dir)))

    def test_vector_store_exists_returns_true_when_sqlite_exists(self):
        sqlite_path = get_vector_store_sqlite_path(str(self.base_dir))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_path.write_text("", encoding="utf-8")

        self.assertTrue(vector_store_exists(str(self.base_dir)))

    def test_ensure_vector_store_ready_raises_when_missing(self):
        with self.assertRaises(FileNotFoundError):
            ensure_vector_store_ready(str(self.base_dir))

    def test_get_knowledge_source_files_raises_when_data_dir_missing(self):
        with self.assertRaises(FileNotFoundError):
            get_knowledge_source_files(str(self.base_dir / "missing_data"))

    def test_vector_store_service_uses_injected_embedding_function(self):
        embedding = object()
        with patch("smart_clean_agent.rag.vector_store.Chroma") as mock_chroma:
            with patch("smart_clean_agent.rag.vector_store.create_embedding_model") as mock_factory:
                VectorStoreService(embedding_function=embedding)

        mock_factory.assert_not_called()
        self.assertEqual(mock_chroma.call_args.kwargs["embedding_function"], embedding)


if __name__ == "__main__":
    unittest.main()

