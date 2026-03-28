import unittest
from unittest.mock import patch

from smart_clean_agent.model.factory import (
    create_chat_model,
    create_embedding_model,
    get_chat_model_name,
    get_embedding_model_name,
)


class ModelFactoryTestCase(unittest.TestCase):
    def test_get_chat_model_name_reads_role_config(self):
        self.assertEqual(get_chat_model_name("primary_chat"), "qwen-plus")
        self.assertEqual(get_chat_model_name("batch_eval"), "qwen-flash")
        self.assertEqual(get_chat_model_name("judge"), "qwen-plus")
        self.assertEqual(get_chat_model_name("normalization"), "qwen-flash")

    def test_get_chat_model_name_rejects_unknown_role(self):
        with self.assertRaises(ValueError):
            get_chat_model_name("unknown")

    @patch("smart_clean_agent.model.factory.ChatTongyi")
    def test_create_chat_model_uses_role_default(self, mock_chat_tongyi):
        create_chat_model(role="batch_eval")
        mock_chat_tongyi.assert_called_once_with(model="qwen-flash")

    @patch("smart_clean_agent.model.factory.ChatTongyi")
    def test_create_chat_model_prefers_explicit_name(self, mock_chat_tongyi):
        create_chat_model(model_name="qwen-turbo", role="batch_eval")
        mock_chat_tongyi.assert_called_once_with(model="qwen-turbo")

    @patch("smart_clean_agent.model.factory.DashScopeEmbeddings")
    def test_create_embedding_model_uses_embedding_role(self, mock_embeddings):
        create_embedding_model()
        mock_embeddings.assert_called_once_with(model="text-embedding-v4")
        self.assertEqual(get_embedding_model_name(), "text-embedding-v4")


if __name__ == "__main__":
    unittest.main()
