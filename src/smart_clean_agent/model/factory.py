from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from smart_clean_agent.utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])



def create_chat_model() -> BaseChatModel:
    model = ChatModelFactory().generator()
    if model is None:
        raise ValueError("聊天模型初始化失败")
    return model



def create_embedding_model() -> Embeddings:
    model = EmbeddingFactory().generator()
    if model is None:
        raise ValueError("Embedding 模型初始化失败")
    return model

