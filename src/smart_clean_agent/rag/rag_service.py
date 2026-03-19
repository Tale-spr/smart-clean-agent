"""
总结服务类:用户提问,搜索参考资料,将提问和参考资料提交给模型,让模型总结回复
"""
from typing import Callable

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from smart_clean_agent.model.factory import create_chat_model
from smart_clean_agent.rag.vector_store import VectorStoreService
from smart_clean_agent.utils.prompt_loader import load_rag_prompts



def print_prompt(prompt):
    print("=" * 20)
    print(prompt)
    print("=" * 20)
    return prompt


class RagSummarizeService:
    def __init__(
        self,
        model: BaseChatModel | None = None,
        vector_store_service: VectorStoreService | None = None,
    ):
        self.vector_store = vector_store_service or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = model or create_chat_model()
        self.last_retrieved_docs: list[Document] = []
        self.trace_callback: Callable[[str, list[Document]], None] | None = None
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        docs = self.retriever.invoke(query)
        self.last_retrieved_docs = docs
        if self.trace_callback is not None:
            self.trace_callback(query, docs)
        return docs

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        context_parts = []
        for index, doc in enumerate(context_docs, start=1):
            context_parts.append(
                f"【参考资料{index}】: {doc.page_content} | 参考元数据: {doc.metadata}"
            )

        return self.chain.invoke(
            {
                "input": query,
                "context": "\n".join(context_parts),
            }
        )


if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))

