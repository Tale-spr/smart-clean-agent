import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smart_clean_agent.model.factory import create_embedding_model
from smart_clean_agent.utils.config_handler import chroma_conf
from smart_clean_agent.utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, pdf_loader, txt_loader
from smart_clean_agent.utils.logger_handler import logger
from smart_clean_agent.utils.path_tool import get_abs_path

VECTOR_STORE_SQLITE = "chroma.sqlite3"



def get_vector_store_directory(persist_directory: str | None = None) -> Path:
    target = persist_directory or chroma_conf["persist_directory"]
    return Path(get_abs_path(target))



def get_vector_store_sqlite_path(persist_directory: str | None = None) -> Path:
    return get_vector_store_directory(persist_directory) / VECTOR_STORE_SQLITE



def vector_store_exists(persist_directory: str | None = None) -> bool:
    return get_vector_store_sqlite_path(persist_directory).exists()



def ensure_vector_store_ready(persist_directory: str | None = None) -> None:
    sqlite_path = get_vector_store_sqlite_path(persist_directory)
    if sqlite_path.exists():
        return

    raise FileNotFoundError(
        f"本地向量库不存在: {sqlite_path}. 请先运行 `python src/smart_clean_agent/rag/ingest.py` 完成知识库构建。"
    )



def get_knowledge_source_files(data_path: str | None = None) -> list[str]:
    target_data_path = get_abs_path(data_path or chroma_conf["data_path"])
    if not os.path.isdir(target_data_path):
        raise FileNotFoundError(f"知识库目录不存在: {target_data_path}")

    return list(
        listdir_with_allowed_type(
            target_data_path,
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )
    )


class VectorStoreService:
    def __init__(self, embedding_function: Embeddings | None = None):
        self.embedding_function = embedding_function or create_embedding_model()
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=self.embedding_function,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self) -> dict[str, int]:
        """
        从数据文件夹内读取数据文件,转为向量存入向量库。
        要计算文件的MD5做去重。
        """

        stats = {"scanned": 0, "loaded": 0, "skipped": 0, "failed": 0}

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_save: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path = get_knowledge_source_files()

        for path in allowed_files_path:
            stats["scanned"] += 1
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                stats["skipped"] += 1
                logger.info(f"[加载知识库]{path}内容已经存在知识库内,跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    stats["skipped"] += 1
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容,跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    stats["skipped"] += 1
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容,跳过")
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                stats["loaded"] += 1
                logger.info(f"[加载知识库]{path}内容加载成功")

            except Exception as e:
                stats["failed"] += 1
                logger.error(f"[加载知识库]{path}加载失败: {str(e)}", exc_info=True)

        return stats

