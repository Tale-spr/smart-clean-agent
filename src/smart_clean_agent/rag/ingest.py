import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smart_clean_agent.rag.vector_store import VectorStoreService, get_knowledge_source_files



def main() -> int:
    files = get_knowledge_source_files()
    print(f"发现 {len(files)} 个可处理知识文件。")

    service = VectorStoreService()
    stats = service.load_document()
    print(
        "建库完成: "
        f"扫描 {stats['scanned']} 个文件, "
        f"新增 {stats['loaded']} 个文件, "
        f"跳过 {stats['skipped']} 个文件, "
        f"失败 {stats['failed']} 个文件。"
    )
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

