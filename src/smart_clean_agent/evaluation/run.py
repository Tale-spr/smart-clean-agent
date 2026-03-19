import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smart_clean_agent.agent.react_agent import ReactAgent
from smart_clean_agent.agent.tools.agent_tools import create_agent_tools
from smart_clean_agent.model.factory import create_chat_model, create_embedding_model
from smart_clean_agent.rag.rag_service import RagSummarizeService
from smart_clean_agent.rag.vector_store import VectorStoreService, ensure_vector_store_ready
from smart_clean_agent.evaluation.service import EvalCase, build_eval_trace, load_eval_cases, run_evaluation, write_evaluation_outputs


class AgentEvaluationExecutor:
    def __init__(self):
        ensure_vector_store_ready()
        chat_model = create_chat_model()
        embedding_model = create_embedding_model()
        vector_store_service = VectorStoreService(embedding_function=embedding_model)
        self.rag_service = RagSummarizeService(
            model=chat_model,
            vector_store_service=vector_store_service,
        )
        self.agent = ReactAgent(
            model=chat_model,
            tools=create_agent_tools(self.rag_service),
        )

    def execute_case(self, case: EvalCase):
        retrieved_docs_trace: list[dict[str, str]] = []

        def capture_retrieved_docs(query: str, docs):
            retrieved_docs_trace.clear()
            for doc in docs:
                metadata = doc.metadata or {}
                retrieved_docs_trace.append(
                    {
                        "query": query,
                        "source": str(metadata.get("source", "")),
                        "page": str(metadata.get("page", "")),
                        "snippet": doc.page_content[:120],
                    }
                )

        runtime_context = {
            "report": False,
            "user_id": case.user_id,
            "city": case.city,
            "session_summary": "",
            "recent_history": "",
            "trace_tool_calls": [],
        }

        self.rag_service.last_retrieved_docs = []
        self.rag_service.trace_callback = capture_retrieved_docs
        try:
            answer = self.agent.execute(case.query, runtime_context)
        finally:
            self.rag_service.trace_callback = None

        trace = build_eval_trace(
            tool_calls=runtime_context.get("trace_tool_calls", []),
            retrieved_docs=retrieved_docs_trace,
        )
        return answer, trace



def print_summary(summary, json_path: Path, csv_path: Path) -> None:
    print("离线评测完成")
    print(f"总样例数: {summary.total_cases}")
    print(f"answer_keyword_hit_rate: {summary.answer_keyword_hit_rate:.2%}")
    print(f"tool_call_success_rate: {summary.tool_call_success_rate:.2%}")
    print(f"retrieval_hit_rate: {summary.retrieval_hit_rate:.2%}")
    print(f"report_generation_success_rate: {summary.report_generation_success_rate:.2%}")
    print(f"JSON结果文件: {json_path}")
    print(f"CSV结果文件: {csv_path}")



def main(dataset_path: str | None = None, output_dir: str | None = None) -> int:
    try:
        cases = load_eval_cases(dataset_path)
        executor = AgentEvaluationExecutor()
        results, summary = run_evaluation(cases, executor.execute_case)
        json_path, csv_path = write_evaluation_outputs(results, summary, output_dir)
        print_summary(summary, json_path, csv_path)
        return 0
    except Exception as exc:
        print(f"离线评测失败: {str(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

