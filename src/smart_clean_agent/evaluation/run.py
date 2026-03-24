import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smart_clean_agent.agent.react_agent import ReactAgent
from smart_clean_agent.agent.tools.agent_tools import create_agent_tools
from smart_clean_agent.evaluation.judge import JudgeEvaluator
from smart_clean_agent.evaluation.service import (
    EvalCase,
    build_eval_trace,
    load_eval_cases,
    run_evaluation,
    write_evaluation_outputs,
)
from smart_clean_agent.model.factory import create_chat_model, create_embedding_model
from smart_clean_agent.rag.rag_service import RagSummarizeService
from smart_clean_agent.rag.vector_store import VectorStoreService, ensure_vector_store_ready


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行离线评测")
    parser.add_argument("--dataset", dest="dataset_path", help="JSONL 格式的评测数据集路径")
    parser.add_argument("--output-dir", dest="output_dir", help="评测结果输出目录")
    parser.add_argument("--with-judge", action="store_true", help="启用 LLM-as-a-Judge 评测")
    parser.add_argument("--chat-model", dest="chat_model_name", help="评测执行链路使用的模型名称")
    parser.add_argument("--judge-model", dest="judge_model_name", help="Judge 使用的模型名称")
    return parser.parse_args(argv)


def build_eval_runtime_context(case: EvalCase) -> dict:
    is_report_case = case.expected_route == "report"
    return {
        "report": is_report_case,
        "force_report_agent": is_report_case,
        "execution_mode": "",
        "user_id": case.user_id,
        "city": case.city,
        "session_summary": "",
        "recent_history": "",
        "trace_tool_calls": [],
        "react_trace": [],
        "react_step_count": 0,
        "react_stop_reason": "",
        "report_tool_sequence": [],
        "report_sequence_violation": False,
        "tool_evidence": [],
    }


class AgentEvaluationExecutor:
    def __init__(
        self,
        with_judge: bool = False,
        judge_model_name: str | None = None,
        chat_model_name: str | None = None,
    ):
        ensure_vector_store_ready()
        chat_model = create_chat_model(model_name=chat_model_name, role="batch_eval")
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
        self.judge = JudgeEvaluator(model_name=judge_model_name) if with_judge else None

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

        runtime_context = build_eval_runtime_context(case)

        self.rag_service.last_retrieved_docs = []
        self.rag_service.trace_callback = capture_retrieved_docs
        try:
            answer = self.agent.execute(case.query, runtime_context)
        finally:
            self.rag_service.trace_callback = None

        trace = build_eval_trace(
            tool_calls=runtime_context.get("trace_tool_calls", []),
            retrieved_docs=retrieved_docs_trace,
            tool_evidence=runtime_context.get("tool_evidence", []),
            step_count=runtime_context.get("react_step_count", 0),
            stop_reason=runtime_context.get("react_stop_reason", ""),
            execution_mode=runtime_context.get("execution_mode", "") or case.expected_route,
        )
        return answer, trace

    def judge_case(self, case: EvalCase, answer: str, trace):
        if self.judge is None:
            raise RuntimeError("Judge 未启用")
        return self.judge.evaluate(case, answer, trace)


def print_summary(rule_summary, judge_summary, json_path: Path, csv_path: Path) -> None:
    def _format_percent(value):
        return "n/a" if value is None else f"{value:.2%}"

    def _format_number(value):
        return "n/a" if value is None else f"{value:.2f}"

    print("离线评测完成")
    print(f"总样例数: {rule_summary.total_cases}")
    print(f"rule.content_pass_rate: {rule_summary.content_pass_rate:.2%}")
    print(f"rule.route_correct_rate: {rule_summary.route_correct_rate:.2%}")
    print(f"rule.required_tool_pass_rate: {rule_summary.required_tool_pass_rate:.2%}")
    print(f"rule.tool_sequence_valid_rate: {rule_summary.tool_sequence_valid_rate:.2%}")
    print(f"rule.retrieval_mode_valid_rate: {rule_summary.retrieval_mode_valid_rate:.2%}")
    print(f"rule.required_point_hit_rate_avg: {rule_summary.required_point_hit_rate_avg:.2%}")
    print(f"rule.report_generation_success_rate: {rule_summary.report_generation_success_rate:.2%}")
    if judge_summary.enabled:
        print(f"judge.judge_pass_rate: {_format_percent(judge_summary.judge_pass_rate)}")
        print(f"judge.avg_correctness_score: {_format_number(judge_summary.avg_correctness_score)}")
        print(f"judge.avg_completeness_score: {_format_number(judge_summary.avg_completeness_score)}")
    else:
        print("judge: disabled")
    print(f"JSON结果文件: {json_path}")
    print(f"CSV结果文件: {csv_path}")


def main(
    dataset_path: str | None = None,
    output_dir: str | None = None,
    with_judge: bool = False,
    judge_model_name: str | None = None,
    chat_model_name: str | None = None,
    argv: list[str] | None = None,
) -> int:
    if argv is not None:
        args = parse_args(argv)
        dataset_path = args.dataset_path
        output_dir = args.output_dir
        with_judge = args.with_judge
        judge_model_name = args.judge_model_name
        chat_model_name = args.chat_model_name

    try:
        cases = load_eval_cases(dataset_path)
        executor = AgentEvaluationExecutor(
            with_judge=with_judge,
            judge_model_name=judge_model_name,
            chat_model_name=chat_model_name,
        )
        results, rule_summary, judge_summary, normal_summary, report_summary = run_evaluation(
            cases,
            executor.execute_case,
            judge=executor.judge_case if with_judge else None,
        )
        json_path, csv_path = write_evaluation_outputs(
            results,
            rule_summary,
            judge_summary,
            normal_summary,
            report_summary,
            output_dir,
        )
        print_summary(rule_summary, judge_summary, json_path, csv_path)
        print(f"normal.content_pass_rate: {normal_summary.content_pass_rate:.2%}")
        print(f"normal.tool_usage_valid_rate: {normal_summary.tool_usage_valid_rate:.2%}")
        print(f"report.tool_dependency_valid_rate: {report_summary.tool_dependency_valid_rate:.2%}")
        print(f"report.time_consistency_valid_rate: {report_summary.time_consistency_valid_rate:.2%}")
        return 0
    except Exception as exc:
        print(f"离线评测失败: {str(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(argv=sys.argv[1:]))
