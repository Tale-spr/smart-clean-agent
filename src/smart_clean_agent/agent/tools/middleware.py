from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from smart_clean_agent.services.status_event_service import record_status_event
from smart_clean_agent.utils.logger_handler import logger
from smart_clean_agent.utils.prompt_loader import load_report_prompts, load_system_prompts

REPORT_TOOL_SEQUENCE = (
    "get_user_id",
    "get_current_month",
    "fill_context_for_report",
    "fetch_external_data",
    "fetch_external_history",
)


def _record_report_tool_sequence(context: dict, tool_name: str) -> None:
    if tool_name not in REPORT_TOOL_SEQUENCE:
        return

    sequence = context.setdefault("report_tool_sequence", [])
    if not isinstance(sequence, list):
        sequence = []
        context["report_tool_sequence"] = sequence

    required_predecessors: dict[str, tuple[str, ...]] = {
        "fill_context_for_report": ("get_user_id", "get_current_month"),
        "fetch_external_data": ("fill_context_for_report",),
        "fetch_external_history": ("fill_context_for_report",),
    }
    missing = [item for item in required_predecessors.get(tool_name, ()) if item not in sequence]
    if missing:
        context["report_sequence_violation"] = True
        logger.warning(
            "[报告工具顺序]工具%s在缺少前置步骤%s时被调用",
            tool_name,
            ",".join(missing),
        )

    sequence.append(tool_name)


@wrap_tool_call
def monitor_tool(  # 工具执行的监控
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    logger.info(f"[工具调用]执行工具: {tool_name}")
    logger.info(f"[工具调用]传入参数: {request.tool_call['args']}")

    trace_tool_calls = request.runtime.context.get("trace_tool_calls")
    if isinstance(trace_tool_calls, list):
        trace_tool_calls.append(tool_name)

    if request.runtime.context.get("force_report_agent") or request.runtime.context.get("report"):
        _record_report_tool_sequence(request.runtime.context, tool_name)

    if tool_name == "rag_summarize":
        record_status_event(
            request.runtime.context,
            event_type="stage.rag",
            title="正在调用知识库",
            detail="正在检索扫地机器人相关知识库资料",
        )
    elif tool_name == "fetch_external_history":
        record_status_event(
            request.runtime.context,
            event_type="stage.trend",
            title="正在汇总多月趋势",
            detail="正在整理最近多个月的使用趋势记录",
        )
    elif tool_name == "fill_context_for_report":
        record_status_event(
            request.runtime.context,
            event_type="stage.report",
            title="正在生成使用报告",
            detail="正在切换到报告生成上下文",
        )
    else:
        record_status_event(
            request.runtime.context,
            event_type="stage.tool",
            title="正在调用工具",
            detail=f"正在调用 {tool_name}",
        )

    try:
        result = handler(request)
        logger.info(f"[工具调用]工具{tool_name}调用成功")
        record_status_event(
            request.runtime.context,
            event_type="tool.success",
            title="工具调用完成",
            detail=f"{tool_name} 调用成功",
        )
        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.info(f"[工具调用]工具{tool_name}调用失败,原因: {str(e)}")
        record_status_event(
            request.runtime.context,
            event_type="error.tool",
            title="工具调用失败",
            detail=f"{tool_name} 调用失败: {str(e)}",
            level="error",
        )
        raise e


@before_model
def log_before_model(
        state: AgentState,
        runtime: Runtime,
):
    logger.info(f"[模型调用]即将调用模型, 带有{len(state['messages'])}条消息.")

    logger.debug(f"[模型调用] {type(state['messages'][-1])} | {state['messages'][-1].content.strip()}")
    title = "正在生成使用报告" if runtime.context.get("report", False) else "正在分析问题"
    detail = "正在整合当前问题、上下文与工具结果" if runtime.context.get("report", False) else "正在判断是否需要检索知识库或调用工具"
    record_status_event(
        runtime.context,
        event_type="stage.model",
        title=title,
        detail=detail,
    )

    return None



def build_runtime_context_prompt(context: dict, is_report: bool = False) -> str:
    session_summary = (context.get("session_summary") or "").strip()
    recent_history = (context.get("recent_history") or "").strip()
    user_memory_summary = (context.get("user_memory_summary") or "").strip()
    report_memory_summary = (context.get("report_memory_summary") or "").strip()
    if not session_summary and not recent_history and not user_memory_summary and (not is_report or not report_memory_summary):
        return ""

    title = "### 最近会话上下文" if is_report else "### 当前会话上下文"
    sections = [title]
    if session_summary:
        sections.append(f"当前会话摘要:\n{session_summary}")
    if recent_history:
        sections.append(f"最近几轮对话:\n{recent_history}")
    if user_memory_summary:
        sections.append(f"用户长期记忆:\n{user_memory_summary}")
    if is_report and report_memory_summary:
        sections.append(f"报告趋势记忆:\n{report_memory_summary}")

    sections.append("请结合以上上下文保持回答连续性；如果当前用户问题与上下文冲突，以当前用户最新问题为准。")
    return "\n\n".join(sections)


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    base_prompt = load_report_prompts() if is_report else load_system_prompts()
    context_prompt = build_runtime_context_prompt(request.runtime.context, is_report=is_report)
    if not context_prompt:
        return base_prompt

    return f"{base_prompt}\n\n{context_prompt}"

