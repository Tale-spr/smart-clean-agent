import re
from datetime import datetime
from typing import Any, NotRequired, TypedDict

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from smart_clean_agent.agent.runtime_context import AgentRuntimeContext
from smart_clean_agent.agent.tools.agent_tools import get_user_location_from_context
from smart_clean_agent.agent.tools.middleware import build_runtime_context_prompt, record_report_tool_sequence
from smart_clean_agent.services.status_event_service import record_status_event
from smart_clean_agent.utils.logger_handler import logger
from smart_clean_agent.utils.prompt_loader import load_report_prompts, load_system_prompts

MAX_REACT_STEPS = 5
STOP_REASONS = {"enough_information", "max_steps_reached", "tool_failed", "unsupported_request", "consistency_failed"}
REPORT_KEYWORDS = ("报告", "月报", "使用记录", "趋势", "统计")
REPORT_HISTORY_KEYWORDS = ("趋势", "变化", "对比", "最近", "近几月", "多月", "长期", "历史")
REPORT_KNOWLEDGE_KEYWORDS = ("保养", "维护", "建议", "耗材", "更换", "风险", "提升", "优化", "注意")
WEATHER_KEYWORDS = ("天气", "气温", "温度", "湿度", "下雨", "降雨", "空气", "潮湿", "干燥", "回南天", "梅雨天", "回南", "梅雨")
DOMAIN_KEYWORDS = ("扫地机器人", "扫拖", "机器人", "清洁", "拖地", "湿拖", "吸力", "滚刷", "滤网", "水箱", "避障", "回充", "漏扫", "地图", "导航", "保养", "维护", "地板", "木地板", "地毯", "瓷砖", "滚刷更换", "主刷", "边刷", "拖布", "HEPA", "粉尘", "WiFi", "门槛", "水痕")
SMALLTALK_KEYWORDS = ("你好", "您好", "谢谢", "感谢", "你是谁", "再见")
IMPLICIT_LOCATION_KEYWORDS = ("我所在城市", "当前城市", "现在这里", "我这边", "所在城市")
ENVIRONMENT_DECISION_KEYWORDS = ("适不适合", "适合", "能不能", "要不要", "会不会影响", "有什么影响")
WEATHER_KNOWLEDGE_KEYWORDS = ("保养", "存放", "耗材", "回充", "导航", "地图", "避障", "滤网", "滚刷", "拖布", "出水量", "水箱", "故障")
CITY_PATTERN = re.compile(r"(?P<city>[\u4e00-\u9fa5]{2,6})(?:今天|现在|当前|这几天)?(?:的)?(?:天气|温度|湿度|空气|下雨|降雨)")
NON_CITY_TOKENS = {"今天", "今日", "现在", "当前", "这几天", "最近"}
YEAR_MONTH_PATTERN = re.compile(r"(?P<year>\d{4})\s*(?:年|/|-)\s*(?P<month>\d{1,2})\s*月?")


class ReActGraphState(TypedDict):
    query: str
    runtime_context: AgentRuntimeContext
    intent: str
    known_facts: dict[str, str]
    tool_history: list[dict[str, Any]]
    observations: list[dict[str, str]]
    remaining_questions: list[str]
    selected_tool: NotRequired[str | None]
    selected_tool_args: NotRequired[dict[str, Any]]
    last_tool_result: NotRequired[str]
    last_tool_error: NotRequired[str]
    final_answer: str
    step_count: int
    stop_reason: str


class ReportGraphState(TypedDict):
    query: str
    runtime_context: AgentRuntimeContext
    report_target_month: str
    user_id: str
    needs_history: bool
    needs_domain_knowledge: bool
    known_facts: dict[str, str]
    tool_history: list[dict[str, Any]]
    observations: list[dict[str, str]]
    missing_inputs: list[str]
    selected_tool: NotRequired[str | None]
    selected_tool_args: NotRequired[dict[str, Any]]
    last_tool_result: NotRequired[str]
    last_tool_error: NotRequired[str]
    final_report: str
    step_count: int
    stop_reason: str


class ReactAgent:
    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.chat_model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.normal_graph = self._build_normal_graph()
        self.report_graph = self._build_report_graph()

    def execute_stream(self, query: str, runtime_context: AgentRuntimeContext):
        if runtime_context.get("force_report_agent") or self._is_report_query(query):
            runtime_context["execution_mode"] = "report"
            final_report = self._run_report_graph(query, runtime_context)
            if final_report:
                yield final_report
            return

        runtime_context["execution_mode"] = "normal"
        final_answer = self._run_normal_graph(query, runtime_context)
        if final_answer:
            yield final_answer

    def execute(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        return "".join(self.execute_stream(query, runtime_context)).strip()

    def _build_normal_graph(self):
        graph = StateGraph(ReActGraphState)
        graph.add_node("analyze_question", self._analyze_question)
        graph.add_node("select_tool_or_finish", self._select_tool_or_finish)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("observe_tool_result", self._observe_tool_result)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_edge(START, "analyze_question")
        graph.add_edge("analyze_question", "select_tool_or_finish")
        graph.add_conditional_edges("select_tool_or_finish", self._route_after_selection, {"execute_tool": "execute_tool", "generate_answer": "generate_answer"})
        graph.add_edge("execute_tool", "observe_tool_result")
        graph.add_edge("observe_tool_result", "select_tool_or_finish")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    def _build_report_graph(self):
        graph = StateGraph(ReportGraphState)
        graph.add_node("analyze_report_request", self._analyze_report_request)
        graph.add_node("select_report_tool_or_finish", self._select_report_tool_or_finish)
        graph.add_node("execute_report_tool", self._execute_report_tool)
        graph.add_node("observe_report_result", self._observe_report_result)
        graph.add_node("generate_report", self._generate_report)
        graph.add_node("validate_report_consistency", self._validate_report_consistency)
        graph.add_edge(START, "analyze_report_request")
        graph.add_edge("analyze_report_request", "select_report_tool_or_finish")
        graph.add_conditional_edges("select_report_tool_or_finish", self._route_report_after_selection, {"execute_report_tool": "execute_report_tool", "generate_report": "generate_report"})
        graph.add_edge("execute_report_tool", "observe_report_result")
        graph.add_edge("observe_report_result", "select_report_tool_or_finish")
        graph.add_edge("generate_report", "validate_report_consistency")
        graph.add_edge("validate_report_consistency", END)
        return graph.compile()

    def _run_normal_graph(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        runtime_context["execution_mode"] = "normal"
        runtime_context.setdefault("trace_tool_calls", [])
        runtime_context.setdefault("react_trace", [])
        initial_state: ReActGraphState = {"query": query, "runtime_context": runtime_context, "intent": "", "known_facts": {}, "tool_history": [], "observations": [], "remaining_questions": [], "final_answer": "", "step_count": 0, "stop_reason": ""}
        try:
            final_state = self.normal_graph.invoke(initial_state)
            runtime_context["react_stop_reason"] = final_state.get("stop_reason", "")
            runtime_context["react_step_count"] = final_state.get("step_count", 0)
            record_status_event(runtime_context, event_type="smart_clean_agent.agent.complete", title="回答生成完成", detail="已完成最终回答生成")
            return final_state.get("final_answer", "").strip()
        except Exception as exc:
            runtime_context["react_stop_reason"] = "tool_failed"
            record_status_event(runtime_context, event_type="error.agent", title="回答生成失败", detail=str(exc), level="error")
            raise

    def _run_report_graph(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        runtime_context["execution_mode"] = "report"
        runtime_context["report"] = True
        runtime_context.setdefault("trace_tool_calls", [])
        runtime_context.setdefault("react_trace", [])
        runtime_context.setdefault("report_tool_sequence", [])
        runtime_context.setdefault("status_events", [])
        initial_state: ReportGraphState = {"query": query, "runtime_context": runtime_context, "report_target_month": "", "user_id": "", "needs_history": False, "needs_domain_knowledge": False, "known_facts": {}, "tool_history": [], "observations": [], "missing_inputs": [], "final_report": "", "step_count": 0, "stop_reason": ""}
        try:
            final_state = self.report_graph.invoke(initial_state)
            runtime_context["react_stop_reason"] = final_state.get("stop_reason", "")
            runtime_context["react_step_count"] = final_state.get("step_count", 0)
            record_status_event(runtime_context, event_type="smart_clean_agent.agent.complete", title="回答生成完成", detail="已完成最终回答生成")
            return final_state.get("final_report", "").strip()
        except Exception as exc:
            runtime_context["react_stop_reason"] = "tool_failed"
            record_status_event(runtime_context, event_type="error.agent", title="回答生成失败", detail=str(exc), level="error")
            raise

    def _analyze_question(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        query = state["query"].strip()
        record_status_event(runtime_context, event_type="stage.model", title="正在分析问题", detail="正在判断是否需要检索知识库或调用工具")

        weather_needed = self._needs_weather(query)
        knowledge_needed = self._needs_knowledge(query)
        direct_reply = self._is_smalltalk(query)
        known_facts: dict[str, str] = {}
        explicit_city = self._extract_city(query)
        if explicit_city:
            known_facts["city"] = explicit_city

        if direct_reply and not weather_needed and not knowledge_needed:
            intent = "direct"
            remaining_questions: list[str] = []
            stop_reason = "enough_information"
        elif weather_needed and knowledge_needed:
            intent = "combined"
            remaining_questions = []
            if "city" not in known_facts:
                remaining_questions.append("city")
            remaining_questions.extend(["weather", "knowledge"])
            stop_reason = ""
        elif weather_needed:
            intent = "weather"
            remaining_questions = []
            if "city" not in known_facts:
                remaining_questions.append("city")
            remaining_questions.append("weather")
            stop_reason = ""
        elif knowledge_needed:
            intent = "knowledge"
            remaining_questions = ["knowledge"]
            stop_reason = ""
        else:
            intent = "unsupported"
            remaining_questions = []
            stop_reason = "unsupported_request"

        self._append_react_trace(runtime_context, phase="analyze", content=f"intent={intent}; remaining={','.join(remaining_questions) or '-'}")
        return {"intent": intent, "known_facts": known_facts, "remaining_questions": remaining_questions, "stop_reason": stop_reason}

    def _select_tool_or_finish(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        step_count = state.get("step_count", 0)
        remaining_questions = list(state.get("remaining_questions", []))
        stop_reason = state.get("stop_reason", "")

        if stop_reason in {"unsupported_request", "tool_failed"}:
            self._append_react_trace(runtime_context, phase="decide", content=f"finish:{stop_reason}")
            return {"selected_tool": None, "selected_tool_args": {}}
        if step_count >= MAX_REACT_STEPS:
            self._append_react_trace(runtime_context, phase="decide", content="finish:max_steps_reached")
            return {"selected_tool": None, "selected_tool_args": {}, "stop_reason": "max_steps_reached"}
        if not remaining_questions:
            self._append_react_trace(runtime_context, phase="decide", content="finish:enough_information")
            return {"selected_tool": None, "selected_tool_args": {}, "stop_reason": stop_reason or "enough_information"}

        next_gap = remaining_questions[0]
        if next_gap == "city":
            decision = ("get_user_location", {})
        elif next_gap == "weather":
            decision = ("get_weather", {"city": state.get("known_facts", {}).get("city", "")})
        else:
            decision = ("rag_summarize", {"query": self._build_rag_query(state["query"])})

        self._append_react_trace(runtime_context, phase="decide", content=f"action={decision[0]}; gap={next_gap}")
        return {"selected_tool": decision[0], "selected_tool_args": decision[1]}

    def _route_after_selection(self, state: ReActGraphState) -> str:
        return "execute_tool" if state.get("selected_tool") else "generate_answer"

    def _execute_tool(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        tool_name = state.get("selected_tool")
        tool_args = dict(state.get("selected_tool_args", {}))
        if not tool_name:
            return {}

        runtime_context.setdefault("trace_tool_calls", []).append(tool_name)
        self._record_tool_stage_event(runtime_context, tool_name)
        self._append_react_trace(runtime_context, phase="act", content=f"{tool_name}({tool_args})")

        tool_history = list(state.get("tool_history", []))
        tool_history.append({"tool_name": tool_name, "tool_args": tool_args, "status": "running"})
        try:
            result = self._invoke_tool(tool_name, tool_args, runtime_context)
            tool_history[-1]["status"] = "success"
            record_status_event(runtime_context, event_type="tool.success", title="工具调用完成", detail=f"{tool_name} 调用成功")
            return {"tool_history": tool_history, "last_tool_result": result, "last_tool_error": "", "step_count": state.get("step_count", 0) + 1}
        except Exception as exc:
            logger.warning(f"[ReAct]工具调用失败: {tool_name} | {str(exc)}")
            tool_history[-1]["status"] = "failed"
            record_status_event(runtime_context, event_type="error.tool", title="工具调用失败", detail=f"{tool_name} 调用失败: {str(exc)}", level="error")
            return {"tool_history": tool_history, "last_tool_result": "", "last_tool_error": str(exc), "step_count": state.get("step_count", 0) + 1, "stop_reason": "tool_failed"}

    def _observe_tool_result(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        tool_name = state.get("selected_tool")
        tool_result = (state.get("last_tool_result") or "").strip()
        tool_error = (state.get("last_tool_error") or "").strip()
        known_facts = dict(state.get("known_facts", {}))
        remaining_questions = list(state.get("remaining_questions", []))
        observations = list(state.get("observations", []))
        if not tool_name:
            return {}
        if tool_error:
            self._append_react_trace(runtime_context, phase="observe", content=f"{tool_name}:error")
            observations.append({"tool_name": tool_name, "summary": f"{tool_name} 调用失败"})
            return {"observations": observations, "stop_reason": "tool_failed"}

        observation_summary = self._summarize_tool_result(tool_name, tool_result)
        observations.append({"tool_name": tool_name, "summary": observation_summary})
        self._append_react_trace(runtime_context, phase="observe", content=f"{tool_name}:{observation_summary}")

        if tool_name == "get_user_location":
            if tool_result.startswith("当前会话未配置"):
                return {"observations": observations, "stop_reason": "tool_failed"}
            known_facts["city"] = tool_result
            remaining_questions = self._remove_gap(remaining_questions, "city")
        elif tool_name == "get_weather":
            if self._is_unusable_tool_result(tool_result):
                return {"observations": observations, "stop_reason": "tool_failed"}
            known_facts["weather_info"] = tool_result
            remaining_questions = self._remove_gap(remaining_questions, "weather")
        elif tool_name == "rag_summarize":
            if self._is_unusable_tool_result(tool_result):
                return {"observations": observations, "stop_reason": "tool_failed"}
            known_facts["knowledge_info"] = tool_result
            remaining_questions = self._remove_gap(remaining_questions, "knowledge")

        return {"known_facts": known_facts, "remaining_questions": remaining_questions, "observations": observations, "last_tool_result": "", "last_tool_error": ""}

    def _generate_answer(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        stop_reason = state.get("stop_reason", "") or "enough_information"
        runtime_context["react_stop_reason"] = stop_reason
        runtime_context["react_step_count"] = state.get("step_count", 0)
        record_status_event(runtime_context, event_type="stage.final", title="正在生成最终建议", detail="正在整理工具结果、知识库内容与上下文信息")

        known_facts = state.get("known_facts", {})
        if stop_reason == "unsupported_request" and not known_facts:
            answer = "我不知道"
        elif stop_reason in {"tool_failed", "max_steps_reached"} and not known_facts:
            answer = "我不知道"
        else:
            answer = self._build_final_answer(state)

        self._append_react_trace(runtime_context, phase="finish", content=f"stop_reason={stop_reason}")
        return {"final_answer": answer}

    def _analyze_report_request(self, state: ReportGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        query = state["query"].strip()
        runtime_context["report"] = True
        record_status_event(runtime_context, event_type="stage.report", title="正在生成使用报告", detail="正在分析报告需求并准备数据依赖")

        known_facts: dict[str, str] = {}
        missing_inputs: list[str] = []
        user_id = (runtime_context.get("user_id") or "").strip()
        if user_id:
            known_facts["user_id"] = user_id
        else:
            missing_inputs.append("user_id")

        target_month = self._extract_report_month(query)
        if target_month:
            known_facts["report_target_month"] = target_month
            runtime_context["report_current_month"] = target_month
        else:
            missing_inputs.append("month")
        missing_inputs.append("monthly_record")

        needs_history = self._needs_report_history(query)
        needs_domain_knowledge = self._needs_report_knowledge(query)
        self._append_react_trace(
            runtime_context,
            phase="analyze",
            content=(
                f"report_target_month={known_facts.get('report_target_month', '-')}; "
                f"needs_history={needs_history}; needs_domain_knowledge={needs_domain_knowledge}; "
                f"missing={','.join(missing_inputs)}"
            ),
        )
        return {
            "report_target_month": known_facts.get("report_target_month", ""),
            "user_id": known_facts.get("user_id", ""),
            "needs_history": needs_history,
            "needs_domain_knowledge": needs_domain_knowledge,
            "known_facts": known_facts,
            "missing_inputs": missing_inputs,
            "stop_reason": "",
        }

    def _select_report_tool_or_finish(self, state: ReportGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        step_count = state.get("step_count", 0)
        missing_inputs = list(state.get("missing_inputs", []))
        stop_reason = state.get("stop_reason", "")
        known_facts = state.get("known_facts", {})

        if stop_reason in {"unsupported_request", "tool_failed", "consistency_failed"}:
            self._append_react_trace(runtime_context, phase="decide", content=f"finish:{stop_reason}")
            return {"selected_tool": None, "selected_tool_args": {}}
        if step_count >= MAX_REACT_STEPS:
            self._append_react_trace(runtime_context, phase="decide", content="finish:max_steps_reached")
            return {"selected_tool": None, "selected_tool_args": {}, "stop_reason": "max_steps_reached"}

        if "user_id" in missing_inputs:
            decision = ("get_user_id", {})
            gap = "user_id"
        elif "month" in missing_inputs:
            decision = ("get_current_month", {})
            gap = "month"
        elif "monthly_record" in missing_inputs:
            user_id = known_facts.get("user_id", "")
            report_target_month = known_facts.get("report_target_month", "")
            if not user_id or not report_target_month:
                self._append_react_trace(runtime_context, phase="decide", content="finish:tool_failed")
                return {"selected_tool": None, "selected_tool_args": {}, "stop_reason": "tool_failed"}
            decision = ("fetch_external_data", {"user_id": user_id, "month": report_target_month})
            gap = "monthly_record"
        elif state.get("needs_history") and "history_info" not in known_facts:
            decision = ("fetch_external_history", {"user_id": known_facts.get("user_id", ""), "months": 3})
            gap = "history_info"
        elif state.get("needs_domain_knowledge") and "knowledge_info" not in known_facts:
            decision = ("rag_summarize", {"query": self._build_report_rag_query(state)})
            gap = "knowledge_info"
        else:
            self._append_react_trace(runtime_context, phase="decide", content="finish:enough_information")
            return {"selected_tool": None, "selected_tool_args": {}, "stop_reason": "enough_information"}

        self._append_react_trace(runtime_context, phase="decide", content=f"action={decision[0]}; gap={gap}")
        return {"selected_tool": decision[0], "selected_tool_args": decision[1]}

    def _route_report_after_selection(self, state: ReportGraphState) -> str:
        return "execute_report_tool" if state.get("selected_tool") else "generate_report"

    def _execute_report_tool(self, state: ReportGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        tool_name = state.get("selected_tool")
        tool_args = dict(state.get("selected_tool_args", {}))
        if not tool_name:
            return {}

        runtime_context.setdefault("trace_tool_calls", []).append(tool_name)
        record_report_tool_sequence(runtime_context, tool_name)
        self._record_tool_stage_event(runtime_context, tool_name)
        self._append_react_trace(runtime_context, phase="act", content=f"{tool_name}({tool_args})")

        tool_history = list(state.get("tool_history", []))
        tool_history.append({"tool_name": tool_name, "tool_args": tool_args, "status": "running"})
        try:
            result = self._invoke_tool(tool_name, tool_args, runtime_context)
            tool_history[-1]["status"] = "success"
            record_status_event(runtime_context, event_type="tool.success", title="工具调用完成", detail=f"{tool_name} 调用成功")
            return {"tool_history": tool_history, "last_tool_result": result, "last_tool_error": "", "step_count": state.get("step_count", 0) + 1}
        except Exception as exc:
            logger.warning(f"[ReportReAct]工具调用失败: {tool_name} | {str(exc)}")
            tool_history[-1]["status"] = "failed"
            record_status_event(runtime_context, event_type="error.tool", title="工具调用失败", detail=f"{tool_name} 调用失败: {str(exc)}", level="error")
            return {"tool_history": tool_history, "last_tool_result": "", "last_tool_error": str(exc), "step_count": state.get("step_count", 0) + 1, "stop_reason": "tool_failed"}

    def _observe_report_result(self, state: ReportGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        tool_name = state.get("selected_tool")
        tool_result = (state.get("last_tool_result") or "").strip()
        tool_error = (state.get("last_tool_error") or "").strip()
        known_facts = dict(state.get("known_facts", {}))
        missing_inputs = list(state.get("missing_inputs", []))
        observations = list(state.get("observations", []))
        if not tool_name:
            return {}
        if tool_error:
            self._append_react_trace(runtime_context, phase="observe", content=f"{tool_name}:error")
            observations.append({"tool_name": tool_name, "summary": f"{tool_name} 调用失败"})
            return {"observations": observations, "stop_reason": "tool_failed"}

        observation_summary = self._summarize_tool_result(tool_name, tool_result)
        observations.append({"tool_name": tool_name, "summary": observation_summary})
        self._append_react_trace(runtime_context, phase="observe", content=f"{tool_name}:{observation_summary}")

        if tool_name == "get_user_id":
            if self._is_unusable_tool_result(tool_result):
                return {"observations": observations, "stop_reason": "tool_failed"}
            known_facts["user_id"] = tool_result
            missing_inputs = self._remove_gap(missing_inputs, "user_id")
        elif tool_name == "get_current_month":
            normalized_month = self._normalize_report_month(tool_result)
            if not normalized_month:
                return {"observations": observations, "stop_reason": "tool_failed"}
            known_facts["report_target_month"] = normalized_month
            runtime_context["report_current_month"] = normalized_month
            missing_inputs = self._remove_gap(missing_inputs, "month")
        elif tool_name == "fetch_external_data":
            if self._is_unusable_tool_result(tool_result):
                return {"observations": observations, "stop_reason": "tool_failed"}
            known_facts["monthly_record"] = tool_result
            missing_inputs = self._remove_gap(missing_inputs, "monthly_record")
        elif tool_name == "fetch_external_history":
            known_facts["history_info"] = tool_result
        elif tool_name == "rag_summarize":
            known_facts["knowledge_info"] = tool_result

        return {"known_facts": known_facts, "missing_inputs": missing_inputs, "observations": observations, "last_tool_result": "", "last_tool_error": ""}

    def _generate_report(self, state: ReportGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        stop_reason = state.get("stop_reason", "") or "enough_information"
        runtime_context["react_stop_reason"] = stop_reason
        runtime_context["react_step_count"] = state.get("step_count", 0)
        record_status_event(runtime_context, event_type="stage.final", title="正在生成使用报告", detail="正在整理本月记录、趋势信息与建议内容")

        known_facts = state.get("known_facts", {})
        monthly_record = (known_facts.get("monthly_record") or "").strip()
        if stop_reason in {"tool_failed", "max_steps_reached", "unsupported_request"} and not monthly_record:
            report = "暂时无法生成可靠的使用报告，请稍后重试。"
        else:
            report = self._build_final_report(state)

        self._append_react_trace(runtime_context, phase="finish", content=f"stop_reason={stop_reason}")
        return {"final_report": report}

    def _validate_report_consistency(self, state: ReportGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        final_report = (state.get("final_report") or "").strip()
        stop_reason = state.get("stop_reason", "") or "enough_information"
        target_month = state.get("known_facts", {}).get("report_target_month") or state.get("report_target_month", "")
        if final_report and target_month and not self._is_report_time_consistent(final_report, target_month):
            record_status_event(runtime_context, event_type="error.report", title="报告校验失败", detail="报告中的时间信息与目标月份不一致", level="error")
            self._append_react_trace(runtime_context, phase="validate", content="consistency_failed")
            return {"stop_reason": "consistency_failed"}
        self._append_react_trace(runtime_context, phase="validate", content="consistency_passed")
        return {"stop_reason": stop_reason}

    def _build_final_answer(self, state: ReActGraphState) -> str:
        runtime_context = state["runtime_context"]
        query = state["query"]
        known_facts = state.get("known_facts", {})
        observations = state.get("observations", [])
        stop_reason = state.get("stop_reason", "") or "enough_information"
        evidence_sections: list[str] = []
        if known_facts.get("city"):
            evidence_sections.append(f"用户城市: {known_facts['city']}")
        if known_facts.get("weather_info"):
            evidence_sections.append(f"天气信息:\n{known_facts['weather_info']}")
        if known_facts.get("knowledge_info"):
            evidence_sections.append(f"知识库信息:\n{known_facts['knowledge_info']}")
        if observations:
            evidence_sections.append("观察摘要:\n" + "\n".join(f"- {item['tool_name']}: {item['summary']}" for item in observations))

        context_prompt = build_runtime_context_prompt(runtime_context, is_report=False)
        focus_terms = self._extract_focus_terms(query, known_facts)
        human_sections = ["请基于以下已知信息，为用户生成最终中文回答。", f"用户问题:\n{query}", f"停止原因: {stop_reason}"]
        if context_prompt:
            human_sections.append(context_prompt)
        if evidence_sections:
            human_sections.append("已知信息:\n" + "\n\n".join(evidence_sections))
        else:
            human_sections.append("当前没有可用的外部信息，请仅在你有把握时回答，否则回复“我不知道”。")
        if focus_terms:
            human_sections.append(f"回答时请自然保留这些关键术语：{', '.join(focus_terms)}。")
        human_sections.append("要求：使用“结论 + 依据 + 建议”的最小完整结构；回答保持简洁，但不要遗漏用户问题中的关键部件名、场景名、天气要素或建议动作。")
        human_sections.append("只输出最终回答；不要暴露内部推理、步骤、工具名或中间分析。")
        return self._invoke_model(load_system_prompts(), human_sections)

    def _build_final_report(self, state: ReportGraphState) -> str:
        runtime_context = state["runtime_context"]
        query = state["query"]
        known_facts = state.get("known_facts", {})
        observations = state.get("observations", [])
        context_prompt = build_runtime_context_prompt(runtime_context, is_report=True)

        human_sections = ["请根据以下结构化信息生成最终中文报告。", f"用户请求:\n{query}"]
        if known_facts.get("report_target_month"):
            human_sections.append(f"目标月份: {known_facts['report_target_month']}")
        if context_prompt:
            human_sections.append(context_prompt)
        if known_facts.get("monthly_record"):
            human_sections.append(f"本月记录:\n{known_facts['monthly_record']}")
        if known_facts.get("history_info"):
            human_sections.append(f"趋势信息:\n{known_facts['history_info']}")
        if known_facts.get("knowledge_info"):
            human_sections.append(f"专业建议参考:\n{known_facts['knowledge_info']}")
        if observations:
            human_sections.append("观察摘要:\n" + "\n".join(f"- {item['tool_name']}: {item['summary']}" for item in observations))
        human_sections.append("要求：先给出该月主结论；只有在明确拿到趋势信息时，才补充最近多月变化和长期建议；不要编造具体数值、月份或趋势；如果某部分信息不足，请明确说明信息范围有限。")
        human_sections.append("只输出最终报告，不要展示内部推理、工具名或中间分析。")
        return self._invoke_model(load_report_prompts(), human_sections)

    def _invoke_model(self, system_prompt: str, human_sections: list[str]) -> str:
        response = self.chat_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content="\n\n".join(section for section in human_sections if section))])
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(str(item) for item in content).strip()
        return str(content).strip()

    def _invoke_tool(self, tool_name: str, tool_args: dict[str, Any], runtime_context: AgentRuntimeContext) -> str:
        if tool_name == "get_user_location":
            return get_user_location_from_context(runtime_context)
        tool = self.tool_map.get(tool_name)
        if tool is None:
            raise ValueError(f"未找到工具: {tool_name}")
        result = tool.invoke(tool_args if tool_args else {})
        return result if isinstance(result, str) else str(result)

    def _record_tool_stage_event(self, runtime_context: AgentRuntimeContext, tool_name: str) -> None:
        if tool_name == "rag_summarize":
            record_status_event(runtime_context, event_type="stage.rag", title="正在调用知识库", detail="正在检索扫地机器人相关知识库资料")
            return
        if tool_name == "fetch_external_history":
            record_status_event(runtime_context, event_type="stage.trend", title="正在汇总多月趋势", detail="正在整理最近多个月的使用趋势记录")
            return
        record_status_event(runtime_context, event_type="stage.tool", title="正在调用工具", detail=f"正在调用 {tool_name}")

    def _append_react_trace(self, runtime_context: AgentRuntimeContext, phase: str, content: str) -> None:
        trace = runtime_context.setdefault("react_trace", [])
        trace.append({"phase": phase, "content": content})
        logger.info(f"[ReAct]phase={phase} | user_id={runtime_context.get('user_id', '-')} | session_id={runtime_context.get('session_id', '-')} | {content}")

    def _is_report_query(self, query: str) -> bool:
        return any(keyword in query for keyword in REPORT_KEYWORDS)

    def _is_smalltalk(self, query: str) -> bool:
        return any(keyword in query for keyword in SMALLTALK_KEYWORDS)

    def _needs_weather(self, query: str) -> bool:
        if any(keyword in query for keyword in WEATHER_KEYWORDS):
            return True
        return any(keyword in query for keyword in IMPLICIT_LOCATION_KEYWORDS) and any(keyword in query for keyword in ENVIRONMENT_DECISION_KEYWORDS)

    def _needs_knowledge(self, query: str) -> bool:
        if self._needs_weather(query):
            return any(keyword in query for keyword in WEATHER_KNOWLEDGE_KEYWORDS)
        return any(keyword in query for keyword in DOMAIN_KEYWORDS)

    def _needs_report_history(self, query: str) -> bool:
        return any(keyword in query for keyword in REPORT_HISTORY_KEYWORDS)

    def _needs_report_knowledge(self, query: str) -> bool:
        return any(keyword in query for keyword in REPORT_KNOWLEDGE_KEYWORDS)

    def _extract_city(self, query: str) -> str:
        match = CITY_PATTERN.search(query)
        if not match:
            return ""
        candidate = match.group("city")
        for token in ("今天", "今日", "现在", "当前", "最近", "这几天"):
            if candidate.endswith(token):
                candidate = candidate[:-len(token)]
                break
        return "" if candidate in NON_CITY_TOKENS else candidate

    def _build_rag_query(self, query: str) -> str:
        cleaned = re.sub(r"[？?！!。,.，]", " ", query.strip())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _build_report_rag_query(self, state: ReportGraphState) -> str:
        query = self._build_rag_query(state["query"])
        monthly_record = (state.get("known_facts", {}).get("monthly_record") or "").strip()
        return f"{query} {monthly_record[:80]}".strip() if monthly_record else query

    def _extract_focus_terms(self, query: str, known_facts: dict[str, str]) -> list[str]:
        focus_terms: list[str] = []
        for keyword in list(DOMAIN_KEYWORDS) + list(WEATHER_KEYWORDS):
            if keyword in query and keyword not in focus_terms:
                focus_terms.append(keyword)
        if "木地板" in query and any(token in query for token in ("拖地", "湿拖", "扫拖")) and "出水量" not in focus_terms:
            focus_terms.append("出水量")
        if "出水量" in known_facts.get("knowledge_info", "") and "出水量" not in focus_terms:
            focus_terms.append("出水量")
        city = known_facts.get("city")
        if city and city not in focus_terms:
            focus_terms.append(city)
        return focus_terms[:6]

    def _extract_report_month(self, query: str) -> str:
        return self._normalize_report_month(query)

    def _normalize_report_month(self, value: str) -> str:
        match = YEAR_MONTH_PATTERN.search(value or "")
        if match:
            return f"{match.group('year')}-{int(match.group('month')):02d}"
        stripped = (value or "").strip()
        return stripped if re.fullmatch(r"20\d{2}-\d{2}", stripped) else ""

    def _remove_gap(self, remaining_questions: list[str], gap_name: str) -> list[str]:
        return [item for item in remaining_questions if item != gap_name]

    def _is_unusable_tool_result(self, result: str) -> bool:
        unusable_markers = ("当前会话未配置", "天气服务未配置", "天气服务暂时不可用", "未查询到", "我不知道")
        normalized = (result or "").strip()
        return not normalized or any(marker in normalized for marker in unusable_markers)

    def _is_report_time_consistent(self, report: str, target_month: str) -> bool:
        mentioned_months: list[str] = []
        seen: set[str] = set()
        for match in YEAR_MONTH_PATTERN.finditer(report or ""):
            month = f"{match.group('year')}-{int(match.group('month')):02d}"
            if month not in seen:
                seen.add(month)
                mentioned_months.append(month)
        if not mentioned_months or target_month not in mentioned_months:
            return False
        return all(month <= target_month for month in mentioned_months)

    def _summarize_tool_result(self, tool_name: str, result: str) -> str:
        normalized = (result or "").strip().replace("\n", " ")
        if len(normalized) > 80:
            normalized = normalized[:80].rstrip() + "..."
        return normalized if normalized else f"{tool_name} 未返回有效结果"
