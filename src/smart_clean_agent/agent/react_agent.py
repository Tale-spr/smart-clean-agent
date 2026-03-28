import json
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
from smart_clean_agent.utils.prompt_loader import load_query_normalize_prompt, load_report_prompts, load_system_prompts

MAX_REACT_STEPS = 5
STOP_REASONS = {"enough_information", "max_steps_reached", "tool_failed", "unsupported_request", "consistency_failed"}
REPORT_KEYWORDS = ("报告", "月报", "使用记录", "趋势", "统计")
REPORT_HISTORY_KEYWORDS = ("趋势", "变化", "对比", "最近", "近几月", "多月", "长期", "历史")
REPORT_KNOWLEDGE_KEYWORDS = ("保养", "维护", "建议", "耗材", "更换", "风险", "提升", "优化", "注意")
WEATHER_KEYWORDS = ("天气", "气温", "温度", "湿度", "下雨", "降雨", "空气", "潮湿", "干燥", "回南天", "梅雨天", "回南", "梅雨")
DOMAIN_KEYWORDS = ("扫地机器人", "扫拖", "机器人", "清洁", "拖地", "湿拖", "吸力", "滚刷", "滤网", "水箱", "避障", "回充", "漏扫", "地图", "导航", "保养", "维护", "地板", "木地板", "地毯", "瓷砖", "滚刷更换", "主刷", "边刷", "拖布", "HEPA", "粉尘", "WiFi", "门槛", "水痕")
SMALLTALK_KEYWORDS = ("你好", "您好", "谢谢", "感谢", "你是谁", "再见")
CAPABILITY_QUERY_KEYWORDS = ("你能帮我做什么", "你能做什么", "可以帮我做什么", "你会什么", "能帮我什么", "能做什么")
IMPLICIT_LOCATION_KEYWORDS = ("我所在城市", "当前城市", "现在这里", "我这边", "所在城市")
ENVIRONMENT_DECISION_KEYWORDS = ("适不适合", "适合", "能不能", "要不要", "会不会影响", "有什么影响")
WEATHER_KNOWLEDGE_KEYWORDS = ("保养", "存放", "耗材", "回充", "导航", "地图", "避障", "滤网", "滚刷", "拖布", "出水量", "水箱", "故障")
CITY_PATTERN = re.compile(r"(?P<city>[\u4e00-\u9fa5]{2,6})(?:今天|现在|当前|这几天)?(?:的)?(?:天气|温度|湿度|空气|下雨|降雨)")
NON_CITY_TOKENS = {"今天", "今日", "现在", "当前", "这几天", "最近", "在的城市", "所在的城市", "我所在的城市"}
YEAR_MONTH_PATTERN = re.compile(r"(?P<year>\d{4})\s*(?:年|/|-)\s*(?P<month>\d{1,2})\s*月?")


class ReActGraphState(TypedDict):
    query: str
    raw_query: str
    runtime_context: AgentRuntimeContext
    normalized_query: str
    normalization_payload: dict[str, Any]
    normalization_confidence: str
    normalization_fallback_used: bool
    retrieved_user_memory_summary: str
    retrieved_user_memory_fields: list[str]
    memory_retrieval_reason: str
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
    def __init__(self, model: BaseChatModel, tools: list[BaseTool], normalization_model: BaseChatModel | None = None):
        self.chat_model = model
        self.normalization_model = normalization_model or model
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
        graph.add_node("normalize_query", self._normalize_query)
        graph.add_node("analyze_question", self._analyze_question)
        graph.add_node("select_tool_or_finish", self._select_tool_or_finish)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("observe_tool_result", self._observe_tool_result)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_edge(START, "normalize_query")
        graph.add_edge("normalize_query", "analyze_question")
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
        runtime_context.setdefault("tool_evidence", [])
        initial_state: ReActGraphState = {
            "query": query,
            "raw_query": query,
            "runtime_context": runtime_context,
            "normalized_query": query,
            "normalization_payload": {},
            "normalization_confidence": "low",
            "normalization_fallback_used": False,
            "retrieved_user_memory_summary": "",
            "retrieved_user_memory_fields": [],
            "memory_retrieval_reason": "",
            "intent": "",
            "known_facts": {},
            "tool_history": [],
            "observations": [],
            "remaining_questions": [],
            "final_answer": "",
            "step_count": 0,
            "stop_reason": "",
        }
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
        runtime_context.setdefault("tool_evidence", [])
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

    def _normalize_query(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        raw_query = (state.get("raw_query") or state["query"]).strip()
        record_status_event(runtime_context, event_type="stage.model", title="正在理解问题表述", detail="正在归一化用户问题并提取稳定意图")

        payload = self._default_normalization_payload(raw_query)
        fallback_used = False
        try:
            response = self.normalization_model.invoke(
                [
                    SystemMessage(content=load_query_normalize_prompt()),
                    HumanMessage(
                        content=(
                            f"用户问题:\n{raw_query}\n\n"
                            f"上下文用户ID: {runtime_context.get('user_id', '')}\n"
                            f"上下文城市: {runtime_context.get('city', '')}\n"
                            f"会话摘要:\n{runtime_context.get('session_summary', '')}\n\n"
                            f"最近历史:\n{runtime_context.get('recent_history', '')}"
                        )
                    ),
                ]
            )
            content = getattr(response, "content", "")
            payload = self._normalize_normalization_payload(self._extract_json_payload(content), raw_query)
        except Exception as exc:
            fallback_used = True
            logger.warning(f"[NormalizeQuery]归一化失败，回退启发式规则: {str(exc)}")
            payload = self._fallback_normalization_payload(raw_query)

        self._append_react_trace(
            runtime_context,
            phase="normalize",
            content=(
                f"intent={payload['intent'] or '-'}; confidence={payload['confidence']}; "
                f"normalized={payload['normalized_query']}; fallback={str(fallback_used).lower()}"
            ),
        )
        return {
            "normalized_query": payload["normalized_query"],
            "normalization_payload": payload,
            "normalization_confidence": payload["confidence"],
            "normalization_fallback_used": fallback_used,
        }

    def _analyze_question(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        query = (state.get("normalized_query") or state["query"]).strip()
        raw_query = (state.get("raw_query") or state["query"]).strip()
        record_status_event(runtime_context, event_type="stage.model", title="正在分析问题", detail="正在判断是否需要检索知识库或调用工具")

        normalization_payload = state.get("normalization_payload") or {}
        if normalization_payload and normalization_payload.get("intent"):
            intent = str(normalization_payload.get("intent") or "")
            known_facts: dict[str, str] = {}
            explicit_city = str(normalization_payload.get("city") or "").strip()
            if explicit_city:
                known_facts["city"] = explicit_city
            weather_premise_type = str(normalization_payload.get("user_weather_premise_type") or "").strip()
            weather_premise_text = str(normalization_payload.get("user_weather_premise_text") or "").strip()
            if weather_premise_type and weather_premise_text:
                known_facts["user_weather_premise"] = weather_premise_text
                known_facts["weather_premise_type"] = weather_premise_type
            remaining_questions = self._normalize_missing_slots(
                normalization_payload,
                intent=intent,
                has_city=bool(explicit_city),
            )
            stop_reason = "unsupported_request" if intent == "unsupported" else ("enough_information" if intent == "direct" and not remaining_questions else "")
        else:
            weather_needed = self._needs_weather(query)
            knowledge_needed = self._needs_knowledge(query)
            direct_reply = self._is_smalltalk(query)
            known_facts = {}
            explicit_city = self._extract_city(query)
            if explicit_city:
                known_facts["city"] = explicit_city
            weather_premise = self._extract_weather_premise(raw_query)
            if weather_premise:
                known_facts["user_weather_premise"] = weather_premise["text"]
                known_facts["weather_premise_type"] = weather_premise["type"]

            if direct_reply and not weather_needed and not knowledge_needed:
                intent = "direct"
                remaining_questions = []
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

        retrieved_user_memory_fields, retrieved_user_memory_summary, memory_retrieval_reason = self._retrieve_user_memory_for_normal_query(
            raw_query=raw_query,
            normalized_query=query,
            intent=intent,
            known_facts=known_facts,
            runtime_context=runtime_context,
        )
        runtime_context["retrieved_user_memory_fields"] = retrieved_user_memory_fields
        runtime_context["retrieved_user_memory_summary"] = retrieved_user_memory_summary
        runtime_context["memory_retrieval_reason"] = memory_retrieval_reason
        self._append_react_trace(
            runtime_context,
            phase="memory",
            content=(
                f"fields={','.join(retrieved_user_memory_fields) or '-'}; "
                f"reason={memory_retrieval_reason or '-'}"
            ),
        )

        self._append_react_trace(runtime_context, phase="analyze", content=f"intent={intent}; remaining={','.join(remaining_questions) or '-'}")
        return {
            "intent": intent,
            "known_facts": known_facts,
            "remaining_questions": remaining_questions,
            "stop_reason": stop_reason,
            "retrieved_user_memory_fields": retrieved_user_memory_fields,
            "retrieved_user_memory_summary": retrieved_user_memory_summary,
            "memory_retrieval_reason": memory_retrieval_reason,
        }

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
            decision = ("rag_summarize", {"query": self._build_rag_query(state.get("normalized_query") or state["query"])})

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
            runtime_context.setdefault("tool_evidence", []).append(
                {
                    "tool_name": tool_name,
                    "summary": self._summarize_tool_result(tool_name, result),
                }
            )
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
            if known_facts.get("weather_premise_type"):
                known_facts["weather_conflict"] = "true" if self._weather_conflicts_user_premise(known_facts["weather_premise_type"], tool_result) else "false"
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
        answer_style = self._determine_answer_style(state)
        if answer_style == "smalltalk":
            answer = self._build_smalltalk_answer(state)
        elif answer_style == "capability_intro":
            answer = self._build_capability_intro_answer()
        elif stop_reason == "unsupported_request" and not known_facts:
            answer = "我不知道"
        elif stop_reason in {"tool_failed", "max_steps_reached"} and not known_facts:
            answer = "我不知道"
        elif self._should_use_guarded_environment_answer(state):
            answer = self._build_guarded_environment_answer(state)
        else:
            answer = self._build_final_answer(state, answer_style=answer_style)

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
            runtime_context.setdefault("tool_evidence", []).append(
                {
                    "tool_name": tool_name,
                    "summary": self._summarize_tool_result(tool_name, result),
                }
            )
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
                known_facts["monthly_record_unavailable"] = "true"
                known_facts["monthly_record"] = ""
                missing_inputs = self._remove_gap(missing_inputs, "monthly_record")
                return {"known_facts": known_facts, "missing_inputs": missing_inputs, "observations": observations, "last_tool_result": "", "last_tool_error": ""}
            known_facts["monthly_record_unavailable"] = "false"
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
        elif known_facts.get("monthly_record_unavailable") == "true" or not monthly_record:
            report = self._build_limited_report(state)
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

    def _build_final_answer(self, state: ReActGraphState, *, answer_style: str) -> str:
        runtime_context = state["runtime_context"]
        query = state.get("raw_query") or state["query"]
        known_facts = state.get("known_facts", {})
        observations = state.get("observations", [])
        stop_reason = state.get("stop_reason", "") or "enough_information"
        evidence_sections: list[str] = []
        if known_facts.get("city"):
            evidence_sections.append(f"用户城市: {known_facts['city']}")
        if known_facts.get("weather_info"):
            evidence_sections.append(f"天气信息:\n{known_facts['weather_info']}")
        if known_facts.get("user_weather_premise"):
            evidence_sections.append(f"用户明确前提:\n{known_facts['user_weather_premise']}")
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
        if known_facts.get("weather_conflict") == "true":
            human_sections.append("天气工具返回与用户明确描述存在差异。不要直接否定用户前提，请说明信息差异，并按更保守的场景给建议。")
        if known_facts.get("weather_premise_type") in {"humid", "rain"} and "出水量" in query:
            human_sections.append("对于潮湿或雨天与出水量调整相关的问题，如果知识库没有明确支持，不要输出“无需降低出水量”这类强结论，应优先给出保守建议。")
        if answer_style == "concise_fact":
            human_sections.append("回答风格：自然、简洁、直接。优先用 1 到 2 句先回答用户问题，必要时再补一句原因或提醒；不要使用“结论：”“依据：”“建议：”等显式标题。")
        else:
            human_sections.append("回答风格：自然、专业、像真实客服交流。先直接回答用户问题，再在必要时自然补充原因和可执行建议；最多使用两小段，不要使用“结论：”“依据：”“建议：”等显式标题。")
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
        human_sections.append("要求：先给出该月主结论；只有在明确拿到趋势信息时，才补充最近多月变化和长期建议；只能引用“本月记录”和“趋势信息”中已明确出现的具体数值、次数、比例、寿命天数；如果某部分信息不足，请明确说明信息范围有限，不要自行补齐具体数字。")
        human_sections.append("只输出最终报告，不要展示内部推理、工具名或中间分析。")
        return self._invoke_model(load_report_prompts(), human_sections)

    def _build_limited_report(self, state: ReportGraphState) -> str:
        known_facts = state.get("known_facts", {})
        target_month = known_facts.get("report_target_month") or "目标月份"
        knowledge_info = (known_facts.get("knowledge_info") or "").strip()
        history_info = (known_facts.get("history_info") or "").strip()
        lines = [
            "# 扫地机器人使用情况报告与保养建议",
            "",
            f"## 本月主结论（{target_month}）",
            "",
            "当前未获取到足够完整的本月使用记录，因此无法可靠给出覆盖率、次数、寿命天数等具体量化结论。",
            "",
            "## 信息范围说明",
            "",
            "本报告仅能基于已有上下文和通用维护知识给出保守建议，建议后续补充当月外部记录后再生成完整版月报。",
        ]
        if history_info:
            lines.extend(["", "## 已有趋势信息", "", history_info])
        if knowledge_info:
            lines.extend(["", "## 保守建议", "", knowledge_info])
        else:
            lines.extend(
                [
                    "",
                    "## 保守建议",
                    "",
                    "1. 先检查主刷、边刷、滤网和拖布是否存在明显缠绕、堵塞或磨损。",
                    "2. 如近期有湿拖任务，优先从低档或中低档出水量开始，观察地面残留情况再调整。",
                    "3. 若后续补齐当月记录，可重新生成更完整的使用报告与耗材建议。",
                ]
            )
        return "\n".join(lines).strip()

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

    def _is_capability_query(self, query: str) -> bool:
        normalized = query.strip()
        return any(keyword in normalized for keyword in CAPABILITY_QUERY_KEYWORDS)

    def _determine_answer_style(self, state: ReActGraphState) -> str:
        query = (state.get("raw_query") or state["query"]).strip()
        intent = state.get("intent", "")
        known_facts = state.get("known_facts", {})
        if self._is_smalltalk(query):
            return "smalltalk"
        if self._is_capability_query(query):
            return "capability_intro"
        if intent == "weather" and "knowledge_info" not in known_facts:
            return "concise_fact"
        return "natural_advice"

    def _build_smalltalk_answer(self, state: ReActGraphState) -> str:
        query = (state.get("raw_query") or state["query"]).strip()
        if any(token in query for token in ("谢谢", "感谢")):
            return "不客气，有需要随时告诉我。"
        if "再见" in query:
            return "好的，随时需要时再来找我。"
        if "你是谁" in query:
            return "我是扫地机器人客服助手，可以帮你看使用建议、保养维护、故障排查和报告生成。"
        return "你好，我可以帮你解答扫地机器人使用、保养、故障排查和环境适配相关问题。"

    def _build_capability_intro_answer(self) -> str:
        return "我可以帮你查看天气对使用的影响，解答扫地机器人使用、保养和故障排查问题，也可以生成月度使用报告。"

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
        if candidate in NON_CITY_TOKENS or "城市" in candidate:
            return ""
        return candidate

    def _build_rag_query(self, query: str) -> str:
        cleaned = re.sub(r"[？?！!。,.，]", " ", query.strip())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _retrieve_user_memory_for_normal_query(
        self,
        *,
        raw_query: str,
        normalized_query: str,
        intent: str,
        known_facts: dict[str, str],
        runtime_context: AgentRuntimeContext,
    ) -> tuple[list[str], str, str]:
        payload = runtime_context.get("user_memory_payload") or {}
        if not isinstance(payload, dict) or not payload:
            return [], "", "no_user_memory_payload"

        merged_query = f"{raw_query} {normalized_query}".strip()
        if self._should_skip_user_memory_retrieval(
            merged_query=merged_query,
            intent=intent,
            runtime_context=runtime_context,
        ):
            return [], "", "skip_for_smalltalk_or_weather"

        selected_fields: list[str] = []
        lowered_query = merged_query.lower()

        environment_keywords = ("适合", "适不适合", "拖地", "湿拖", "出水量", "地板", "地面", "宠物", "养猫", "养狗", "毛发", "缠绕")
        cleaning_plan_keywords = ("计划", "模式", "频率", "每天", "每日", "预约", "定时", "清扫", "拖扫", "拖地")
        troubleshooting_keywords = ("漏扫", "回充", "缠绕", "滤网", "水箱", "地图", "联网", "故障", "主刷", "边刷", "滚刷", "拖布", "怎么办")
        profile_keywords = ("户型", "地板", "地面", "小户型", "大户型", "老人", "儿童", "城市")

        if intent == "combined" or any(keyword in merged_query for keyword in environment_keywords):
            selected_fields.extend(["environment", "preferences"])
        if any(keyword in merged_query for keyword in cleaning_plan_keywords):
            selected_fields.extend(["cleaning_habits", "preferences", "environment"])
        if any(keyword in lowered_query for keyword in troubleshooting_keywords):
            selected_fields.extend(["pain_points", "recent_focuses"])
        if any(keyword in merged_query for keyword in profile_keywords):
            selected_fields.append("profile_snapshot")

        if not selected_fields:
            return [], "", "no_relevant_memory_fields"

        deduped_fields: list[str] = []
        for field in selected_fields:
            if field not in deduped_fields:
                deduped_fields.append(field)

        summary = self._build_retrieved_user_memory_summary(
            payload=payload,
            selected_fields=deduped_fields,
            query=merged_query,
            known_facts=known_facts,
        )
        if not summary:
            return [], "", "selected_fields_empty_after_filter"
        return deduped_fields, summary, "retrieved_relevant_user_memory"

    def _should_skip_user_memory_retrieval(
        self,
        *,
        merged_query: str,
        intent: str,
        runtime_context: AgentRuntimeContext,
    ) -> bool:
        if runtime_context.get("is_new_session_first_turn") and (
            intent == "direct" or self._is_smalltalk(merged_query) or self._is_capability_query(merged_query)
        ):
            return True
        if self._is_smalltalk(merged_query) or self._is_capability_query(merged_query):
            return True
        if intent == "weather":
            return True
        return False

    def _build_retrieved_user_memory_summary(
        self,
        *,
        payload: dict[str, Any],
        selected_fields: list[str],
        query: str,
        known_facts: dict[str, str],
    ) -> str:
        lines: list[str] = []
        profile_snapshot = payload.get("profile_snapshot") or {}

        if "profile_snapshot" in selected_fields and isinstance(profile_snapshot, dict):
            profile_parts: list[str] = []
            city = (profile_snapshot.get("city") or "").strip()
            house_type = (profile_snapshot.get("house_type") or "").strip()
            floor_type = (profile_snapshot.get("floor_type") or "").strip()
            if "城市" in query and city:
                profile_parts.append(f"城市: {city}")
            if any(token in query for token in ("户型", "小户型", "大户型", "老人", "儿童")) and house_type:
                profile_parts.append(f"户型: {house_type}")
            if any(token in query for token in ("地板", "地面", "拖地", "湿拖", "出水量")) and floor_type:
                profile_parts.append(f"地面: {floor_type}")
            if profile_parts:
                lines.append("用户画像: " + "；".join(profile_parts))

        field_labels = {
            "preferences": "相关偏好",
            "environment": "相关环境特征",
            "cleaning_habits": "相关清洁习惯",
            "pain_points": "相关历史问题",
            "recent_focuses": "相关最近关注",
        }
        for field in ("preferences", "environment", "cleaning_habits", "pain_points", "recent_focuses"):
            if field not in selected_fields:
                continue
            values = payload.get(field) or []
            if not isinstance(values, list):
                continue
            filtered_values = self._filter_user_memory_field_values(field, values, query, known_facts)
            if filtered_values:
                lines.append(f"{field_labels[field]}: " + "、".join(filtered_values[:3]))

        return "\n".join(lines).strip()

    def _filter_user_memory_field_values(
        self,
        field_name: str,
        values: list[Any],
        query: str,
        known_facts: dict[str, str],
    ) -> list[str]:
        normalized_values = [str(value).strip() for value in values if str(value).strip()]
        if not normalized_values:
            return []

        tokens_map: dict[str, tuple[str, ...]] = {
            "preferences": ("湿拖", "拖地", "静音", "安静"),
            "environment": ("木地板", "瓷砖", "地毯", "养宠", "宠物", "猫", "狗", "小户型", "大户型", "多层", "老人", "儿童"),
            "cleaning_habits": ("定时", "预约", "每天", "每日", "高频", "沿边", "清扫", "拖地"),
            "pain_points": ("避障", "回充", "漏扫", "主刷", "缠绕", "滤网", "水箱", "地图", "联网", "故障", "滚刷", "拖布"),
            "recent_focuses": ("木地板", "瓷砖", "地毯", "养宠", "漏扫", "回充", "缠绕", "滤网", "水箱", "拖地", "湿拖", "出水量", "滚刷", "拖布"),
        }
        query_tokens = tokens_map.get(field_name, ())
        matched = [value for value in normalized_values if any(token in query or token in value for token in query_tokens)]

        if field_name == "environment" and not matched and (
            "出水量" in query or "拖地" in query or "湿拖" in query or known_facts.get("weather_premise_type")
        ):
            matched = [value for value in normalized_values if value in {"木地板", "瓷砖", "地毯", "养宠", "小户型", "大户型"}]

        if field_name == "preferences" and not matched and ("拖地" in query or "湿拖" in query or "出水量" in query):
            matched = [value for value in normalized_values if "湿拖" in value or "静音" in value]

        if field_name == "recent_focuses" and matched and "pain_points" not in known_facts:
            return matched[:2]

        return matched or (normalized_values[:2] if field_name in {"environment", "preferences"} and ("适合" in query or "建议" in query) else [])

    def _extract_json_payload(self, content: Any) -> dict[str, Any]:
        normalized = content if isinstance(content, str) else str(content or "")
        normalized = normalized.strip()
        if normalized.startswith("```"):
            normalized = re.sub(r"^```(?:json)?\s*", "", normalized)
            normalized = re.sub(r"\s*```$", "", normalized)
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未找到合法 JSON 对象")
        return json.loads(normalized[start : end + 1])

    def _default_normalization_payload(self, raw_query: str) -> dict[str, Any]:
        return {
            "normalized_query": raw_query,
            "intent": "",
            "needs_weather": False,
            "needs_knowledge": False,
            "city": "",
            "user_weather_premise_type": "",
            "user_weather_premise_text": "",
            "missing_slots": [],
            "reason": "",
            "confidence": "low",
        }

    def _normalize_normalization_payload(self, payload: dict[str, Any], raw_query: str) -> dict[str, Any]:
        normalized = self._default_normalization_payload(raw_query)
        normalized["normalized_query"] = str(payload.get("normalized_query") or raw_query).strip() or raw_query

        intent = str(payload.get("intent") or "").strip()
        if intent in {"direct", "weather", "knowledge", "combined", "unsupported"}:
            normalized["intent"] = intent

        normalized["needs_weather"] = self._coerce_bool(payload.get("needs_weather"))
        normalized["needs_knowledge"] = self._coerce_bool(payload.get("needs_knowledge"))
        normalized["city"] = str(payload.get("city") or "").strip()

        premise_type = str(payload.get("user_weather_premise_type") or "").strip()
        normalized["user_weather_premise_type"] = premise_type if premise_type in {"", "rain", "humid", "dry"} else ""
        normalized["user_weather_premise_text"] = str(payload.get("user_weather_premise_text") or "").strip()

        confidence = str(payload.get("confidence") or "").strip().lower()
        normalized["confidence"] = confidence if confidence in {"high", "medium", "low"} else "low"
        normalized["reason"] = str(payload.get("reason") or "").strip()

        allowed_slots = {"city", "weather", "knowledge"}
        missing_slots: list[str] = []
        for slot in payload.get("missing_slots") or []:
            if slot in allowed_slots and slot not in missing_slots:
                missing_slots.append(slot)
        normalized["missing_slots"] = missing_slots
        return normalized

    def _fallback_normalization_payload(self, raw_query: str) -> dict[str, Any]:
        payload = self._default_normalization_payload(raw_query)
        weather_needed = self._needs_weather(raw_query)
        knowledge_needed = self._needs_knowledge(raw_query)
        explicit_city = self._extract_city(raw_query)
        premise = self._extract_weather_premise(raw_query)

        if self._is_smalltalk(raw_query) and not weather_needed and not knowledge_needed:
            payload["intent"] = "direct"
            payload["confidence"] = "medium"
        elif weather_needed and knowledge_needed:
            payload["intent"] = "combined"
            payload["needs_weather"] = True
            payload["needs_knowledge"] = True
            payload["confidence"] = "low"
        elif weather_needed:
            payload["intent"] = "weather"
            payload["needs_weather"] = True
            payload["confidence"] = "low"
        elif knowledge_needed:
            payload["intent"] = "knowledge"
            payload["needs_knowledge"] = True
            payload["confidence"] = "low"
        else:
            payload["intent"] = "unsupported"
            payload["confidence"] = "low"

        payload["city"] = explicit_city
        if premise:
            payload["user_weather_premise_type"] = premise["type"]
            payload["user_weather_premise_text"] = premise["text"]
        payload["missing_slots"] = self._normalize_missing_slots(payload, intent=payload["intent"], has_city=bool(explicit_city))
        return payload

    def _normalize_missing_slots(self, payload: dict[str, Any], intent: str, has_city: bool) -> list[str]:
        missing_slots = [slot for slot in payload.get("missing_slots", []) if slot in {"city", "weather", "knowledge"}]
        needs_weather = bool(payload.get("needs_weather"))
        needs_knowledge = bool(payload.get("needs_knowledge"))
        if intent in {"weather", "combined"} and not has_city and "city" not in missing_slots:
            missing_slots.insert(0, "city")
        if needs_weather and "weather" not in missing_slots:
            missing_slots.append("weather")
        if needs_knowledge and "knowledge" not in missing_slots:
            missing_slots.append("knowledge")
        if intent == "direct":
            return []
        if intent == "unsupported" and not missing_slots:
            return []
        normalized: list[str] = []
        for slot in missing_slots:
            if slot not in normalized:
                normalized.append(slot)
        return normalized

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return False

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

    def _extract_weather_premise(self, query: str) -> dict[str, str] | None:
        normalized = query.strip()
        if any(token in normalized for token in ("最近下雨", "下雨", "雨天", "降雨")):
            return {"type": "rain", "text": "用户明确提到最近下雨或处于雨天场景"}
        if any(token in normalized for token in ("潮湿", "湿度高", "回南天", "梅雨", "梅雨天")):
            return {"type": "humid", "text": "用户明确提到当前环境潮湿或湿度较高"}
        if any(token in normalized for token in ("干燥", "湿度低", "空气干燥")):
            return {"type": "dry", "text": "用户明确提到当前环境干燥或湿度较低"}
        return None

    def _normalize_report_month(self, value: str) -> str:
        match = YEAR_MONTH_PATTERN.search(value or "")
        if match:
            return f"{match.group('year')}-{int(match.group('month')):02d}"
        stripped = (value or "").strip()
        return stripped if re.fullmatch(r"20\d{2}-\d{2}", stripped) else ""

    def _weather_conflicts_user_premise(self, premise_type: str, weather_info: str) -> bool:
        normalized = (weather_info or "").strip()
        humidity_match = re.search(r"湿度[:：]\s*(\d{1,3})", normalized)
        humidity = int(humidity_match.group(1)) if humidity_match else None
        if premise_type == "rain":
            return not any(token in normalized for token in ("雨", "降雨"))
        if premise_type == "humid":
            return humidity is not None and humidity < 60
        if premise_type == "dry":
            return humidity is not None and humidity > 65
        return False

    def _should_use_guarded_environment_answer(self, state: ReActGraphState) -> bool:
        query = (state.get("normalized_query") or state["query"]).strip()
        known_facts = state.get("known_facts", {})
        premise_type = known_facts.get("weather_premise_type")
        if premise_type in {"humid", "rain"} and "出水量" in query:
            return True
        if premise_type == "rain" and any(token in query for token in ("注意", "建议", "注意事项")):
            return True
        return False

    def _build_guarded_environment_answer(self, state: ReActGraphState) -> str:
        query = (state.get("normalized_query") or state["query"]).strip()
        known_facts = state.get("known_facts", {})
        premise_type = known_facts.get("weather_premise_type", "")
        weather_info = (known_facts.get("weather_info") or "").strip()
        weather_conflict = known_facts.get("weather_conflict") == "true"

        basis_parts: list[str] = []
        if known_facts.get("user_weather_premise"):
            basis_parts.append(known_facts["user_weather_premise"])
        if weather_info:
            basis_parts.append(f"工具天气信息：{weather_info.replace(chr(10), '；')}")
        if weather_conflict:
            basis_parts.append("天气工具返回与用户描述存在差异，因此以下建议按更保守场景给出。")

        if "出水量" in query and premise_type in {"humid", "rain"}:
            return (
                "在潮湿或下雨环境下，建议先从低档或中低档出水量开始，再根据地面残留和水痕情况慢慢微调。"
                f"{' ' + '；'.join(basis_parts) if basis_parts else ''}"
                "如果地面本身已经偏湿、通风较差或容易留下水痕，就尽量不要直接开高档出水量，清洁后也记得及时擦干机身底部和拖布。"
            )

        return (
            "雨天或地面容易潮湿时，机器人一般还能继续清洁，但要更注意防滑、防潮和传感器误判。"
            f"{' ' + '；'.join(basis_parts) if basis_parts else ''}"
            "使用前先处理积水或明显湿滑区域，避免机器人直接驶入过湿地面；清洁完成后及时擦干机身和拖布，再顺手检查一下底盘、传感器和回充区域有没有受潮。"
        )

    def _remove_gap(self, remaining_questions: list[str], gap_name: str) -> list[str]:
        return [item for item in remaining_questions if item != gap_name]

    def _is_unusable_tool_result(self, result: str) -> bool:
        unusable_markers = ("当前会话未配置", "天气服务未配置", "天气服务暂时不可用", "未查询到", "我不知道")
        normalized = (result or "").strip()
        return not normalized or any(marker in normalized for marker in unusable_markers)

    def _is_report_time_consistent(self, report: str, target_month: str) -> bool:
        report = self._extract_report_analysis_scope(report)
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

    def _extract_report_analysis_scope(self, report: str) -> str:
        normalized = report or ""
        lines = normalized.splitlines(keepends=True)
        keywords = ("建议", "提升方向", "长期使用", "保养与使用")
        offset = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if index == 0:
                offset += len(line)
                continue
            if stripped.startswith("#") and any(keyword in stripped for keyword in keywords):
                return normalized[:offset]
            offset += len(line)
        return normalized

    def _summarize_tool_result(self, tool_name: str, result: str) -> str:
        normalized = (result or "").strip().replace("\n", " ")
        if len(normalized) > 80:
            normalized = normalized[:80].rstrip() + "..."
        return normalized if normalized else f"{tool_name} 未返回有效结果"
