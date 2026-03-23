import re
from typing import Any, NotRequired, TypedDict

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from smart_clean_agent.agent.runtime_context import AgentRuntimeContext
from smart_clean_agent.agent.tools.agent_tools import get_user_location_from_context
from smart_clean_agent.agent.tools.middleware import build_runtime_context_prompt, log_before_model, monitor_tool, report_prompt_switch
from smart_clean_agent.services.status_event_service import record_status_event
from smart_clean_agent.utils.logger_handler import logger
from smart_clean_agent.utils.prompt_loader import load_system_prompts

MAX_REACT_STEPS = 5
STOP_REASONS = {"enough_information", "max_steps_reached", "tool_failed", "unsupported_request"}
REPORT_KEYWORDS = ("报告", "月报", "使用记录", "趋势", "统计")
WEATHER_KEYWORDS = ("天气", "气温", "温度", "湿度", "下雨", "降雨", "空气", "潮湿", "干燥", "回南天", "梅雨天", "回南", "梅雨")
DOMAIN_KEYWORDS = (
    "扫地机器人",
    "扫拖",
    "机器人",
    "清洁",
    "拖地",
    "湿拖",
    "吸力",
    "滚刷",
    "滤网",
    "水箱",
    "避障",
    "回充",
    "漏扫",
    "地图",
    "导航",
    "保养",
    "维护",
    "地板",
    "地毯",
    "瓷砖",
    "滚刷更换",
    "主刷",
    "边刷",
    "拖布",
    "HEPA",
    "粉尘",
    "WiFi",
    "门槛",
    "水痕",
)
SMALLTALK_KEYWORDS = ("你好", "您好", "谢谢", "感谢", "你是谁", "再见")
IMPLICIT_LOCATION_KEYWORDS = ("我所在城市", "当前城市", "现在这里", "我这边", "所在城市")
ENVIRONMENT_DECISION_KEYWORDS = ("适不适合", "适合", "能不能", "要不要", "会不会影响", "有什么影响")
WEATHER_KNOWLEDGE_KEYWORDS = (
    "保养",
    "存放",
    "耗材",
    "回充",
    "导航",
    "地图",
    "避障",
    "滤网",
    "滚刷",
    "拖布",
    "出水量",
    "水箱",
    "故障",
)
CITY_PATTERN = re.compile(r"(?P<city>[\u4e00-\u9fa5]{2,6})(?:今天|现在|当前|这几天)?(?:的)?(?:天气|温度|湿度|空气|下雨|降雨)")
NON_CITY_TOKENS = {"今天", "今日", "现在", "当前", "这几天", "最近"}


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


class ReactAgent:
    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.chat_model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.report_agent = create_agent(
            model=self.chat_model,
            system_prompt=load_system_prompts(),
            tools=self.tools,
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            context_schema=AgentRuntimeContext,
        )
        self.agent = self.report_agent
        self.normal_graph = self._build_normal_graph()

    def execute_stream(self, query: str, runtime_context: AgentRuntimeContext):
        if runtime_context.get("force_report_agent") or self._is_report_query(query):
            runtime_context["execution_mode"] = "report"
            yield from self._execute_report_stream(query, runtime_context)
            return

        runtime_context["execution_mode"] = "normal"
        final_answer = self._run_normal_graph(query, runtime_context)
        if final_answer:
            yield final_answer

    def execute(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        return "".join(self.execute_stream(query, runtime_context)).strip()

    def _execute_report_stream(self, query: str, runtime_context: AgentRuntimeContext):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        final_stage_recorded = False
        accumulated_content = ""
        try:
            for chunk in self.report_agent.stream(input_dict, stream_mode="values", context=runtime_context):
                latest_message = chunk["messages"][-1]
                if not isinstance(latest_message, (AIMessage, AIMessageChunk)):
                    continue

                current_content = latest_message.content or ""
                if not isinstance(current_content, str) or not current_content:
                    continue

                if not final_stage_recorded:
                    record_status_event(
                        runtime_context,
                        event_type="stage.final",
                        title="正在生成最终建议",
                        detail="正在整理工具结果、知识库内容与上下文信息",
                    )
                    final_stage_recorded = True

                if current_content.startswith(accumulated_content):
                    delta = current_content[len(accumulated_content):]
                else:
                    delta = current_content

                if not delta:
                    continue

                accumulated_content = current_content
                yield delta

            record_status_event(
                runtime_context,
                event_type="smart_clean_agent.agent.complete",
                title="回答生成完成",
                detail="已完成最终回答生成",
            )
        except Exception as exc:
            record_status_event(
                runtime_context,
                event_type="error.agent",
                title="回答生成失败",
                detail=str(exc),
                level="error",
            )
            raise

    def _build_normal_graph(self):
        graph = StateGraph(ReActGraphState)
        graph.add_node("analyze_question", self._analyze_question)
        graph.add_node("select_tool_or_finish", self._select_tool_or_finish)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("observe_tool_result", self._observe_tool_result)
        graph.add_node("generate_answer", self._generate_answer)

        graph.add_edge(START, "analyze_question")
        graph.add_edge("analyze_question", "select_tool_or_finish")
        graph.add_conditional_edges(
            "select_tool_or_finish",
            self._route_after_selection,
            {
                "execute_tool": "execute_tool",
                "generate_answer": "generate_answer",
            },
        )
        graph.add_edge("execute_tool", "observe_tool_result")
        graph.add_edge("observe_tool_result", "select_tool_or_finish")
        graph.add_edge("generate_answer", END)
        return graph.compile()

    def _run_normal_graph(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        runtime_context["execution_mode"] = "normal"
        runtime_context.setdefault("trace_tool_calls", [])
        runtime_context.setdefault("react_trace", [])
        initial_state: ReActGraphState = {
            "query": query,
            "runtime_context": runtime_context,
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
            record_status_event(
                runtime_context,
                event_type="smart_clean_agent.agent.complete",
                title="回答生成完成",
                detail="已完成最终回答生成",
            )
            return final_state.get("final_answer", "").strip()
        except Exception as exc:
            runtime_context["react_stop_reason"] = "tool_failed"
            record_status_event(
                runtime_context,
                event_type="error.agent",
                title="回答生成失败",
                detail=str(exc),
                level="error",
            )
            raise

    def _analyze_question(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        query = state["query"].strip()
        record_status_event(
            runtime_context,
            event_type="stage.model",
            title="正在分析问题",
            detail="正在判断是否需要检索知识库或调用工具",
        )

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

        self._append_react_trace(
            runtime_context,
            phase="analyze",
            content=f"intent={intent}; remaining={','.join(remaining_questions) or '-'}",
        )

        return {
            "intent": intent,
            "known_facts": known_facts,
            "remaining_questions": remaining_questions,
            "stop_reason": stop_reason,
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
            return {
                "selected_tool": None,
                "selected_tool_args": {},
                "stop_reason": stop_reason or "enough_information",
            }

        next_gap = remaining_questions[0]
        if next_gap == "city":
            decision = ("get_user_location", {})
        elif next_gap == "weather":
            decision = ("get_weather", {"city": state.get("known_facts", {}).get("city", "")})
        else:
            decision = ("rag_summarize", {"query": self._build_rag_query(state["query"])})

        self._append_react_trace(
            runtime_context,
            phase="decide",
            content=f"action={decision[0]}; gap={next_gap}",
        )
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
        self._append_react_trace(
            runtime_context,
            phase="act",
            content=f"{tool_name}({tool_args})",
        )

        tool_history = list(state.get("tool_history", []))
        tool_history.append(
            {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "status": "running",
            }
        )

        try:
            result = self._invoke_normal_tool(tool_name, tool_args, runtime_context)
            tool_history[-1]["status"] = "success"
            record_status_event(
                runtime_context,
                event_type="tool.success",
                title="工具调用完成",
                detail=f"{tool_name} 调用成功",
            )
            return {
                "tool_history": tool_history,
                "last_tool_result": result,
                "last_tool_error": "",
                "step_count": state.get("step_count", 0) + 1,
            }
        except Exception as exc:
            logger.warning(f"[ReAct]工具调用失败: {tool_name} | {str(exc)}")
            tool_history[-1]["status"] = "failed"
            record_status_event(
                runtime_context,
                event_type="error.tool",
                title="工具调用失败",
                detail=f"{tool_name} 调用失败: {str(exc)}",
                level="error",
            )
            return {
                "tool_history": tool_history,
                "last_tool_result": "",
                "last_tool_error": str(exc),
                "step_count": state.get("step_count", 0) + 1,
                "stop_reason": "tool_failed",
            }

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
        self._append_react_trace(
            runtime_context,
            phase="observe",
            content=f"{tool_name}:{observation_summary}",
        )

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

        return {
            "known_facts": known_facts,
            "remaining_questions": remaining_questions,
            "observations": observations,
            "last_tool_result": "",
            "last_tool_error": "",
        }

    def _generate_answer(self, state: ReActGraphState) -> dict[str, Any]:
        runtime_context = state["runtime_context"]
        stop_reason = state.get("stop_reason", "") or "enough_information"
        runtime_context["react_stop_reason"] = stop_reason
        runtime_context["react_step_count"] = state.get("step_count", 0)

        record_status_event(
            runtime_context,
            event_type="stage.final",
            title="正在生成最终建议",
            detail="正在整理工具结果、知识库内容与上下文信息",
        )

        known_facts = state.get("known_facts", {})
        if stop_reason == "unsupported_request" and not known_facts:
            answer = "我不知道"
        elif stop_reason in {"tool_failed", "max_steps_reached"} and not known_facts:
            answer = "我不知道"
        else:
            answer = self._build_final_answer(state)

        self._append_react_trace(runtime_context, phase="finish", content=f"stop_reason={stop_reason}")
        return {"final_answer": answer}

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
            evidence_sections.append(
                "观察摘要:\n" + "\n".join(
                    f"- {item['tool_name']}: {item['summary']}" for item in observations
                )
            )

        context_prompt = build_runtime_context_prompt(runtime_context, is_report=False)
        focus_terms = self._extract_focus_terms(query, known_facts)
        human_sections = [
            "请基于以下已知信息，为用户生成最终中文回答。",
            f"用户问题:\n{query}",
            f"停止原因: {stop_reason}",
        ]
        if context_prompt:
            human_sections.append(context_prompt)
        if evidence_sections:
            human_sections.append("已知信息:\n" + "\n\n".join(evidence_sections))
        else:
            human_sections.append("当前没有可用的外部信息，请仅在你有把握时回答，否则回复“我不知道”。")
        if focus_terms:
            human_sections.append(f"回答时请自然保留这些关键术语：{', '.join(focus_terms)}。")
        human_sections.append(
            "要求：使用“结论 + 依据 + 建议”的最小完整结构；回答保持简洁，但不要遗漏用户问题中的关键部件名、场景名、天气要素或建议动作。"
        )
        human_sections.append("只输出最终回答；不要暴露内部推理、步骤、工具名或中间分析。")

        response = self.chat_model.invoke(
            [
                SystemMessage(content=load_system_prompts()),
                HumanMessage(content="\n\n".join(human_sections)),
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(str(item) for item in content).strip()
        return str(content).strip()

    def _invoke_normal_tool(self, tool_name: str, tool_args: dict[str, Any], runtime_context: AgentRuntimeContext) -> str:
        if tool_name == "get_user_location":
            return get_user_location_from_context(runtime_context)

        tool = self.tool_map.get(tool_name)
        if tool is None:
            raise ValueError(f"未找到工具: {tool_name}")

        result = tool.invoke(tool_args if tool_args else {})
        if isinstance(result, str):
            return result
        return str(result)

    def _record_tool_stage_event(self, runtime_context: AgentRuntimeContext, tool_name: str) -> None:
        if tool_name == "rag_summarize":
            record_status_event(
                runtime_context,
                event_type="stage.rag",
                title="正在调用知识库",
                detail="正在检索扫地机器人相关知识库资料",
            )
            return

        record_status_event(
            runtime_context,
            event_type="stage.tool",
            title="正在调用工具",
            detail=f"正在调用 {tool_name}",
        )

    def _append_react_trace(self, runtime_context: AgentRuntimeContext, phase: str, content: str) -> None:
        trace = runtime_context.setdefault("react_trace", [])
        entry = {"phase": phase, "content": content}
        trace.append(entry)
        logger.info(
            f"[ReAct]phase={phase} | user_id={runtime_context.get('user_id', '-')} | "
            f"session_id={runtime_context.get('session_id', '-')} | {content}"
        )

    def _is_report_query(self, query: str) -> bool:
        return any(keyword in query for keyword in REPORT_KEYWORDS)

    def _is_smalltalk(self, query: str) -> bool:
        return any(keyword in query for keyword in SMALLTALK_KEYWORDS)

    def _needs_weather(self, query: str) -> bool:
        if any(keyword in query for keyword in WEATHER_KEYWORDS):
            return True
        return any(keyword in query for keyword in IMPLICIT_LOCATION_KEYWORDS) and any(
            keyword in query for keyword in ENVIRONMENT_DECISION_KEYWORDS
        )

    def _needs_knowledge(self, query: str) -> bool:
        if self._needs_weather(query):
            return any(keyword in query for keyword in WEATHER_KNOWLEDGE_KEYWORDS)
        return any(keyword in query for keyword in DOMAIN_KEYWORDS)

    def _extract_city(self, query: str) -> str:
        match = CITY_PATTERN.search(query)
        if match:
            candidate = match.group("city")
            for token in ("今天", "今日", "现在", "当前", "最近", "这几天"):
                if candidate.endswith(token):
                    candidate = candidate[:-len(token)]
                    break
            if candidate in NON_CITY_TOKENS:
                return ""
            return candidate
        return ""

    def _build_rag_query(self, query: str) -> str:
        cleaned = query.strip()
        cleaned = re.sub(r"[？?！!。,.，]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _extract_focus_terms(self, query: str, known_facts: dict[str, str]) -> list[str]:
        candidates = list(DOMAIN_KEYWORDS) + list(WEATHER_KEYWORDS)
        focus_terms: list[str] = []
        for keyword in candidates:
            if keyword in query and keyword not in focus_terms:
                focus_terms.append(keyword)
        city = known_facts.get("city")
        if city and city not in focus_terms:
            focus_terms.append(city)
        return focus_terms[:6]

    def _remove_gap(self, remaining_questions: list[str], gap_name: str) -> list[str]:
        return [item for item in remaining_questions if item != gap_name]

    def _is_unusable_tool_result(self, result: str) -> bool:
        unusable_markers = ("当前会话未配置", "天气服务未配置", "天气服务暂时不可用", "未查询到", "我不知道")
        normalized = (result or "").strip()
        return not normalized or any(marker in normalized for marker in unusable_markers)

    def _summarize_tool_result(self, tool_name: str, result: str) -> str:
        normalized = (result or "").strip().replace("\n", " ")
        if len(normalized) > 80:
            normalized = normalized[:80].rstrip() + "..."
        if not normalized:
            return f"{tool_name} 未返回有效结果"
        return normalized
