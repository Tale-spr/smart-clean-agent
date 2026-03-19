from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import BaseTool

from smart_clean_agent.agent.runtime_context import AgentRuntimeContext
from smart_clean_agent.agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from smart_clean_agent.services.status_event_service import record_status_event
from smart_clean_agent.utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.chat_model = model
        self.tools = tools
        self.agent = create_agent(
            model=self.chat_model,
            system_prompt=load_system_prompts(),
            tools=self.tools,
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            context_schema=AgentRuntimeContext,
        )

    def execute_stream(self, query: str, runtime_context: AgentRuntimeContext):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        final_stage_recorded = False
        accumulated_content = ""
        try:
            for chunk in self.agent.stream(input_dict, stream_mode="values", context=runtime_context):
                latest_message = chunk["messages"][-1]
                if not isinstance(latest_message, (AIMessage, AIMessageChunk)):
                    continue

                current_content = (latest_message.content or "")
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

    def execute(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        return "".join(self.execute_stream(query, runtime_context)).strip()

