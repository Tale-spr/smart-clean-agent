import csv
import os
from datetime import datetime

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from smart_clean_agent.agent.runtime_context import AgentRuntimeContext
from smart_clean_agent.rag.rag_service import RagSummarizeService
from smart_clean_agent.services.report_memory_service import build_report_memory_summary, refresh_report_memory
from smart_clean_agent.services.weather_service import get_weather_by_city
from smart_clean_agent.utils.config_handler import agent_conf
from smart_clean_agent.utils.logger_handler import logger
from smart_clean_agent.utils.path_tool import get_abs_path

external_data: dict[str, dict[str, dict[str, str]]] = {}



def build_rag_summarize_tool(rag_service: RagSummarizeService) -> BaseTool:
    @tool(description="从向量存储中检索参考资料")
    def rag_summarize(query: str) -> str:
        return rag_service.rag_summarize(query)

    return rag_summarize


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return get_weather_by_city(city)



def get_user_id_from_context(context: AgentRuntimeContext | dict | None) -> str:
    if not context:
        return "当前会话未配置用户ID"
    return context.get("user_id") or "当前会话未配置用户ID"



def get_user_location_from_context(context: AgentRuntimeContext | dict | None) -> str:
    if not context:
        return "当前会话未配置用户所在城市"
    return context.get("city") or "当前会话未配置用户所在城市"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location(runtime: ToolRuntime) -> str:
    return get_user_location_from_context(runtime.context)


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id(runtime: ToolRuntime) -> str:
    return get_user_id_from_context(runtime.context)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")



def generate_external_data():
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_fields = {"用户ID", "特征", "清洁效率", "耗材", "对比", "时间"}

            if not reader.fieldnames or not required_fields.issubset(set(reader.fieldnames)):
                raise ValueError("外部数据文件缺少必要字段")

            for row in reader:
                user_id = row["用户ID"]
                feature = row["特征"]
                efficiency = row["清洁效率"]
                consumables = row["耗材"]
                comparison = row["对比"]
                time = row["时间"]

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }



def format_external_data(record: dict[str, str], month: str) -> str:
    return (
        f"月份: {month}\n"
        f"特征: {record['特征']}\n"
        f"清洁效率: {record['效率']}\n"
        f"耗材: {record['耗材']}\n"
        f"对比: {record['对比']}"
    )


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return format_external_data(external_data[user_id][month], month)
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="从外部系统中获取指定用户最近多个月的使用趋势，以纯字符串形式返回，内容包含覆盖月份、清洁效率趋势、耗材趋势、问题变化趋势和总结建议")
def fetch_external_history(user_id: str, months: int = 3) -> str:
    memory = refresh_report_memory(user_id, months=months)
    summary = build_report_memory_summary(memory)
    if not summary:
        return f"未查询到用户{user_id}最近{months}个月的趋势记录"
    return summary


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"



def create_agent_tools(rag_service: RagSummarizeService) -> list[BaseTool]:
    return [
        build_rag_summarize_tool(rag_service),
        get_weather,
        get_user_location,
        get_user_id,
        get_current_month,
        fetch_external_data,
        fetch_external_history,
        fill_context_for_report,
    ]

