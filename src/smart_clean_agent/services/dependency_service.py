import os
from pathlib import Path

from dotenv import load_dotenv

from smart_clean_agent.rag.vector_store import get_vector_store_sqlite_path, vector_store_exists
from smart_clean_agent.utils.config_handler import agent_conf, prompts_conf
from smart_clean_agent.utils.path_tool import get_abs_path


PROMPT_DEPENDENCIES = {
    "main_prompt_path": "主提示词文件",
    "rag_summarize_prompt_path": "RAG 提示词文件",
    "report_prompt_path": "报告提示词文件",
}

DATA_DEPENDENCIES = {
    "user_profiles_path": "用户资料文件",
    "external_data_path": "外部记录文件",
}


def load_runtime_environment() -> None:
    env_path = get_abs_path(".env")
    if Path(env_path).exists():
        load_dotenv(env_path, override=False)


def collect_runtime_dependency_issues() -> list[str]:
    load_runtime_environment()
    issues: list[str] = []

    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        issues.append("未配置 DASHSCOPE_API_KEY")

    if not vector_store_exists():
        issues.append(f"本地向量库不存在: {get_vector_store_sqlite_path()}")

    for config_key, label in PROMPT_DEPENDENCIES.items():
        try:
            prompt_path = Path(get_abs_path(prompts_conf[config_key]))
        except KeyError:
            issues.append(f"缺少配置项: {config_key}")
            continue
        if not prompt_path.exists():
            issues.append(f"{label}不存在: {prompt_path}")

    for config_key, label in DATA_DEPENDENCIES.items():
        try:
            data_path = Path(get_abs_path(agent_conf[config_key]))
        except KeyError:
            issues.append(f"缺少配置项: {config_key}")
            continue
        if not data_path.exists():
            issues.append(f"{label}不存在: {data_path}")

    return issues


def validate_runtime_dependencies() -> None:
    issues = collect_runtime_dependency_issues()
    if issues:
        raise RuntimeError("；".join(issues))

