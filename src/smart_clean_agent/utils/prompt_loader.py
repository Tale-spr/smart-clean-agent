from smart_clean_agent.utils.config_handler import prompts_conf
from smart_clean_agent.utils.logger_handler import logger
from smart_clean_agent.utils.path_tool import get_abs_path


def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[加载系统提示词]未找到配置项: main_prompt_path")
        raise e

    try:
        with open(system_prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"[加载系统提示词]解析系统提示词出错: {str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[加载系统提示词]未找到配置项: rag_summarize_prompt_path")
        raise e

    try:
        with open(rag_prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"[加载系统提示词]解析rag总结提示词出错: {str(e)}")
        raise e


def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[加载系统提示词]未找到配置项: report_prompt_path")
        raise e

    try:
        with open(report_prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"[加载系统提示词]解析报告生成提示词出错: {str(e)}")
        raise e


def load_query_normalize_prompt():
    try:
        normalize_prompt_path = get_abs_path(prompts_conf["query_normalize_prompt_path"])
    except KeyError as e:
        logger.error("[加载系统提示词]未找到配置项: query_normalize_prompt_path")
        raise e

    try:
        with open(normalize_prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        logger.error(f"[加载系统提示词]解析问题归一化提示词出错: {str(e)}")
        raise e


if __name__ == '__main__':
    print(load_system_prompts())

