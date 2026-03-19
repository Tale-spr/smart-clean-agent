"""为整个工程提供统一的绝对路径。"""

from pathlib import Path


def get_project_root() -> str:
    """获取项目根目录。"""
    return str(Path(__file__).resolve().parents[3])


def get_abs_path(relative_path: str) -> str:
    """根据项目根目录拼接绝对路径。"""
    return str(Path(get_project_root()) / relative_path)


if __name__ == "__main__":
    print(get_abs_path(r"config\config.txt"))

