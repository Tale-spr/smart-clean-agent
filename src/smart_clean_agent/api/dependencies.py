from fastapi.responses import JSONResponse

from smart_clean_agent.api.schemas import ErrorResponse
from smart_clean_agent.services.dependency_service import collect_runtime_dependency_issues, load_runtime_environment


def get_dependency_issues() -> list[str]:
    return collect_runtime_dependency_issues()


def build_dependency_error_response() -> JSONResponse:
    issues = get_dependency_issues()
    payload = ErrorResponse(
        code="dependency_not_ready",
        message="服务依赖未就绪",
        details=issues,
    )
    return JSONResponse(status_code=503, content=payload.model_dump())


def initialize_api_environment() -> None:
    load_runtime_environment()

