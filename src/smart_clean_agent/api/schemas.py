from pydantic import BaseModel, Field


class StatusEventResponse(BaseModel):
    event_type: str
    title: str
    detail: str
    created_at: str
    level: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    dependencies_ready: bool
    missing_dependencies: list[str] | None = None


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    answer: str
    status_events: list[StatusEventResponse]
    session_summary: str


class ReportRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    session_id: str | None = None


class ReportResponse(BaseModel):
    user_id: str
    session_id: str
    report: str
    status_events: list[StatusEventResponse]
    report_memory_summary: str

