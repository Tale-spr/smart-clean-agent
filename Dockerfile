FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

COPY src ./src
COPY config ./config
COPY data ./data
COPY docs ./docs
COPY prompts ./prompts

EXPOSE 8000

CMD ["uvicorn", "smart_clean_agent.api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
