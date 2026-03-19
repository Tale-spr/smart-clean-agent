# FastAPI 使用说明

## 概述

当前项目除了 `Streamlit` 演示界面外，还提供了一套标准 HTTP API：

- `GET /health`
- `POST /chat`
- `POST /report`

默认启动命令：

```powershell
uvicorn smart_clean_agent.api.main:app --app-dir src --reload
```

默认访问地址：

```text
http://127.0.0.1:8000
```

## 前置条件

启动 API 前，请先确认：

1. 已安装项目依赖：

```powershell
pip install -r requirements.txt
```

2. 已配置 `.env`：

- `DASHSCOPE_API_KEY`
- `AMAP_WEATHER_API_KEY` 可选

3. 已完成离线建库：

```powershell
python src/smart_clean_agent/rag/ingest.py
```

## 接口说明

### 1. 健康检查

请求：

```http
GET /health
```

响应示例：

```json
{
  "status": "healthy",
  "vector_store_ready": true,
  "dependencies_ready": true,
  "missing_dependencies": null
}
```

如果依赖缺失：

```json
{
  "status": "unhealthy",
  "vector_store_ready": false,
  "dependencies_ready": false,
  "missing_dependencies": [
    "未配置 DASHSCOPE_API_KEY",
    "本地向量库不存在: E:\\Python\\Agent项目\\chroma_db\\chroma.sqlite3"
  ]
}
```

### 2. 聊天接口

请求：

```http
POST /chat
Content-Type: application/json
```

请求体：

```json
{
  "user_id": "1001",
  "message": "北京今天天气怎么样？",
  "session_id": "20260318_220000_123456"
}
```

其中：

- `user_id` 必填
- `message` 必填
- `session_id` 选填；不传时会为当前请求创建新会话

响应示例：

```json
{
  "user_id": "1001",
  "session_id": "20260318_220000_123456",
  "answer": "根据查询结果，北京当前天气晴，适合安排日常清扫。",
  "status_events": [
    {
      "event_type": "stage.memory",
      "title": "正在整理历史记忆",
      "detail": "正在整理当前会话摘要、长期记忆与趋势记忆",
      "created_at": "2026-03-18T22:00:00",
      "level": "info"
    }
  ],
  "session_summary": "用户咨询北京天气，并询问是否适合安排清扫。"
}
```

### 3. 报告接口

请求：

```http
POST /report
Content-Type: application/json
```

请求体：

```json
{
  "user_id": "1001",
  "query": "生成我的本月使用报告",
  "session_id": "20260318_220000_123456"
}
```

响应示例：

```json
{
  "user_id": "1001",
  "session_id": "20260318_220000_123456",
  "report": "# 黑马程序员扫地机器人使用情况报告与保养建议\\n...",
  "status_events": [
    {
      "event_type": "stage.report",
      "title": "正在生成使用报告",
      "detail": "正在切换到报告生成上下文",
      "created_at": "2026-03-18T22:00:00",
      "level": "info"
    }
  ],
  "report_memory_summary": "覆盖月份: 2026-01、2026-02、2026-03"
}
```

## 常见错误

### 1. 服务依赖未就绪

状态码：

```text
503
```

响应示例：

```json
{
  "code": "dependency_not_ready",
  "message": "服务依赖未就绪",
  "details": [
    "未配置 DASHSCOPE_API_KEY"
  ]
}
```

### 2. 用户不存在

状态码：

```text
404
```

响应示例：

```json
{
  "code": "user_not_found",
  "message": "用户资料不存在",
  "details": null
}
```

### 3. 参数缺失

状态码：

```text
422
```

此类错误由 FastAPI 自动返回，用于提示缺少 `user_id`、`message`、`query` 等必填字段。

## Docker 启动方式

```powershell
docker build -t agent-service .
docker run -p 8000:8000 --env-file .env agent-service
```

容器启动后同样使用：

```text
http://127.0.0.1:8000/health
```
