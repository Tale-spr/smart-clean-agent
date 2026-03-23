# 智扫通 Agent 项目

一个围绕“扫地机器人智能客服 + 使用报告生成”场景构建的 Agent 项目。项目同时提供 `Streamlit` 演示界面、`FastAPI` 服务接口、RAG 检索、用户记忆、离线评测和 Docker 部署入口。

## 项目亮点

- 支持普通问答、天气查询、知识检索、报告生成、多轮上下文记忆
- 普通问答采用显式 LangGraph ReAct 工作流，支持“分析问题 -> 调工具 -> 观察结果 -> 生成回答”
- 报告生成使用独立链路，强调用户上下文、外部数据、趋势分析和时间一致性
- 提供用户长期记忆和多月趋势记忆，支持更连续的个性化建议
- 提供离线评测能力，支持规则评测、要点评测和可选 Judge 评分
- `Streamlit` 与 `FastAPI` 共享同一业务服务层，便于演示和服务化部署

## 效果截图

![聊天演示截图占位](docs/images/chat-home.png)
![报告生成截图占位](docs/images/report-demo.png)
![API 健康检查截图占位](docs/images/api-health.png)

## 技术栈

- Python 3.13
- Streamlit
- FastAPI
- LangChain / LangGraph
- Chroma
- DashScope
- Requests
- Docker

## 目录结构

```text
.
├─config/                         # 配置文件
├─data/                           # 知识库、用户资料、外部记录、会话、评测数据
├─docs/                           # 使用文档与截图
├─prompts/                        # 提示词模板
├─src/
│  └─smart_clean_agent/
│     ├─agent/                    # Agent、tools、middleware
│     ├─api/                      # FastAPI 入口与 schema
│     ├─evaluation/               # 离线评测
│     ├─model/                    # 模型工厂
│     ├─rag/                      # RAG 与向量库
│     ├─services/                 # 会话、记忆、依赖校验、聊天服务
│     ├─ui/                       # Streamlit UI 组件
│     ├─utils/                    # 通用工具
│     └─web/                      # Streamlit 入口与 bootstrap
├─tests/                          # 测试
├─Dockerfile
├─pyproject.toml
├─README.md
└─requirements.txt
```

## 模型配置

默认模型角色配置在 [rag.yml](config/rag.yml)：

- 在线主链路：`qwen-plus`
- RAG 总结：`qwen-plus`
- 离线批量评测：`qwen-flash`
- Judge：`qwen-plus`
- Embedding：`text-embedding-v4`

当前实现只支持按角色切换模型名，不包含 `enable_thinking`、温度、`top_p` 等额外推理参数配置。

## 本地运行

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. 配置环境变量

在根目录创建 `.env`，或直接配置系统环境变量，至少需要：

- `DASHSCOPE_API_KEY`
- `AMAP_WEATHER_API_KEY`（可选）

示例：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
AMAP_WEATHER_API_KEY=your_amap_weather_api_key
```

### 3. 构建向量库

```powershell
python src/smart_clean_agent/rag/ingest.py
```

### 4. 启动 Streamlit

```powershell
streamlit run src/smart_clean_agent/web/app.py
```

### 5. 启动 FastAPI

```powershell
uvicorn smart_clean_agent.api.main:app --app-dir src --reload
```

## API 说明

当前提供 3 个接口：

- `GET /health`
- `POST /chat`
- `POST /report`

启动后可访问：

- Swagger 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

详细接口说明见 [API 使用说明](docs/api_usage_guide.md)。

### 聊天接口示例

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body '{"user_id":"1001","message":"北京今天天气怎么样？"}'
```

### 报告接口示例

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/report" `
  -ContentType "application/json" `
  -Body '{"user_id":"1001","query":"生成我的本月使用报告"}'
```

## 离线评测

运行：

```powershell
python src/smart_clean_agent/evaluation/run.py
```

启用 Judge：

```powershell
python src/smart_clean_agent/evaluation/run.py --with-judge
```

如果需要临时覆盖评测模型：

```powershell
python src/smart_clean_agent/evaluation/run.py --chat-model qwen-plus
```

如果需要单独指定 Judge 模型：

```powershell
python src/smart_clean_agent/evaluation/run.py --with-judge --judge-model qwen-turbo
```

评测结果默认输出到 `data/eval/results/`，详细说明见 [Eval 使用说明](docs/eval_usage_guide.md)。

## Docker 部署

### 1. 构建镜像

```powershell
docker build -t agent-service .
```

### 2. 启动容器

```powershell
docker run -p 8000:8000 --env-file .env agent-service
```

说明：

- 仓库默认不上传 `chroma_db`
- 启动 Docker 前请先在本地完成 `python src/smart_clean_agent/rag/ingest.py`
- 如果镜像里也需要可用向量库，请在部署方案中额外挂载或打包向量库产物

## 当前能力

- 用户资料管理与会话切换
- 普通客服问答
- 天气工具调用
- RAG 检索问答
- 使用报告生成
- 长期用户记忆
- 多月趋势记忆
- Streamlit 演示界面
- FastAPI 标准服务接口
- Docker 部署入口
- 离线评测与结果对比

## 注意事项

- 当前项目默认使用文件型存储，不是数据库方案
- 当前 Docker 方案是单容器部署，不包含多服务编排
- 仓库默认不提交 `.env`、`chroma_db`、日志、评测结果和本地 memory 运行产物

## 相关文档

- [API 使用说明](docs/api_usage_guide.md)
- [Eval 使用说明](docs/eval_usage_guide.md)
