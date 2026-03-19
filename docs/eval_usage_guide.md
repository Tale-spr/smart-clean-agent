# Eval 模块使用说明书

## 1. 文档目的

本文档用于说明当前项目中 `eval` 模块的作用、使用方式、数据格式、指标含义、输出结果和常见问题，便于你后续：

- 在优化前后做效果对比
- 向面试官展示“我不仅会做 Agent，还会评估 Agent”
- 持续扩充评测集并沉淀成自己的项目方法论

当前 `evaluation` 模块是一个**离线评测工具**，不会接入 Streamlit 页面，而是通过命令行独立运行。

---

## 2. 模块定位

`evaluation` 模块的目标不是做复杂 benchmark，也不是做 LLM-as-a-judge，而是提供一套**最小可用、规则透明、可持续迭代**的评测能力。

它会复用当前项目的真实 Agent 链路，对评测样例逐条执行，并产出以下几类信息：

- 模型最终回答内容
- 工具调用情况
- RAG 检索是否命中
- 关键词是否命中
- 报告类请求是否成功生成
- 最终汇总指标

当前实现入口：

- [run.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/run.py)
- [service.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/service.py)

---

## 3. 当前实现结构

### 3.1 主要文件

- [run.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/run.py)
  负责命令行入口、初始化评测执行器、运行整套评测、输出结果文件路径。

- [service.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/service.py)
  负责：
  - 读取评测样例
  - 校验数据格式
  - 单条评测逻辑
  - 汇总指标计算
  - JSON / CSV 结果输出

- [eval_cases.csv](/e:/Python/Agent项目/data/eval/eval_cases.csv)
  当前默认评测数据集。

- [results](/e:/Python/Agent项目/data/eval/results)
  评测结果输出目录。

### 3.2 当前覆盖的场景类别

当前 `category` 允许值只有这 4 类：

- `faq`
- `troubleshooting`
- `environment_fit`
- `report_generation`

如果写入其他类别，评测会直接报错。

---

## 4. 运行前置条件

在运行 `evaluation` 模块前，建议先确认以下条件：

### 4.1 本地向量库已经构建完成

`eval` 会调用真实 RAG 链路，所以必须先完成离线建库。

```powershell
python src/smart_clean_agent/rag/ingest.py
```

如果本地向量库不存在，评测会失败。

### 4.2 环境变量已配置

至少需要：

- `DASHSCOPE_API_KEY`

可选但推荐：

- `AMAP_WEATHER_API_KEY`

说明：

- FAQ / 故障排查 / RAG 类问题依赖大模型能力，因此 `DASHSCOPE_API_KEY` 基本是必需的。
- 天气类评测如果没有配置高德天气 Key，天气工具会返回“天气服务未配置”，这会影响环境类评测表现。

### 4.3 当前环境允许联网调用模型

`eval` 不是纯本地 mock 测试，它会真实调用：

- 大模型
- 天气 API（部分样例）

如果网络受限，评测可能失败或结果偏差较大。

---

## 5. 标准运行方式

### 5.1 默认运行命令

在项目根目录执行：

```powershell
python src/smart_clean_agent/evaluation/run.py
```

该命令会：

1. 读取默认评测数据集 [eval_cases.csv](/e:/Python/Agent项目/data/eval/eval_cases.csv)
2. 初始化真实 Agent、RAG、工具链
3. 按样例逐条执行
4. 统计汇总指标
5. 在 [results](/e:/Python/Agent项目/data/eval/results) 下生成 `JSON` 和 `CSV` 两份结果

### 5.2 命令成功时控制台输出示例

输出大致包含：

- 总样例数
- `answer_keyword_hit_rate`
- `tool_call_success_rate`
- `retrieval_hit_rate`
- `report_generation_success_rate`
- 结果文件路径

### 5.3 返回码约定

- 成功：返回 `0`
- 失败：返回 `1`

失败时会在标准错误输出打印：

```text
离线评测失败: ...
```

---

## 6. 评测数据集格式说明

### 6.1 默认数据集路径

默认读取：

- [eval_cases.csv](/e:/Python/Agent项目/data/eval/eval_cases.csv)

### 6.2 必填字段

评测数据文件至少要包含以下字段：

- `query`
- `category`
- `expected_keywords`
- `expected_tool`
- `expected_retrieval_hit`

如果缺少其中任意字段，评测会报错：

```text
评测数据文件缺少必要字段
```

### 6.3 当前支持的可选字段

为了让评测尽量贴近真实会话上下文，当前还支持这些字段：

- `case_id`
- `user_id`
- `city`

如果不填：

- `case_id` 会自动生成 `case_001`、`case_002` 这种形式
- `user_id` 默认使用 `1001`
- `city` 默认使用 `北京`

### 6.4 字段含义说明

#### `case_id`
评测样例唯一标识，用于结果定位。

#### `query`
用户提问内容，会原样传给 Agent。

#### `category`
样例类别，当前只允许：

- `faq`
- `troubleshooting`
- `environment_fit`
- `report_generation`

#### `expected_keywords`
期望回答中包含的关键词，多个关键词用 `|` 分隔。

例如：

```csv
激光导航|视觉导航
```

当前实现是**大小写不敏感的子串匹配**。只要回答中缺少任意一个关键词，就会判定该样例 `answer_keyword_hit = false`。

#### `expected_tool`
期望至少调用一次的工具名。

例如：

- `rag_summarize`
- `get_weather`
- `get_user_location`
- `fill_context_for_report`
- `fetch_external_data`

当前规则是：

- 只检查“是否出现在实际工具调用列表里”
- 不检查调用次数
- 不检查工具调用顺序

#### `expected_retrieval_hit`
是否期望本条样例发生 RAG 检索命中。

支持的真值：

- `true`
- `1`
- `yes`
- `y`
- `是`

支持的假值：

- `false`
- `0`
- `no`
- `n`
- `否`
- 空字符串

### 6.5 当前数据集样例结构示例

```csv
case_id,query,category,expected_keywords,expected_tool,expected_retrieval_hit,user_id,city
faq_001,激光导航和视觉导航有什么区别？,faq,激光导航|视觉导航,rag_summarize,true,1001,北京
env_001,北京今天天气适合让机器人拖地吗？,environment_fit,天气|拖地,get_weather,false,1001,北京
report_001,请根据我2025-03的使用记录生成一份报告。,report_generation,报告|建议,fill_context_for_report,false,1001,北京
```

---

## 7. 当前评测执行逻辑

### 7.1 执行器初始化

[run.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/run.py) 中的 `AgentEvaluationExecutor` 会显式初始化：

- 聊天模型
- Embedding 模型
- VectorStoreService
- RagSummarizeService
- Agent 工具集
- ReactAgent

这意味着 `eval` 走的是当前项目的**真实业务链路**，而不是伪造结果。

### 7.2 每条样例执行时会注入的上下文

每个样例会生成独立的 `runtime_context`，当前包含：

- `report`
- `user_id`
- `city`
- `session_summary`
- `recent_history`
- `trace_tool_calls`

其中：

- `user_id` 和 `city` 来自数据集字段
- `session_summary` 和 `recent_history` 当前默认为空字符串
- `trace_tool_calls` 用于记录工具调用痕迹

### 7.3 工具调用埋点

当前工具调用埋点在 [middleware.py](/e:/Python/Agent项目/src/smart_clean_agent/agent/tools/middleware.py)。

机制是：

- 每次工具调用前，middleware 会把工具名写入 `runtime_context["trace_tool_calls"]`
- 最终评测结果会据此得到 `actual_tools`

### 7.4 检索埋点

当前 RAG 检索埋点在 [rag_service.py](/e:/Python/Agent项目/src/smart_clean_agent/rag/rag_service.py)。

机制是：

- 每次 `rag_summarize` 检索到文档时，会把文档信息写入 trace
- 最终如果 `retrieved_docs` 非空，则认为本条样例 `actual_retrieval_hit = true`

注意：

当前“检索命中”的定义是**是否检索到至少一条文档**，不是“是否检索到真正正确的文档”。这是第一版的简化实现。

---

## 8. 指标定义说明

当前共有 4 个核心指标。

### 8.1 `answer_keyword_hit_rate`

定义：

- 全部样例中，回答命中全部期望关键词的比例

计算方式：

```text
命中全部 expected_keywords 的样例数 / 总样例数
```

特点：

- 简单直接
- 可解释性强
- 但不能完整衡量回答质量，只能作为第一版粗粒度指标

### 8.2 `tool_call_success_rate`

定义：

- 所有配置了 `expected_tool` 的样例中，实际调用了该工具的比例

计算方式：

```text
tool_call_checked = true 且 tool_call_success = true 的样例数 / 配置了 expected_tool 的样例数
```

说明：

- 如果某条样例 `expected_tool` 为空，则不会计入这个分母
- 当前只判断是否调用过，不判断顺序和次数

### 8.3 `retrieval_hit_rate`

定义：

- 所有 `expected_retrieval_hit = true` 的样例中，实际发生 RAG 检索命中的比例

计算方式：

```text
expected_retrieval_hit = true 且 actual_retrieval_hit = true 的样例数 / expected_retrieval_hit = true 的样例数
```

注意：

这个指标当前不是“预测是否匹配”的严格 accuracy，而是偏向“该检索的样例里，有多少真的检索到了文档”。

### 8.4 `report_generation_success_rate`

定义：

- 报告类样例中，满足报告生成成功条件的比例

当前成功条件有 3 个：

1. 回答非空
2. 命中全部 `expected_keywords`
3. 命中 `expected_tool`

也就是说，当前**报告成功率并不会单独检查报告格式是否漂亮、逻辑是否完整**，而是规则化最小判断。

---

## 9. 结果文件说明

每次评测完成后，会在 [results](/e:/Python/Agent项目/data/eval/results) 下生成两份文件：

- `eval_results_时间戳.json`
- `eval_results_时间戳.csv`

### 9.1 JSON 文件结构

JSON 顶层包含：

- `generated_at`
- `summary`
- `results`

其中：

- `summary` 是整次评测的汇总指标
- `results` 是逐条样例的详细结果

### 9.2 CSV 文件字段

当前导出的 CSV 字段包括：

- `case_id`
- `category`
- `query`
- `answer_keyword_hit`
- `missing_keywords`
- `expected_tool`
- `actual_tools`
- `tool_call_checked`
- `tool_call_success`
- `expected_retrieval_hit`
- `actual_retrieval_hit`
- `retrieval_expectation_met`
- `report_generation_success`
- `user_id`
- `city`
- `answer`

### 9.3 最适合重点关注的字段

你做优化前后对比时，最建议重点看这些字段：

- `summary.answer_keyword_hit_rate`
- `summary.tool_call_success_rate`
- `summary.retrieval_hit_rate`
- `summary.report_generation_success_rate`
- `results[].missing_keywords`
- `results[].actual_tools`
- `results[].answer`

---

## 10. 如何扩展评测集

### 10.1 推荐扩展方式

建议按照当前分类继续扩充：

- `faq`
  关注基础知识问答
- `troubleshooting`
  关注故障排查、修复建议
- `environment_fit`
  关注天气、湿度、地面环境适配
- `report_generation`
  关注用户月报、趋势描述、耗材建议

### 10.2 写样例时的建议

#### 关键词不要写得太死

不建议把 `expected_keywords` 写成完整句子，建议写高信息密度的短词。

推荐：

```text
滤网|清理
```

不推荐：

```text
建议每周清理一次滤网并保持风道通畅
```

原因是当前规则是子串匹配，长句非常容易误判失败。

#### 工具期望要写“关键工具”

例如一条环境类问题，真正想验证的是是否调用了 `get_weather`，那就只写这个关键工具，不要试图把所有中间步骤都写进去。

#### 报告类样例建议明确月份

当前 `records.csv` 里是 2025 年月份数据，而系统当前时间可能不是 2025 年。

因此报告类评测**最好显式写月份**，例如：

```text
请根据我2025-03的使用记录生成一份报告。
```

这样可以避免模型走“当前月份”分支时，因为时间漂移导致取不到数据。

---

## 11. 当前实现边界与局限

### 11.1 不是严格意义的回答质量评审

当前实现只做：

- 关键词命中
- 工具命中
- 检索是否发生
- 报告是否最小成功

它还不能判断：

- 回答是否真的专业
- 回答是否存在幻觉
- 报告结构是否优雅
- 建议是否足够有针对性

### 11.2 `retrieval_hit_rate` 仍然比较粗糙

当前只要检索到了文档，就视为命中。

它还没有检查：

- 检索到的文档是不是最相关
- 文档来源是否正确
- 文档内容是否真正支持最终回答

### 11.3 工具调用评测不检查顺序

例如报告类请求实际上有“固定工具调用流程”约束，但当前评测只检查关键工具有没有出现，不检查：

- 调用顺序
- 调用次数
- 中间参数是否完全正确

### 11.4 当前没有命令行参数解析

虽然 `main(dataset_path=None, output_dir=None)` 支持传参，但当前并没有用 `argparse` 做标准 CLI。

也就是说，目前标准用法只有：

```powershell
python src/smart_clean_agent/evaluation/run.py
```

如果你想临时换数据集或输出目录，需要用 Python 调用 `main()`，例如：

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
python -c "from smart_clean_agent.evaluation.run import main; raise SystemExit(main('data/eval/eval_cases.csv', 'data/eval/results'))"
```

---

## 12. 常见失败原因排查

### 12.1 提示“评测数据文件不存在”

原因：

- 数据集路径错误
- 当前工作目录不在项目根目录

排查方式：

- 确认你在项目根目录执行命令
- 确认 [eval_cases.csv](/e:/Python/Agent项目/data/eval/eval_cases.csv) 存在

### 12.2 提示“评测数据文件缺少必要字段”

原因：

- CSV 表头不完整
- 字段名拼写不一致

需要至少包含：

- `query`
- `category`
- `expected_keywords`
- `expected_tool`
- `expected_retrieval_hit`

### 12.3 提示“category非法”

原因：

- 写入了不支持的分类名

当前只允许：

- `faq`
- `troubleshooting`
- `environment_fit`
- `report_generation`

### 12.4 提示“本地向量库不存在”

原因：

- 没有先运行建库命令

先执行：

```powershell
python src/smart_clean_agent/rag/ingest.py
```

### 12.5 提示模型或天气相关错误

原因通常包括：

- `DASHSCOPE_API_KEY` 未配置
- `AMAP_WEATHER_API_KEY` 未配置
- 网络不可用
- 第三方接口暂时不可用

---

## 13. 推荐使用方式

对于你当前这个项目，我建议这样使用 `eval` 模块：

### 13.1 每做完一轮优化，就跑一次评测

例如：

- 优化 RAG 检索前跑一次
- 优化 RAG prompt 后再跑一次
- 优化工具调用策略后再跑一次

这样你就能形成“优化前 / 优化后”对照。

### 13.2 把结果文件保留下来

建议保留关键版本的 `JSON` 和 `CSV` 结果，这些非常适合：

- 写简历项目描述
- 做面试项目复盘
- 解释你做过哪些优化以及结果如何变化

### 13.3 后续可继续迭代的方向

后续如果你继续升级 `eval` 模块，最值得做的方向有：

- 为报告类请求增加工具调用顺序校验
- 为 RAG 增加“来源文档正确率”校验
- 为回答增加更细粒度规则评分
- 引入人工标注结果对照
- 后续再考虑接入 LLM-as-a-judge

---

## 14. 当前结论

当前 `eval` 模块已经具备以下价值：

- 能跑真实 Agent 链路
- 能使用真实样例集做离线评测
- 能输出可解释的规则化指标
- 能生成 JSON / CSV 结果文件
- 能作为项目“工程化 + 可评测”的重要展示点

对你现在这个阶段来说，它已经足够支撑这样一种项目叙事：

> 我不仅实现了 Agent + RAG + 工具调用，还为系统补了一套离线评测机制，能够从关键词命中、工具调用命中、检索命中和报告生成成功率等维度，量化比较不同优化版本的效果。
