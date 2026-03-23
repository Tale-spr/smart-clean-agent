# Eval 模块使用说明

## 1. 模块定位

当前 `evaluation` 模块是一个**离线评测系统**，用于复用项目真实 Agent 链路，对样例集做规则校验、要点评估和可选的 Judge 评估。它不会接入 `Streamlit` 页面，也不会进入线上主链路。

当前默认评测两类不同性质的链路：

- 普通问答：显式 LangGraph ReAct 主链路
- 报告生成：独立高约束报告链路

因此新版 `evaluation` 已不再把所有任务都按同一套“严格工具顺序”来评，而是按任务类型分别计算指标。

当前实现目录：

- [run.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/run.py)
- [service.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/service.py)
- [judge.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/judge.py)
- [compare.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/compare.py)
- [migrate_dataset.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/migrate_dataset.py)

## 2. 三层评测结构

### 2.1 Rule-Based

规则层主要评估：

- 路由是否正确
- 必需工具是否完整
- 普通问答工具是否合理
- 报告链路关键依赖是否成立
- 检索模式是否符合预期
- required points 是否全部命中
- 报告时间引用是否一致

当前理解方式：

- 普通问答主要看 `tool_usage_valid`
- 报告生成主要看 `tool_dependency_valid + time_consistency_valid`

`tool_sequence_valid` 仍然保留，但对普通问答已经降级为观测指标，不再作为主要硬门槛。

### 2.2 Point-Based

内容层不再只做 `expected_keywords` 的布尔命中，而是改成 `required_points / optional_points`：

- `required_points`：全部命中才算 `content_pass=true`
- `optional_points`：只参与增强评分，不参与硬失败

每个 point 支持多个 `aliases`，命中任一别名即视为命中该点。

### 2.3 LLM-as-a-Judge

Judge 层是可选离线评估：

- 默认关闭
- 通过 `--with-judge` 开启
- 只用于离线评分，不进入线上推理

Judge 输出结构化结果：

- `correctness_score`
- `completeness_score`
- `groundedness_score`
- `tool_usage_score`
- `report_quality_score`
- `passed`
- `reason`

Judge 更适合发现这类问题：

- 答案结构完整，但证据不够扎实
- 答案命中了要点，但存在过度推断
- 报告写得很像样，但数据和月份不一致

## 3. 数据集格式

默认数据集为：

- [eval_cases.jsonl](/e:/Python/Agent项目/data/eval/eval_cases.jsonl)

格式已从旧版 `CSV` 升级为 `JSONL`，每行一条 JSON。

### 3.1 必填字段

- `case_id`
- `query`
- `category`
- `expected_route`
- `required_tools`
- `optional_tools`
- `required_points`
- `optional_points`
- `expected_retrieval_mode`

可选字段：

- `user_id`
- `city`
- `notes`
- `forbidden_tools`
- `allow_no_tool`
- `target_month`
- `allowed_trend_window`

### 3.2 枚举值约束

`category` 允许：

- `faq`
- `troubleshooting`
- `environment_fit`
- `report_generation`

`expected_route` 允许：

- `normal`
- `report`

`expected_retrieval_mode` 允许：

- `required`
- `optional`
- `forbidden`

### 3.3 新版字段语义

普通问答样例默认更关注：

- `content_pass`
- `tool_usage_valid`
- 是否调用了越界工具
- 是否出现明显冗余或重复工具调用
- 是否允许无工具直接回答

报告类样例默认更关注：

- `required_tools_present`
- `tool_dependency_valid`
- `time_consistency_valid`
- 报告内容质量

### 3.4 point 结构

每个 point 为对象：

```json
{
  "point_id": "required_01",
  "label": "保养建议",
  "aliases": ["保养建议", "维护建议", "建议"]
}
```

### 3.5 样例

```json
{
  "case_id": "env_005",
  "query": "广州湿度高，扫拖一体机要怎么保养？",
  "category": "environment_fit",
  "user_id": "1008",
  "city": "广州",
  "expected_route": "normal",
  "required_tools": ["get_weather"],
  "optional_tools": ["rag_summarize"],
  "required_points": [
    {
      "point_id": "required_01",
      "label": "湿度信息",
      "aliases": ["湿度", "潮湿", "湿度高"]
    },
    {
      "point_id": "required_02",
      "label": "保养建议",
      "aliases": ["保养", "维护", "清理", "晾干"]
    }
  ],
  "optional_points": [],
  "expected_retrieval_mode": "optional",
  "allow_no_tool": false,
  "notes": "天气后可补充保养知识"
}
```

## 4. 运行前置条件

运行评测前，至少确认：

1. 本地依赖已安装
2. 向量库已构建
3. `DASHSCOPE_API_KEY` 已配置
4. 如需天气相关链路更真实，建议配置 `AMAP_WEATHER_API_KEY`

当前默认模型角色如下：

- 在线主链路：`qwen-plus`
- RAG 总结：`qwen-plus`
- 批量离线评测：`qwen-flash`
- Judge：`qwen-plus`

先建库：

```powershell
python src/smart_clean_agent/rag/ingest.py
```

## 5. 运行方式

### 5.1 默认运行

```powershell
python src/smart_clean_agent/evaluation/run.py
```

### 5.2 指定数据集和输出目录

```powershell
python src/smart_clean_agent/evaluation/run.py `
  --dataset data/eval/eval_cases.jsonl `
  --output-dir data/eval/results
```

### 5.3 启用 Judge

```powershell
python src/smart_clean_agent/evaluation/run.py --with-judge
```

如果只想临时覆盖批量评测模型：

```powershell
python src/smart_clean_agent/evaluation/run.py --chat-model qwen-plus
```

指定 Judge 模型：

```powershell
python src/smart_clean_agent/evaluation/run.py `
  --with-judge `
  --judge-model qwen-plus
```

说明：

- `--chat-model` 只覆盖本轮评测执行模型，不影响线上主链路
- `--judge-model` 只覆盖 Judge，不影响评测执行模型
- 默认模型角色为：
  - 评测执行：`qwen-flash`
  - Judge：`qwen-plus`
- 当前实现只支持切换模型名，不开放 `enable_thinking` 等推理参数

## 6. 输出结构

每次评测会生成：

- `eval_results_时间戳.json`
- `eval_results_时间戳.csv`

目录默认是：

- [results](/e:/Python/Agent项目/data/eval/results)

### 6.1 JSON 顶层结构

```json
{
  "generated_at": "...",
  "rule_based_summary": {},
  "judge_based_summary": {},
  "normal_summary": {},
  "report_summary": {},
  "results": []
}
```

### 6.2 单条 result 结构

每条结果包含：

- 基础字段：
  - `case_id`
  - `query`
  - `category`
  - `expected_route`
  - `execution_mode`
  - `actual_tools`
  - `answer`
- `rule_based`
- `judge_based`

### 6.3 Rule-Based 关键字段

- `route_correct`
- `required_tools_present`
- `missing_required_tools`
- `unexpected_tools`
- `retrieval_mode_valid`
- `required_point_hit_rate`
- `optional_point_hit_rate`
- `missing_required_points`
- `content_pass`
- `tool_sequence_valid`
- `tool_usage_valid`
- `unnecessary_tool_calls`
- `repeated_tool_calls`
- `tool_dependency_valid`
- `time_consistency_valid`
- `step_count`
- `stop_reason`

字段理解建议：

- 普通问答优先看：
  - `content_pass`
  - `tool_usage_valid`
  - `retrieval_mode_valid`
- 报告生成优先看：
  - `required_tools_present`
  - `tool_dependency_valid`
  - `time_consistency_valid`
  - `content_pass`

### 6.4 summary 指标

规则层 summary：

- `route_correct_rate`
- `required_tool_pass_rate`
- `tool_sequence_valid_rate`
- `retrieval_mode_valid_rate`
- `content_pass_rate`
- `required_point_hit_rate_avg`
- `optional_point_hit_rate_avg`
- `report_generation_success_rate`

普通问答 summary：

- `content_pass_rate`
- `tool_usage_valid_rate`
- `unexpected_tool_rate`
- `required_point_hit_rate_avg`
- `judge_pass_rate`

报告 summary：

- `required_tools_present_rate`
- `tool_dependency_valid_rate`
- `time_consistency_valid_rate`
- `content_pass_rate`
- `avg_groundedness_score`
- `avg_report_quality_score`

Judge summary：

- `judge_pass_rate`
- `avg_correctness_score`
- `avg_completeness_score`
- `avg_groundedness_score`
- `avg_tool_usage_score`
- `avg_report_quality_score`

推荐解读方式：

- `normal_summary` 用来判断普通问答 ReAct 是否稳定
- `report_summary` 用来判断报告链路是否 grounded、是否时间一致
- `judge_based_summary` 用来判断答案语义质量，尤其适合发现“看起来答对，但证据不够”的问题

## 7. 对比两个版本

可以用 [compare.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/compare.py) 对比两个评测结果文件。

```powershell
python src/smart_clean_agent/evaluation/compare.py `
  data/eval/results/before.json `
  data/eval/results/after.json
```

可选输出到文件：

```powershell
python src/smart_clean_agent/evaluation/compare.py `
  data/eval/results/before.json `
  data/eval/results/after.json `
  --output data/eval/results/comparison.json
```

对比结果包含：

- `rule_based_summary_diff`
- `judge_based_summary_diff`
- `normal_summary_diff`
- `report_summary_diff`
- `improved_cases`
- `regressed_cases`

## 8. 旧版 CSV 迁移

如果你还有旧版 `eval_cases.csv`，可以先迁移：

```powershell
python src/smart_clean_agent/evaluation/migrate_dataset.py
```

或指定路径：

```powershell
python src/smart_clean_agent/evaluation/migrate_dataset.py `
  --input data/eval/eval_cases.csv `
  --output data/eval/eval_cases.jsonl
```

注意：

- 迁移脚本是**一次性转换入口**
- 生成结果是机械迁移版本
- 迁移后建议手动补充 `required_tools / optional_tools / aliases / expected_retrieval_mode`

## 9. 当前推荐使用方式

### 9.1 日常回归

每做完一轮优化后，先跑：

```powershell
python src/smart_clean_agent/evaluation/run.py
```

这条命令默认会用 `qwen-flash`，更适合低成本批量回归，重点看：

- 主流程是否回退
- 普通问答 `normal_summary`
- 报告链路 `report_summary`

### 9.2 更高质量判断

当你觉得规则层已经不足以反映真实效果时，再加：

```powershell
python src/smart_clean_agent/evaluation/run.py --with-judge
```

Judge 默认使用 `qwen-plus`，更适合：

- 判断答案语义质量
- 发现 groundedness 问题
- 判断“流程差不多，但答案是否真的可信”

### 9.3 大版本对比

大改动后建议两次跑结果，再执行：

```powershell
python src/smart_clean_agent/evaluation/compare.py before.json after.json
```

## 10. 常见问题

### 10.1 提示数据集文件不存在

确认你当前在项目根目录，并且：

- [eval_cases.jsonl](/e:/Python/Agent项目/data/eval/eval_cases.jsonl) 存在

### 10.2 提示 category / route / retrieval mode 非法

说明 JSONL case 不符合 schema，需要检查字段值。

### 10.3 报告类通过率很低

优先检查：

- `execution_mode`
- `route_correct`
- `missing_required_tools`
- `tool_dependency_valid`
- `time_consistency_valid`
- `missing_required_points`

### 10.4 Judge 失败

优先检查：

- `DASHSCOPE_API_KEY`
- Judge 模型是否可用
- Judge 输出是否为合法 JSON

Judge 现在已经做了更宽松的结构化解析，但如果模型输出完全偏离 JSON，仍可能出现单条 case 的 Judge 失败。

## 11. 当前结论

新版 `evaluation` 已不再只是“关键词命中脚本”，而是：

- 规则层校验链路正确性
- point 层校验内容覆盖
- 可选 judge 层校验语义质量

同时评测规则已经按任务类型区分：

- 普通问答：结果正确性和工具合理性优先
- 报告生成：关键数据依赖和时间一致性优先

这让它更适合用来比较：

- 普通问答 ReAct 的稳定性
- 报告链路的关键依赖和时间一致性
- 优化前后内容质量的真实变化
