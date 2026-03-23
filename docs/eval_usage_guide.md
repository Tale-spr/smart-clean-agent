# Eval 模块使用说明

## 1. 模块定位

当前 `evaluation` 模块是一个**离线评测系统**，用于复用项目的真实 Agent 链路，对样例集做规则校验、要点评估和可选的 Judge 评估。它不会接入 `Streamlit` 页面，也不会进入线上主链路。

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
- 工具顺序是否合理
- 检索模式是否符合预期
- required points 是否全部命中
- 报告链路是否完整走通

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

### 3.3 point 结构

每个 point 为对象：

```json
{
  "point_id": "required_01",
  "label": "保养建议",
  "aliases": ["保养建议", "维护建议", "建议"]
}
```

### 3.4 样例

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
  "notes": "天气后可补充保养知识"
}
```

## 4. 运行前置条件

运行评测前，至少确认：

1. 本地依赖已安装
2. 向量库已构建
3. `DASHSCOPE_API_KEY` 已配置
4. 如需天气相关链路更真实，建议配置 `AMAP_WEATHER_API_KEY`

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

指定 Judge 模型：

```powershell
python src/smart_clean_agent/evaluation/run.py `
  --with-judge `
  --judge-model qwen-plus
```

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
- `step_count`
- `stop_reason`

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

Judge summary：

- `judge_pass_rate`
- `avg_correctness_score`
- `avg_completeness_score`
- `avg_groundedness_score`
- `avg_tool_usage_score`
- `avg_report_quality_score`

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

### 9.2 大版本对比

大改动后建议两次跑结果，再执行：

```powershell
python src/smart_clean_agent/evaluation/compare.py before.json after.json
```

### 9.3 更高质量判断

当你觉得规则层已经不足以反映真实效果时，再加：

```powershell
python src/smart_clean_agent/evaluation/run.py --with-judge
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
- `tool_sequence_valid`
- `missing_required_points`

### 10.4 Judge 失败

优先检查：

- `DASHSCOPE_API_KEY`
- Judge 模型是否可用
- Judge 输出是否为合法 JSON

## 11. 当前结论

新版 `evaluation` 已不再只是“关键词命中脚本”，而是：

- 规则层校验链路正确性
- point 层校验内容覆盖
- 可选 judge 层校验语义质量

这让它更适合用来比较：

- 普通问答 ReAct 的稳定性
- 报告链路的工具顺序
- 优化前后内容质量的真实变化
