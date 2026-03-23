# Eval 模块使用说明

## 1. 模块定位

`evaluation` 是项目的离线评测模块，用于复用真实 Agent 链路，对样例集执行自动化评估。它不接入 `Streamlit` 页面，也不会进入线上主链路。

模块目录：

- [run.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/run.py)
- [service.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/service.py)
- [judge.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/judge.py)
- [compare.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/compare.py)
- [migrate_dataset.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/migrate_dataset.py)

## 2. 评测结构

当前评测由三层组成：

### 2.1 Rule-Based

规则层负责评估链路行为和结构化结果，主要关注：

- 路由是否正确
- 必需工具是否完整
- 工具使用是否合理
- 检索模式是否符合预期
- required points 是否命中
- 报告类时间引用是否一致

### 2.2 Point-Based

内容层不再依赖单纯关键词命中，而是改成 `required_points / optional_points`：

- `required_points`：全部命中才算 `content_pass=true`
- `optional_points`：只参与增强评分，不参与硬失败

每个 point 支持多个 `aliases`，命中任一别名即视为命中。

### 2.3 LLM-as-a-Judge

Judge 层是可选语义评估：

- 默认关闭
- 通过 `--with-judge` 开启
- 只用于离线评分，不进入线上推理

Judge 输出：

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

格式为 `JSONL`，每行一条 case。

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

### 3.2 可选字段

- `user_id`
- `city`
- `notes`
- `forbidden_tools`
- `allow_no_tool`
- `target_month`
- `allowed_trend_window`

### 3.3 枚举约束

`category`：

- `faq`
- `troubleshooting`
- `environment_fit`
- `report_generation`

`expected_route`：

- `normal`
- `report`

`expected_retrieval_mode`：

- `required`
- `optional`
- `forbidden`

### 3.4 point 结构

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
  "allow_no_tool": false
}
```

## 4. 运行前置条件

运行评测前，至少确认：

1. 本地依赖已安装
2. 向量库已构建
3. `DASHSCOPE_API_KEY` 已配置
4. 如需天气链路更真实，建议配置 `AMAP_WEATHER_API_KEY`

先建库：

```powershell
python src/smart_clean_agent/rag/ingest.py
```

当前默认模型角色：

- 在线主链路：`qwen-plus`
- RAG 总结：`qwen-plus`
- 批量离线评测：`qwen-flash`
- Judge：`qwen-plus`

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

### 5.4 覆盖评测模型

```powershell
python src/smart_clean_agent/evaluation/run.py --chat-model qwen-plus
```

### 5.5 指定 Judge 模型

```powershell
python src/smart_clean_agent/evaluation/run.py `
  --with-judge `
  --judge-model qwen-plus
```

说明：

- `--chat-model` 只覆盖本轮评测执行模型
- `--judge-model` 只覆盖 Judge 模型
- 当前实现只支持切换模型名，不开放 `enable_thinking` 等推理参数

## 6. 输出结构

每次评测会生成：

- `eval_results_时间戳.json`
- `eval_results_时间戳.csv`

默认输出目录：

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

## 7. summary 指标说明

### 7.1 rule_based_summary

- `route_correct_rate`
- `required_tool_pass_rate`
- `tool_sequence_valid_rate`
- `retrieval_mode_valid_rate`
- `content_pass_rate`
- `required_point_hit_rate_avg`
- `optional_point_hit_rate_avg`
- `report_generation_success_rate`

### 7.2 normal_summary

主要用于观察普通问答链路：

- `content_pass_rate`
- `tool_usage_valid_rate`
- `unexpected_tool_rate`
- `required_point_hit_rate_avg`
- `judge_pass_rate`

### 7.3 report_summary

主要用于观察报告链路：

- `required_tools_present_rate`
- `tool_dependency_valid_rate`
- `time_consistency_valid_rate`
- `content_pass_rate`
- `avg_groundedness_score`
- `avg_report_quality_score`

### 7.4 judge_based_summary

- `judge_pass_rate`
- `avg_correctness_score`
- `avg_completeness_score`
- `avg_groundedness_score`
- `avg_tool_usage_score`
- `avg_report_quality_score`

## 8. 结果解读建议

普通问答更适合重点看：

- `normal_summary.content_pass_rate`
- `normal_summary.tool_usage_valid_rate`
- `judge_based_summary.avg_correctness_score`
- `judge_based_summary.avg_completeness_score`

报告生成更适合重点看：

- `report_summary.required_tools_present_rate`
- `report_summary.tool_dependency_valid_rate`
- `report_summary.time_consistency_valid_rate`
- `report_summary.avg_groundedness_score`
- `report_summary.avg_report_quality_score`

## 9. 对比两个结果文件

可以用 [compare.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/compare.py) 对比两个结果文件：

```powershell
python src/smart_clean_agent/evaluation/compare.py `
  data/eval/results/before.json `
  data/eval/results/after.json
```

可选写入文件：

```powershell
python src/smart_clean_agent/evaluation/compare.py `
  data/eval/results/before.json `
  data/eval/results/after.json `
  --output data/eval/results/comparison.json
```

输出包含：

- `rule_based_summary_diff`
- `judge_based_summary_diff`
- `normal_summary_diff`
- `report_summary_diff`
- `improved_cases`
- `regressed_cases`

## 10. 旧版 CSV 迁移

如果仍有旧版 `eval_cases.csv`，可以执行：

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

- 迁移脚本是一次性转换入口
- 生成结果是机械迁移版本
- 迁移后建议手动调整 `required_tools / optional_tools / aliases / expected_retrieval_mode`

## 11. 常见问题

### 11.1 提示数据集文件不存在

确认当前位于项目根目录，并且：

- [eval_cases.jsonl](/e:/Python/Agent项目/data/eval/eval_cases.jsonl) 存在

### 11.2 提示 category / route / retrieval mode 非法

说明 JSONL case 不符合 schema，需要检查字段值。

### 11.3 报告类通过率偏低

优先检查：

- `execution_mode`
- `route_correct`
- `missing_required_tools`
- `tool_dependency_valid`
- `time_consistency_valid`
- `missing_required_points`

### 11.4 Judge 失败

优先检查：

- `DASHSCOPE_API_KEY`
- Judge 模型是否可用
- Judge 输出是否为合法 JSON

## 12. 模块定位总结

当前 `evaluation` 不只是关键词命中脚本，而是一套完整的离线评测体系：

- 规则层校验链路行为
- point 层校验内容覆盖
- Judge 层校验语义质量

它更适合用来观察：

- 普通问答 ReAct 的稳定性
- 报告链路的数据依赖和时间一致性
- 不同模型配置下的质量差异
