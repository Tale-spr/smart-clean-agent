# 新版 Evaluation 模块升级计划

## 1. 背景与目标

基于当前最新评测结果可以看出，代码层面的普通问答 ReAct、评测路由和部分报告链路问题已经得到一定修复，但 `evaluation` 模块本身仍然存在明显局限：

- 当前 `answer_keyword_hit` 主要依赖固定关键词子串匹配，容易误伤“语义正确但表述不同”的回答。
- 报告类 case 目前仍以单个 `expected_tool` 为核心，无法完整表达多工具链路的合理性。
- 部分环境类问题在真实产品上允许“天气 + 知识库”组合回答，但评测集中被默认视为偏离预期。
- 现有 summary 更擅长衡量“是否命中规则”，不擅长衡量“答案质量是否足够好”。

下一轮 `evaluation` 升级的目标，不是替换掉当前规则评测，而是把它扩展成更稳定、更接近真实质量判断的三层评测体系：

1. **规则层**：继续校验路由、工具调用、顺序、步数、停止条件。
2. **内容层**：从“关键词命中”升级为“核心要点覆盖”。
3. **Judge 层**：引入独立模型作为离线裁判，对答案质量做语义评分。

---

## 2. 升级原则

### 2.1 保留规则评测，不直接废弃

当前的规则评测仍然有价值，尤其适合验证：

- 报告 case 是否真的走了报告链路
- 普通问答是否按预期使用天气工具或知识库
- ReAct 是否在步数上限内收敛
- 报告工具顺序是否合理

因此新版 `evaluation` 不应删除以下字段，而应继续保留并增强：

- `execution_mode`
- `route_expected`
- `route_correct`
- `actual_tools`
- `tool_call_success`
- `step_count`
- `stop_reason`
- `tool_sequence_valid`

### 2.2 内容评测要从“词命中”升级为“要点覆盖”

当前 `expected_keywords` 的问题在于：

- 过于脆弱，容易被同义表达绕开
- 对长答案偏友好，对精炼答案偏苛刻
- 无法区分“核心点缺失”和“某个术语没写出来”

因此建议引入：

- `required_points`
- `optional_points`

并且把 case 评测从布尔值改成更细粒度覆盖率，例如：

- `required_point_hit_rate`
- `optional_point_hit_rate`
- `content_pass`

### 2.3 Judge 模型用于“质量裁判”，不是替代规则层

Judge 模型的职责是补足语义判断，不替代规则层。

Judge 负责回答这些问题：

- 回答是否正确
- 回答是否完整
- 回答是否贴合用户问题
- 回答是否利用了已知证据
- 报告是否具备结构性和建议性

规则层和 Judge 层应同时存在，避免出现：

- 只看模型主观评分，不知道链路是否跑偏
- 只看硬规则，不知道回答是否真的对用户有用

---

## 3. 推荐的新版评测架构

### 3.1 第一层：Rule-Based Evaluation

这一层保留现有 `evaluation/service.py` 的核心框架，并重点升级以下内容：

#### 3.1.1 case schema 升级

当前 CSV 字段建议扩展为：

- `case_id`
- `query`
- `category`
- `user_id`
- `city`
- `expected_route`
- `required_tools`
- `optional_tools`
- `required_points`
- `optional_points`
- `expected_retrieval_mode`
- `notes`

其中：

- `expected_route` 取值建议为：
  - `normal`
  - `report`
- `required_tools` 用 `|` 分隔，表示必需出现的工具
- `optional_tools` 表示允许出现但不强制
- `required_points` 表示答案必须覆盖的关键点
- `optional_points` 表示更优回答可覆盖的附加点

#### 3.1.2 规则层指标升级

普通问答与报告类统一输出以下规则指标：

- `route_correct`
- `required_tools_present`
- `unexpected_tools_present`
- `tool_sequence_valid`
- `step_limit_respected`
- `stop_reason_valid`
- `retrieval_mode_valid`
- `required_point_hit_rate`
- `optional_point_hit_rate`

#### 3.1.3 报告链路规则增强

报告类 case 不再只看一个 `expected_tool`，而是看：

- 是否包含 `fill_context_for_report`
- 是否包含 `fetch_external_data`
- 如 case 需要趋势分析，是否包含 `fetch_external_history`
- 工具顺序是否满足：
  - `get_user_id`
  - `get_current_month`
  - `fill_context_for_report`
  - `fetch_external_data`
  - `fetch_external_history`

同时顺序校验应升级为：

- 不仅检查“如果都存在，顺序是否正确”
- 还要检查“必需工具是否缺失”

---

### 3.2 第二层：Point-Based Content Evaluation

这一层用于替代纯关键词布尔命中。

#### 3.2.1 required / optional point 设计建议

例如：

`faq_004: 带宠物的家庭选购机器人应该关注哪些能力？`

- required_points:
  - 宠物家庭场景
  - 毛发/缠绕问题
  - 清洁能力或吸力
- optional_points:
  - 防缠绕
  - 胶刷
  - 大尘盒
  - 激光导航

`report_004: 给我出一份2025-06的使用报告，顺便说说要不要更换耗材。`

- required_points:
  - 6月使用情况总结
  - 耗材状态
  - 是否需要更换耗材
  - 具体建议
- optional_points:
  - 趋势分析
  - 维护周期
  - 风险提醒

#### 3.2.2 判定方式

第一版仍可先用规则方式做点覆盖：

- 用多个同义短语映射到一个 point
- point 命中则记为 1
- 汇总为覆盖率

例如一个 point 可以有多个匹配词：

- `防缠绕`: `["防缠绕", "胶刷", "减少缠绕", "抗缠绕"]`
- `耗材更换`: `["更换耗材", "建议更换", "需要更换", "更换周期"]`

这样比单个关键词硬匹配更稳。

---

### 3.3 第三层：LLM-as-a-Judge

这一层建议新增独立 judge 模块，而不是把现有 agent 改造成 self-reflection。

#### 3.3.1 为什么不是直接做 Reflexion

Reflexion 更适合：

- 先生成答案
- 再批评答案
- 再重新生成答案

而当前 `evaluation` 模块需要的是：

- 对现有答案进行独立评分

所以更准确的方案应是：

- `answer agent` 负责作答
- `judge agent` 负责离线评估

#### 3.3.2 judge 输入建议

Judge 输入建议包含：

- `query`
- `category`
- `answer`
- `required_points`
- `optional_points`
- `actual_tools`
- `execution_mode`
- `retrieved_docs_summary`（可选）

#### 3.3.3 judge 输出建议

Judge 输出建议固定为结构化 JSON：

- `correctness_score`：0-5
- `completeness_score`：0-5
- `groundedness_score`：0-5
- `tool_usage_score`：0-5
- `report_quality_score`：0-5，仅报告类
- `passed`：bool
- `reason`：简短中文解释

#### 3.3.4 judge 使用方式

Judge 建议只用于：

- 离线评测
- 对比不同版本
- 不进入线上主流程

避免增加线上时延与成本。

---

## 4. 建议的数据与结果结构调整

### 4.1 新版评测数据结构

建议把当前 `eval_cases.csv` 升级成支持以下字段：

- `required_tools`
- `optional_tools`
- `required_points`
- `optional_points`
- `expected_route`
- `expected_retrieval_mode`

如果 CSV 过于难维护，可进一步考虑：

- 迁移为 `JSONL`
- 每条 case 更容易表达数组结构

### 4.2 新版结果结构

建议评测输出分成两层：

#### `rule_based`
- `route_correct`
- `tool_sequence_valid`
- `required_tools_present`
- `required_point_hit_rate`
- `optional_point_hit_rate`
- `step_count`
- `stop_reason`

#### `judge_based`
- `correctness_score`
- `completeness_score`
- `groundedness_score`
- `tool_usage_score`
- `report_quality_score`
- `passed`
- `reason`

### 4.3 summary 也应分层

最终 summary 建议同时输出：

- `rule_based_summary`
- `judge_based_summary`

例如：

- `route_correct_rate`
- `required_tool_pass_rate`
- `required_point_hit_rate_avg`
- `judge_pass_rate`
- `avg_correctness_score`
- `avg_completeness_score`

---

## 5. 推荐的实施顺序

### 第一阶段：升级规则层

优先做：

- case schema 扩展
- 报告类 `required_tools` / 顺序校验增强
- 环境类 `expected_route` / `expected_retrieval_mode` 明确化

目标：

- 让评测先做到“链路判断公平”

### 第二阶段：升级内容层

新增：

- `required_points`
- `optional_points`
- 点覆盖评测逻辑

目标：

- 降低“答案本质正确，但没命中固定词”的误伤

### 第三阶段：新增 Judge 层

新增：

- `evaluation/judge.py`
- judge prompt
- judge result schema
- judge 汇总逻辑

目标：

- 让评测从“规则正确”走向“质量可解释”

### 第四阶段：加入版本对比报告

最后再补：

- 两个评测结果文件的对比脚本
- 输出哪些 case 变好、哪些 case 变差
- 输出 rule-based 和 judge-based 两套差异

目标：

- 支持更规范的版本回归分析

---

## 6. 建议优先修复的当前评测问题

从当前最新结果出发，优先级建议如下：

### 高优先级

1. 报告类 case 从单个 `expected_tool` 升级为 `required_tools`
2. 报告顺序校验增加“必需步骤缺失”判断
3. `expected_keywords` 升级为 `required_points` / `optional_points`

### 中优先级

4. 环境类 case 增加 `expected_retrieval_mode`
5. 区分“纯天气问题”和“天气 + 设备建议问题”

### 后续增强

6. 新增 judge 模块
7. 新增评测对比脚本

---

## 7. 最终目标

新版 `evaluation` 模块的最终目标不是追求一个更高的单一分数，而是形成一套更可信的评测体系：

- 能判断 Agent 是否按设计运行
- 能判断答案是否覆盖核心要点
- 能判断答案是否真正对用户有价值
- 能支持新旧版本客观对比

当这三层能力都具备后，你的项目就不再只是“有个 eval 脚本”，而是真正具备了**Agent 系统迭代评估能力**。
