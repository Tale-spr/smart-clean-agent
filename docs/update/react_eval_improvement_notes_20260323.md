# ReAct 改造后评测结果下降的可修复项整理

本文档用于记录 2026-03-23 这一轮显式 ReAct 改造后，离线评测分数下降的主要原因，以及当前可以直接修复或继续完善的方向。

当前对比结果：

- 旧版：
  - `answer_keyword_hit_rate = 0.9667`
  - `tool_call_success_rate = 1.0`
  - `retrieval_hit_rate = 1.0`
  - `report_generation_success_rate = 1.0`
- 新版：
  - `answer_keyword_hit_rate = 0.6333`
  - `tool_call_success_rate = 0.8`
  - `retrieval_hit_rate = 1.0`
  - `report_generation_success_rate = 0.2857`

## 1. 优先修复：评测中的报告类 case 没有强制走报告链路

### 当前现象

在 [src/smart_clean_agent/evaluation/run.py](/e:/Python/Agent项目/src/smart_clean_agent/evaluation/run.py) 中，评测执行时统一构造：

- `report = False`
- `force_report_agent = False`

这会导致：

- 有些报告类 query 因为包含“报告”字样，会被 `ReactAgent._is_report_query()` 识别为报告请求
- 有些报告类 query 没有明显“报告”关键词，只会落到普通问答 ReAct 图中

### 直接后果

这会让部分 `report_generation` case：

- 只调用 `rag_summarize`
- 不走 `fill_context_for_report -> fetch_external_data` 主流程
- 最终显著拉低 `report_generation_success_rate`

### 当前最明显的异常 case

- `report_002`
- `report_003`
- `report_005`

### 建议修复

在 `evaluation/run.py` 中：

- 对 `case.category == "report_generation"` 的 case，显式设置：
  - `force_report_agent = True`
- 不依赖 query 文本是否包含“报告”来判断报告链路

这是当前最应该优先修的项。

---

## 2. 普通问答显式 ReAct 的意图识别规则还不够完整

### 当前现象

[src/smart_clean_agent/agent/react_agent.py](/e:/Python/Agent项目/src/smart_clean_agent/agent/react_agent.py) 目前对普通问答的识别主要依赖：

- `_needs_weather()`
- `_needs_knowledge()`
- `_extract_city()`

这些规则仍然比较启发式，覆盖面有限。

### 直接后果

部分环境适配问题没有被正确识别成“天气/环境类问题”，例如：

- `env_006: 现在我所在城市适不适合高频湿拖？`

当前结果：

- `actual_tools = []`
- `stop_reason = unsupported_request`
- `answer = 我不知道`

### 建议修复方向

补强意图识别规则，尤其是以下隐含表达：

- “适不适合湿拖”
- “会不会影响机器人使用”
- “高频湿拖”
- “这种天气/这种环境下”
- “所在城市”
- “我这边”

建议把这类 query 识别为：

- 若未给城市：先 `get_user_location`
- 再根据问题内容决定是否 `get_weather`
- 必要时再配合 `rag_summarize`

---

## 3. 普通问答最终回答变得更短，和当前关键词评测不再完全匹配

### 当前现象

新版普通问答的回答明显更收敛，不再像旧版那样大段复述知识库内容。

### 直接后果

很多 case 实际回答方向没错，但没命中评测关键词，导致 `answer_keyword_hit_rate` 下降。

### 典型 case

- `faq_002` 缺 `扫地机器人`
- `faq_003` 缺 `出水量`
- `faq_004` 缺 `防缠绕`
- `faq_007` 直接返回 `我不知道`

### 说明

这部分不一定代表新版“能力退化”，更可能是：

- 旧版回答更长，更容易命中关键词
- 新版回答更克制，但当前评测规则仍然是“关键词子串硬匹配”

### 可选优化方向

有两种思路：

1. 优化生成阶段，让最终答案更显式覆盖关键术语  
2. 调整评测样例关键词，让其更贴近“关键概念”而不是“特定措辞”

建议优先做第 1 点，先不要急着改评测规则。

---

## 4. 环境类问题有时会过度调用 `rag_summarize`

### 当前现象

一些环境类 case 在新版中走成了：

- `get_weather`
- `rag_summarize`

例如：

- `env_001`
- `env_007`

### 问题

这不一定是错，但会带来两个副作用：

1. 调用链路更长
2. 某些本来只期望天气工具的 case，看起来“偏离最小工具路径”

### 建议修复方向

在普通问答图中加一层更明确的决策规则：

- 如果天气信息本身已经足够支持回答，则直接生成答案
- 只有当 query 明显还需要“扫地机器人专业知识补充”时，才继续调用 `rag_summarize`

换句话说，要避免：

- “任何天气类问题都顺手再调一次知识库”

---

## 5. 报告链路的工具顺序仍需更严格约束

### 当前现象

虽然这轮只重构了普通问答，但新版评测结果里能看到报告链路有顺序异常：

例如：

- `report_001`
- `report_004`
- `report_007`

出现了：

- `fetch_external_data` 先于 `fill_context_for_report`
- `tool_sequence_valid = false`

### 说明

这说明报告链路虽然还能生成结果，但“固定主流程约束”还不够稳定。

### 建议修复方向

后续应继续强化报告链路：

- `fill_context_for_report` 必须在 `fetch_external_data / fetch_external_history` 之前
- 在代码侧而不是 prompt 侧，真正约束顺序

这不是本轮普通问答 ReAct 的主任务，但它已经是下一轮应该处理的明确问题。

---

## 6. 当前最推荐的修复顺序

建议按下面顺序处理：

1. 修 `evaluation/run.py`
   - 让 `report_generation` case 强制走报告链路
2. 补强普通问答的环境/天气意图识别
   - 优先修 `env_006` 这类 query
3. 优化普通问答最终答案生成
   - 提高关键词覆盖率
4. 收紧环境类问题的二次 `rag_summarize` 决策
5. 后续单独收报告链路的工具顺序约束

---

## 7. 当前结论

本轮评测结果下降，不代表显式 ReAct 方案本身失败。当前掉分的主要原因是：

1. 评测执行路径和真实业务路径不一致  
2. 意图识别规则还不够全  
3. 新版回答风格和旧评测方式不完全匹配  
4. 报告链路原有顺序约束问题仍然存在

因此，下一步不应该直接否定这轮显式 ReAct 改造，而应该优先修正：

- 评测路由
- 环境类意图识别
- 最终答案生成阶段的关键词覆盖能力

# TODO
工作区里像 README.md、prompts/main_prompt.txt、data/memory/... 这些还有既有改动，这轮没有继续动它们。