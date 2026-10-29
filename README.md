# AgentEval

基于 Agent Trace 的多工具 Agent 自动化评测与质量诊断系统。它读取电商购物助手的多轮对话和工具调用日志，清洗异构、带噪声的数据，用分层 Rubric + LLM-as-a-Judge 判定每个回复是否合格，再和人工标签对照、做错误归因和回归验证。

方法论参考《淘宝 AI 购物助手在线评测落地实践》（四层清洗、CNT 六层漏斗、幻觉压制、跨轮隔离、四层归因），LLM 评测基础设施复用 [DeepEval](https://github.com/confident-ai/deepeval)。

## 当前状态

| 部分 | 状态 |
|---|---|
| 200 条评测集、5 类异常、噪声注入 | ✅ 已生成（`data/`） |
| 四层清洗 + 跨轮隔离 | ✅ 已实现，离线统计见 `results/cleaning_stats.md` |
| 六层 Rubric、短路聚合、结构化输出三层防线 | ✅ 已实现并有单元测试 |
| 批量 pipeline、一致率/误判率/Temperature 实验、四层归因、回放 | ✅ 代码完成，测试覆盖 |
| **真实 Judge 实验结果（88% / 25% 等）** | ⏳ **未运行**：需要配置 API Key |
| **人工标注** | ⏳ **未完成**：目前只有构造标签，需要用 `annotate` 人工复核 |

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

pytest                            # 55 passed, 1 skipped（真实 Judge 冒烟测试，无 Key 时跳过）
```

### 离线命令（不需要 API Key）

```bash
python -m agenteval.cli build-dataset                         # 重新生成 200 条评测集（固定 seed，结果不变）
python -m agenteval.cli clean-stats                           # 四层清洗统计 → results/cleaning_stats.md
python -m agenteval.cli inspect --trace-id tr_062 --prompt    # 单条 trace：原始日志 → 清洗 → 证据 → Prompt
python -m agenteval.cli inspect --trace-id tr_062 --config baseline_geval --prompt   # 对照基线看到的输入
python -m agenteval.cli annotate                              # 人工标注 → data/human_labels.jsonl
```

### 真实评测（需要 API Key）

复制 `.env.example` 为 `.env`，填入任意 OpenAI 兼容接口：

```bash
JUDGE_API_KEY=sk-xxx
JUDGE_MODEL=gpt-4o              # 或 deepseek-chat / qwen-plus 等
JUDGE_BASE_URL=                 # 非 OpenAI 时填对应的 base_url
```

```bash
python -m agenteval.cli experiment-all          # 一键：Temperature 实验 → 基线/v2/消融 → 一致率 → 对比 → 归因 → 回放

# 也可以分步跑
python -m agenteval.cli run --config v2_improved
python -m agenteval.cli run --config baseline_geval     # 对照基线：原始日志 + DeepEval 通用 GEval
python -m agenteval.cli consistency --run results/runs/v2_improved
python -m agenteval.cli compare --runs baseline_geval v2_improved
python -m agenteval.cli attribution --run results/runs/v2_improved
python -m agenteval.cli replay --old results/runs/baseline_geval --config v2_improved
python -m agenteval.cli temperature --temps 0 0.1 0.3 0.7 1.0 --repeats 3 --n 50
pytest tests/test_live_judge.py -v              # 真实 Judge 冒烟测试（DeepEval assert_test）
```

人工标签 `data/human_labels.jsonl` 标满 200 条后，所有命令自动改用人工标签；否则使用构造标签，并在报告开头注明。

## 系统结构

```
原始异构日志 ──► preprocessing.py ──► evaluator.prepare ──► evaluator.infer ──► check ──► evaluator.aggregate_run
 (3 种格式)      四层清洗+跨轮隔离     证据+Prompt 落盘       并发调 LLM、断点续跑   条数校验   解析/归一化/短路聚合
                                                                                                    │
                    experiments.py（一致率/误判率/Temperature） ◄── results.jsonl ──► attribution.py（四层归因）
                                                                                   └─► regression.py（Bad Case 回放）
```

| 文件 | 职责 | 来源 |
|---|---|---|
| `agenteval/dataset.py` | 200 条多轮 Trace 构建、五类异常、噪声注入、三种异构格式 | 自研 |
| `agenteval/preprocessing.py` | 四层清洗：格式统一 / 语义对齐 / 屏蔽干扰 / 跨轮隔离 | 自研（淘宝方法论） |
| `agenteval/rubrics.py` | CNT 六层 Rubric、v2 Prompt、版本 changelog | 自研（淘宝方法论） |
| `agenteval/judge.py` | LLM 调用（`DeepEvalBaseLLM` 封装）+ 结构化输出三层防线 | DeepEval + 自研 |
| `agenteval/evaluator.py` | 短路聚合、批量 pipeline、DeepEval 自定义 Metric、对照基线（原始日志 + GEval） | 自研 + DeepEval |
| `agenteval/experiments.py` | 一致率、Kappa、误判率对比、Temperature 梯度 | 自研 |
| `agenteval/attribution.py` | 数据 / Prompt / Rubric / Agent 四层归因 | 自研（淘宝方法论） |
| `agenteval/regression.py` | Bad Case + 护栏样本回放，fixed / still_bad / regressed | 自研 |
| `agenteval/cli.py` | 命令行入口 | 自研 |

## 核心设计

1. **跨轮证据隔离**：当前轮保留完整工具调用，历史轮只保留对话文本，避免 Judge 拿上一轮的工具返回（比如上一单的金额）评判当前回复。`preprocessing.isolate_turn_evidence`
2. **四层清洗**：三种日志格式统一，超时重试去重，超长文本截断 → 补提问时间锚点，状态码映射为用户语义 → 剔除噪声调用，正则剔除透传的系统提示词 → 跨轮隔离
3. **CNT 六层漏斗**：CTX → OOSR → OOC → TCC → CNT 串行短路，RED 并联恒评；CTX 证据不足时剔除样本、不进分母。`evaluator.aggregate`
4. **幻觉压制**：事实锚定、先引用后判断、禁止推断（不足则剔除）、Few-shot 边界样本（空态兜底 / 状态语义 / 时间锚点 / 证据不足）。`rubrics.PROMPT_V2_*`
5. **结构化输出三层防线**：Prompt 约束 JSON Schema → Pydantic 校验（失败重试一次）→ 标签归一化。`judge.parse_judge_output`
6. **批量 pipeline 解耦**：prepare / infer / check / aggregate 各自落盘；改聚合规则不用重新推理；Check 节点比对输入输出条数
7. **四层归因**：数据层（残留噪声、未隔离、缺锚点）→ Prompt 层（证据引用不是原文、推测性措辞、该剔除未剔除）→ Rubric 层（批量统计的系统性判严/判松）→ Agent 层
8. **回放回归**：旧版本判错的样本 + 护栏样本用新配置重跑，检查是否修复、有没有引入回归
9. **对照基线**：误判率对比的基线是“原始日志 + DeepEval 通用 GEval”（不做证据治理、不做 Judge 策略优化），规则在实验前固定，同一模型 / temperature / 样本 / 人工标签。结论只说明优于这一通用基线，不代表优于 DeepEval 的 Agent 专用指标或其他评测系统。`evaluator.run_baseline_geval`

## DeepEval 复用边界

- **复用**：`GEval` 作为对照基线的评测器；`BaseMetric` 自定义指标（`CNTFunnelMetric`）+ `LLMTestCase` / `ToolCall` + `assert_test` 做 pytest 集成；`DeepEvalBaseLLM` 模型抽象（同一个模型对象供 v2 和 GEval 共用）
- **说明**：v2 的批量评测主链路（prepare → infer → check → aggregate）不经过 DeepEval
- **自研**：Trace 清洗与证据重建、六层短路聚合、结构化输出归一化、批量 pipeline 与 Check 节点、一致率/误判率/Temperature 实验、四层归因、回放回归

## 文档

- `LEARN.md`：用一条真实样本（tr_062）走完整条数据流，每一步对应函数和数据变化
- `docs/anomaly_types.md`：评测集与五类异常定义
- `docs/experiments.md`：实验设计与指标口径
- `results/`：所有实验产物（由脚本生成）

## License

MIT
