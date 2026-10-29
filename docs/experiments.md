# 实验设计

> 状态：**实验尚未运行**（截至 2026-10-26 没有配置 Judge API Key）。本文档只描述实验设计；
> 结果全部由脚本写入 `results/`，不要在这里手填数字。

## 一键复现

```bash
# .env 中配置 JUDGE_API_KEY / JUDGE_MODEL（可选 JUDGE_BASE_URL）
python -m agenteval.cli experiment-all
```

执行顺序：Temperature 梯度实验 → 用选出的 temperature 跑基线 / v2 / 4 个消融配置 → 一致率报告 → 对比报告 → v2 四层归因 → 基线 Bad Case 用 v2 回放。

## 1. 评测单元与标签

- 样本：200 条 Trace 的目标轮，每条一个最终标签 `pass / fail / excluded`
- 标签：`data/labels_construction.jsonl` 是构造标签；简历写的是"人工标注一致率"，因此**正式口径必须使用人工标注**
  （`python -m agenteval.cli annotate` 生成 `data/human_labels.jsonl`，标满 200 条后所有命令自动改用人工标签）
- 所有报告的开头都会注明标签来源

## 2. Judge-Human 一致率（简历：88%）

以下口径在跑实验之前规定，引用数字时必须注明口径：

| 指标 | 定义 |
|---|---|
| 全样本一致率 | 200 条，pass / fail / excluded 三类，Judge 与人工相同才算一致 |
| 可判样本一致率 | 只看人工标为 pass / fail 的样本（约 185 条）；Judge 在这里输出 excluded（误拒评）计为不一致 |
| N/A 识别 | 人工 excluded 中 Judge 也剔除的比例；可判样本上的误拒评数；不支持拒评的方法记为"不支持" |
| ERROR | 调用或解析失败的样本数；在两种一致率中都计为不一致，不从分母剔除 |

- 同时报告：Cohen's Kappa、混淆矩阵、分类型判对率、失败层定位准确率
- 简历中的 88% 必须说明是全样本还是可判样本口径
- 产物：`results/consistency_report.md`

## 3. 误判率相对降低（简历：25%）

- 误判率 = 1 - 一致率
- 相对降低 = (基线误判率 - v2 误判率) / 基线误判率，全样本和可判样本两个口径各算一次
- **baseline_geval（对照基线）**：不做证据清洗、不做 Judge 策略优化的通用评测方案
  - 最小字段适配：目标轮用户问题 → `input`，Agent 回复 → `actual_output`；整条原始日志原样放进 `context`。不去噪、不隔离历史工具结果、不补时间锚点、不重构证据
  - 评测器：DeepEval `GEval`。`criteria` 覆盖与六层 Rubric 相同的评测要求，并附同一张能力卡；不加事实锚定 / 证据引用 / Few-shot / 禁止推断
  - `evaluation_steps` 由 GEval 根据 criteria 生成一次，冻结后用于全部样本（`results/runs/baseline_geval/geval_steps.json`）
  - 判定规则：GEval 原生，0~1 分，≥ 0.5 判 pass；没有"剔除"输出
- **v2_improved**：四层清洗 + 跨轮隔离 + 时间锚点 + 状态语义表 + 六层 Rubric（取证总原则、CTX 剔除、模式 B、Few-shot）+ 代码强制短路
- 两者使用同一个 Judge 模型、同一个 temperature、同一批样本、同一份人工标签；以上规则在实验前确定，不按结果回调
- 公平性说明：
  - 基线没有剔除能力，全样本口径下 15 条证据不足样本必然记为基线误判，所以判断准确性以**可判样本口径**为主，N/A 识别单独报告
  - 两者判定规则不同（连续分 + 阈值 vs 六层短路），只比较与人工标签的一致性，不比较通过率；报告附基线阈值敏感性（0.3~0.7）检查结论是否依赖 0.5，不据此回调
- 结论边界：只说明优于"原始日志 + DeepEval G-Eval 通用整体质量评测"这一基线，不代表优于 DeepEval 的 Agent 专用指标或其他评测系统
- 产物：`results/comparison_report.md`

## 4. 消融实验（定位每项改进的贡献）

| 配置 | 相对 v2 关掉的东西 |
|---|---|
| abl_no_isolation | 跨轮证据隔离（历史轮工具返回进入 Judge 输入） |
| abl_no_noise_mask | 第三层噪声屏蔽（保留噪声调用与透传提示词） |
| abl_no_semantic | 第二层语义对齐 + 时间锚点 + 状态语义表 |
| abl_no_hallu | Prompt 取证总原则 + Few-shot（幻觉压制） |

## 5. Temperature 梯度实验

- 温度：0.0 / 0.1 / 0.3 / 0.7 / 1.0；每个温度重复 3 次；按类型分层抽样 50 条
- 稳定性 = 3 次运行最终标签完全相同的样本占比；同时记录平均一致率与解析失败次数
- 选择规则：稳定性优先，其次一致率（淘宝文章最终选 0.1，本项目以实测为准）
- 产物：`results/temperature_report.md`

## 6. 四层归因 + 回放

- `results/attribution_v2_improved.md`：v2 不一致样本的根因分布（data / prompt / rubric / unresolved），以及确认的 Agent 真实缺陷（基线只有一个总分，没有分层证据可归因）
- `results/replay_baseline_geval_to_v2_improved.md`：基线的 Bad Case + 30 条护栏样本用 v2 重跑，统计 fixed / still_bad / regressed

## 7. 结果与简历不一致时

如实写真实值。可分析的方向：人工标注口径与 Rubric 不一致、Few-shot 不足、Judge 模型能力、样本量（200 条下 1 个样本 = 0.5 个百分点）。
