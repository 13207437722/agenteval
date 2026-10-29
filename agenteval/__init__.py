"""
AgentEval：基于 Agent Trace 的自动化评测系统

模块分工：
- dataset.py        200 条含异常与噪声的多轮 Trace 构建
- preprocessing.py  四层数据清洗 + 跨轮证据隔离
- rubrics.py        CNT 六层漏斗 Rubric 与 Judge Prompt（v2）
- judge.py          LLM 调用（DeepEval 模型封装）+ 结构化输出解析与标签归一化
- evaluator.py      分层短路聚合 + 批量评测 pipeline（清洗 → 推理 → 聚合）
- experiments.py    一致率 / 误判率 / Temperature 梯度实验
- attribution.py    数据 / Prompt / Rubric / Agent 四层错误归因
- regression.py     异常样本回放与回归验证
"""

__version__ = "2.0.0"
