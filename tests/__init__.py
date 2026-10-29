"""
AgentEval 测试套件

- test_preprocessing.py  四层清洗 + 跨轮证据隔离
- test_dataset.py        评测集规模、类别分布、确定性、标签不泄露
- test_judge.py          结构化输出解析 + 标签归一化
- test_evaluator.py      分层短路聚合 + 批量 pipeline 接线 + DeepEval Metric 集成
- test_experiments.py    一致率 / Kappa / 误判率 / Temperature 指标计算
- test_attribution.py    四层归因规则
- test_regression.py     回放状态迁移
- test_live_judge.py     真实 Judge 冒烟测试（未配置 API Key 时自动跳过）

注意：conftest.py 中的 ScriptedLLM 只用于验证代码接线，返回预设字符串，
它产生的任何结果都不是评测结论，也不会写入 results/。
"""
