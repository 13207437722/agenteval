"""
真实 Judge 冒烟测试（DeepEval pytest 集成）

配置 .env 中的 JUDGE_API_KEY / JUDGE_MODEL 后运行：pytest tests/test_live_judge.py -v
未配置时自动跳过。这里只验证"真实模型能按 Schema 输出、短路聚合能跑通"，
正式指标以 results/ 下的批量实验为准。
"""

import os

import pytest

from agenteval.judge import load_env

load_env()
pytestmark = pytest.mark.skipif(not (os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")) or not os.getenv("JUDGE_MODEL"),
                                reason="未配置真实 Judge（JUDGE_API_KEY / JUDGE_MODEL）")


def test_live_judge_smoke(openai_raw):
    from deepeval import assert_test

    from agenteval.evaluator import CNTFunnelMetric, evidence_to_test_case
    from agenteval.judge import judge_from_env
    from agenteval.preprocessing import build_evidence, preprocess_trace

    metric = CNTFunnelMetric(judge_from_env())
    tc = evidence_to_test_case(build_evidence(preprocess_trace(openai_raw), 1))
    assert_test(tc, [metric])  # 订单 B 回复与当前轮工具返回一致，应判通过
    assert metric.case_result.parse_status in ("ok", "normalized")
