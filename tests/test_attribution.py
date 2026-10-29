"""四层归因规则"""

import json

from agenteval.attribution import (_quote_grounded, check_data_layer, check_prompt_layer, check_rubric_layer,
                                   rubric_clusters)
from agenteval.evaluator import aggregate
from agenteval.judge import parse_judge_output
from agenteval.models import CaseResult, GroundTruth
from agenteval.preprocessing import RAW_ONLY, build_evidence, preprocess_trace


def agg(text):
    layers, status, notes = parse_judge_output(text)
    return aggregate("t1", 1, "x", layers, status, notes, text)


def test_data_layer_flags_raw_evidence(openai_raw, jj):
    raw_ev = build_evidence(preprocess_trace(openai_raw, RAW_ONLY), 1, False, False, False, False)
    issues = check_data_layer(raw_ev, agg(jj(CNT="0")))
    text = " ".join(issues)
    assert "噪声工具" in text and "提示词" in text and "跨轮隔离" in text and "语义映射" in text

    clean_ev = build_evidence(preprocess_trace(openai_raw), 1)
    assert check_data_layer(clean_ev, agg(jj(CNT="0"))) == []


def test_prompt_layer(openai_raw):
    ev = build_evidence(preprocess_trace(openai_raw), 1)
    out = {l: {"evidence_quote": "", "verdict": "1", "reason": ""} for l in ["CTX", "OOSR", "OOC", "TCC", "CNT", "RED"]}
    out["CNT"] = {"evidence_quote": "订单B实付999元并且已经签收完成", "verdict": "0", "reason": "金额可能不对"}
    r = agg(json.dumps(out, ensure_ascii=False))
    issues = check_prompt_layer(ev, "", r, GroundTruth(trace_id="t1", turn_id=1, label="pass"))
    assert any("不是输入原文" in s for s in issues) and any("推测" in s for s in issues)


def test_quote_grounding():
    assert _quote_grounded("订单B实付158元", '{"agent_reply": "订单B实付158元，已发货。"}')
    assert not _quote_grounded("用户要求退还全部货款并赔偿", '{"agent_reply": "订单B实付158元"}')


def test_rubric_cluster():
    def r(tid):
        return CaseResult(trace_id=tid, turn_id=0, config="x", final_label="fail", failed_layer="CNT")
    g = GroundTruth(trace_id="x", turn_id=0, label="pass")
    items = [(r(f"t{i}"), g, ["empty_state"]) for i in range(3)]
    clusters = rubric_clusters(items)
    assert clusters[("CNT", "strict", "empty_state")] == 3
    assert "系统性判严" in check_rubric_layer(items[0][0], g, ["empty_state"], clusters)[0]
    assert check_rubric_layer(items[0][0], g, ["other"], clusters) == []
