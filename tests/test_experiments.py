"""指标计算：全部基于输入结果现场计算"""

import pytest

from agenteval.experiments import (cohen_kappa, comparison_report, consistency_metrics, relative_reduction,
                                   stratified_sample, temperature_metrics)
from agenteval.models import CaseResult, GroundTruth


def gt(tid, label, layer=None, typ=None):
    return GroundTruth(trace_id=tid, turn_id=0, label=label, failed_layer=layer, anomaly_type=typ)


def res(tid, label, layer=None):
    return CaseResult(trace_id=tid, turn_id=0, config="x", final_label=label, failed_layer=layer)


LABELS = {(g.trace_id, 0): g for g in [gt("a", "pass"), gt("b", "fail", "TCC", "A1"), gt("c", "fail", "CNT", "A2"),
                                       gt("d", "excluded", "CTX", "CTX"), gt("e", "pass")]}


def test_consistency():
    results = [res("a", "pass"), res("b", "fail", "TCC"), res("c", "fail", "TCC"), res("d", "pass"), res("e", None)]
    m = consistency_metrics(results, LABELS)
    assert m["n"] == 5 and m["agree"] == 3  # a, b, c 一致；d 判错；e 解析失败计为不一致
    assert m["consistency"] == pytest.approx(0.6) and m["misjudgment"] == pytest.approx(0.4)
    assert m["parse_failed"] == 1
    assert m["layer_localization"] == {"both_fail": 2, "same_layer": 1, "acc": 0.5}
    m2 = consistency_metrics(results, LABELS, include_excluded=False)
    assert m2["n"] == 4 and m2["agree"] == 3


def test_kappa():
    assert cohen_kappa([("p", "p"), ("f", "f")]) == pytest.approx(1.0)
    # po=0.5, pe=0.5 → 0
    assert cohen_kappa([("p", "p"), ("p", "f"), ("f", "p"), ("f", "f")]) == pytest.approx(0.0)


def test_relative_reduction_and_report():
    assert relative_reduction(0.4, 0.3) == pytest.approx(0.25)
    assert relative_reduction(0, 0.1) is None
    runs = {"base": [res("a", "fail"), res("b", "fail"), res("c", "pass"), res("d", "pass"), res("e", "pass")],
            "new": [res("a", "pass"), res("b", "fail"), res("c", "fail"), res("d", "excluded"), res("e", "pass")]}
    text = comparison_report(runs, LABELS, "base", "new")
    assert "误判率相对降低" in text and "construction" in text  # 未人工标注时报告必须注明


def test_temperature_stability():
    labels = {(k, 0): gt(k, "pass") for k in "abcd"}
    runs = {0.1: [[res(k, "pass") for k in "abcd"]] * 3,
            1.0: [[res(k, "pass") for k in "abcd"], [res("a", "fail")] + [res(k, "pass") for k in "bcd"],
                  [res(k, "pass") for k in "abcd"]]}
    rows = {r["temperature"]: r for r in temperature_metrics(runs, labels)}
    assert rows[0.1]["stability"] == 1.0 and rows[1.0]["stability"] == 0.75


def test_stratified_sample_covers_types():
    labels = {(f"n{i}", 0): gt(f"n{i}", "pass") for i in range(100)}
    labels.update({(f"x{i}", 0): gt(f"x{i}", "fail", "TCC", "A1") for i in range(20)})
    ids = stratified_sample(labels, 12)
    assert any(i.startswith("x") for i in ids) and len(ids) <= 12


def test_na_metrics_and_baseline_report():
    from agenteval.experiments import na_metrics
    from agenteval.models import LayerResult
    # v2：d 正确剔除，a 误拒评，e 解析失败
    layer = {"CTX": LayerResult(layer="CTX", verdict="1")}
    v2 = [res("a", "excluded"), res("b", "fail", "TCC"), res("c", "fail", "CNT"), res("d", "excluded"), res("e", None)]
    for r in v2:
        r.layers = layer
    na = na_metrics(v2, LABELS)
    assert na == {"na_total": 1, "na_hit": 1, "na_recall": 1.0, "false_na": 1, "judgeable": 4, "errors": 1}
    # 基线：只有连续分，没有分层，也就不支持剔除
    base = [CaseResult(trace_id=k, turn_id=0, config="b", score=s, final_label="pass" if s >= 0.5 else "fail")
            for k, s in zip("abcde", [0.9, 0.2, 0.6, 0.7, 0.8])]
    text = comparison_report({"base": base, "v2": v2}, LABELS, "base", "v2")
    assert "不支持（0/1）" in text and "阈值敏感性" in text and "结论边界" in text
    assert "全样本口径" in text and "可判样本口径" in text
