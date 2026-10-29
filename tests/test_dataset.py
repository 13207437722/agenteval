"""评测集：规模、分布、确定性、标签不泄露、可被清洗"""

import json
from collections import Counter

from agenteval.dataset import build_dataset
from agenteval.preprocessing import load_raw_traces, preprocess_trace


def test_dataset_build(tmp_path):
    stats = build_dataset(str(tmp_path))
    assert stats["total"] == 200
    anomalies = {k: v for k, v in stats["by_type"].items() if k.startswith("A")}
    assert len(anomalies) == 5 and all(v >= 10 for v in anomalies.values())

    raws = load_raw_traces(str(tmp_path / "agent_traces_v1.0.jsonl"))
    labels = [json.loads(l) for l in open(tmp_path / "labels_construction.jsonl", encoding="utf-8")]
    assert len(raws) == len(labels) == 200
    assert all(l["label_source"] == "construction" for l in labels)

    # 原始 trace 中不能出现标签信息
    text = json.dumps(raws, ensure_ascii=False)
    assert "anomaly" not in text and "A1_tool" not in text and "label" not in text

    # 三种格式都有，且全部能被清洗、目标轮在范围内
    fmts = Counter()
    for raw, lb in zip(raws, labels):
        t = preprocess_trace(raw)
        fmts[t.source_format] += 1
        assert t.trace_id == lb["trace_id"]
        assert 0 <= lb["turn_id"] < len(t.turns)
    assert set(fmts) == {"openai", "event_log", "turn_table"}
    assert sum(len(preprocess_trace(r).turns) > 1 for r in raws) > 100  # 多轮为主


def test_dataset_deterministic(tmp_path):
    build_dataset(str(tmp_path / "a"))
    build_dataset(str(tmp_path / "b"))
    for name in ["agent_traces_v1.0.jsonl", "labels_construction.jsonl"]:
        assert (tmp_path / "a" / name).read_text(encoding="utf-8") == (tmp_path / "b" / name).read_text(encoding="utf-8")


def test_repo_dataset_matches_generator(tmp_path):
    """仓库中的 data/ 文件就是生成器的产物（没有手工改动标签）"""
    build_dataset(str(tmp_path))
    for name in ["agent_traces_v1.0.jsonl", "labels_construction.jsonl"]:
        assert (tmp_path / name).read_text(encoding="utf-8") == open(f"data/{name}", encoding="utf-8").read()
