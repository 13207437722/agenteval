"""结构化输出三层防线：JSON 提取 → Schema 校验 → 标签归一化"""

import json

import pytest

from agenteval.judge import extract_json, judge_from_env, normalize_verdict, parse_judge_output


def test_extract_json_from_fenced_text(jj):
    text = "好的，以下是结果：\n```json\n" + jj() + "\n```\n以上。"
    assert extract_json(text)["CTX"]["verdict"] == "1"
    assert extract_json("没有 JSON") is None


@pytest.mark.parametrize("raw,layer,expected", [
    ("1", "TCC", "1"), (1, "TCC", "1"), ("通过", "CNT", "1"), ("PASS", "CNT", "1"),
    ("0", "CNT", "0"), ("不通过", "OOSR", "0"), (0, "RED", "0"),
    ("剔除", "CTX", "N/A"), ("N/A", "OOC", "N/A"), (None, "TCC", "N/A"),
    ("0", "CTX", "N/A"),  # CTX 判 0 视为证据不足
])
def test_normalize(raw, layer, expected):
    assert normalize_verdict(raw, layer)[0] == expected


def test_normalize_rejects_unknown_and_red_na():
    assert normalize_verdict("大概合格", "CNT")[0] is None
    assert normalize_verdict("N/A", "RED")[0] is None


def test_parse_ok_and_normalized(jj):
    layers, status, notes = parse_judge_output(jj())
    assert status == "ok" and layers["TCC"]["verdict"] == "1"
    layers, status, notes = parse_judge_output(jj(CNT="不通过"))
    assert status == "normalized" and layers["CNT"]["verdict"] == "0" and notes


def test_parse_lowercase_keys(jj):
    obj = {k.lower(): v for k, v in json.loads(jj()).items()}
    assert parse_judge_output(json.dumps(obj))[1] == "ok"


def test_parse_failures(jj):
    assert parse_judge_output("not json")[1] == "failed"
    obj = json.loads(jj())
    del obj["RED"]
    assert parse_judge_output(json.dumps(obj))[1] == "failed"  # Schema 缺字段
    assert parse_judge_output(jj(CNT="还行吧"))[1] == "failed"  # 非标值无法归一化


def test_no_key_no_judge(monkeypatch):
    """未配置 API Key 时必须报错，不能静默退化成假 Judge"""
    for k in ["JUDGE_API_KEY", "OPENAI_API_KEY", "JUDGE_MODEL"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("agenteval.judge.load_env", lambda: None)
    with pytest.raises(RuntimeError):
        judge_from_env()
