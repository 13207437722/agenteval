"""四层清洗 + 跨轮证据隔离"""

import copy
import json
import random
from datetime import timedelta

import pytest

from agenteval.dataset import Ctx, _render_event_log, _render_openai, _render_turn_table, n_logistics, history_turn
from agenteval.preprocessing import (RAW_ONLY, CleanConfig, MAX_FIELD_CHARS, build_evidence, isolate_turn_evidence,
                                     normalize_trace, parse_time, preprocess_trace)


class TestLayer1Normalize:
    def test_three_formats_give_same_turns(self):
        """同一段对话渲染成三种异构格式，统一后内容完全一致"""
        rng = random.Random(1)
        c = Ctx(rng, 0)
        turns = [history_turn(c, "order"), n_logistics(c)[0]]
        times = [c.t0 - timedelta(minutes=3), c.t0]  # 事件流按时间排序，轮次间需有真实时间间隔
        raws = [_render_openai(rng, "x", "s", turns, times), _render_event_log(rng, "x", "s", turns, times),
                _render_turn_table(rng, "x", "s", turns, times, False)]
        views = []
        for raw in raws:
            t = normalize_trace(raw)
            views.append([(x.user_query, x.agent_reply, [(tc.tool_name, tc.arguments, tc.result) for tc in x.tool_calls],
                           x.request_time) for x in t.turns])
        assert views[0] == views[1] == views[2]
        assert {normalize_trace(r).source_format for r in raws} == {"openai", "event_log", "turn_table"}

    def test_parse_time_formats(self):
        assert parse_time("2026-09-12T10:00:00+08:00") == "2026-09-12T10:00:00"
        assert parse_time("2026/09/12 10:00:00") == "2026-09-12T10:00:00"
        assert parse_time(None) is None

    def test_retry_dedup_and_error_split(self, openai_raw):
        t = normalize_trace(openai_raw)
        names = [tc.tool_name for tc in t.turns[1].tool_calls]
        assert names == ["log_debug", "query_order"]  # 超时失败的那次被去掉
        assert t.turns[1].tool_calls[1].result["amount"] == 158
        raw_t = normalize_trace(openai_raw, RAW_ONLY)
        assert raw_t.turns[1].tool_calls[1].error.startswith("TIMEOUT")  # 不清洗时保留

    def test_truncate_long_text(self):
        raw = {"turns": [{"turn_id": 0, "user_query": "q", "agent_reply": "r",
                          "tool_calls": [{"tool_name": "get_product_detail", "result": {"desc": "长" * 1000}}]}]}
        t = normalize_trace(raw)
        desc = t.turns[0].tool_calls[0].result["desc"]
        assert len(desc) < 400 and desc.startswith("长" * MAX_FIELD_CHARS) and "已截断" in desc

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            normalize_trace({"foo": 1})


class TestLayer2And3:
    def test_semantic_alignment(self, openai_raw):
        t = preprocess_trace(openai_raw)
        res = t.turns[1].tool_calls[0].result
        assert res["order_status_cn"] == "已发货"
        assert t.turns[1].request_time == "2026-09-12T10:03:00"

    def test_noise_masking(self, openai_raw):
        t = preprocess_trace(openai_raw)
        calls = t.turns[1].tool_calls
        assert [tc.tool_name for tc in calls] == ["query_order"]
        assert "_sys_prompt" not in calls[0].result
        assert "SYS" not in calls[0].result["note"] and calls[0].result["note"] == "正常文本"

    def test_raw_only_keeps_noise(self, openai_raw):
        t = preprocess_trace(openai_raw, RAW_ONLY)
        assert any(tc.tool_name == "log_debug" for tc in t.turns[1].tool_calls)
        assert "_sys_prompt" in json.dumps([tc.result for tc in t.turns[1].tool_calls], ensure_ascii=False)


class TestLayer4Isolation:
    def test_history_has_no_tool_calls(self, openai_raw):
        t = preprocess_trace(openai_raw)
        ev = isolate_turn_evidence(t, 1)
        assert ev["conversation_history"] == [{"user": "订单A多少钱", "assistant": "订单A实付299元。"}]
        assert ev["current_turn"]["tool_calls"][0]["result"]["amount"] == 158
        # 上一轮的工具返回 299 只能出现在历史文本里，不能出现在工具证据里
        assert "299" not in json.dumps(ev["current_turn"]["tool_calls"])

    def test_without_isolation_history_tools_leak(self, openai_raw):
        t = preprocess_trace(openai_raw)
        ev = isolate_turn_evidence(t, 1, isolate=False)
        assert ev["conversation_history"][0]["tool_calls"][0]["result"]["amount"] == 299

    def test_out_of_range(self, openai_raw):
        with pytest.raises(ValueError):
            isolate_turn_evidence(preprocess_trace(openai_raw), 5)

    def test_build_evidence_integrity_marks(self):
        raw = {"traceId": "x", "turnList": [{"turnNo": 3, "qry": "那第二个订单呢？", "resp": "已签收", "reqTime": "2026/09/12 10:00:00",
                                            "toolCalls": [{"toolName": "query_logistics", "toolArgs": "{}", "toolResult": None,
                                                           "errMsg": "LOG_TRUNCATED: tool result not persisted"}]}]}
        ev = build_evidence(preprocess_trace(raw), 0)
        assert any("历史轮次缺失" in m for m in ev["log_integrity"])
        assert any("未落日志" in m for m in ev["log_integrity"])
        assert ev["request_time"] == "2026-09-12T10:00:00"
        ev_raw = build_evidence(preprocess_trace(raw, RAW_ONLY), 0, False, False, False, False)
        assert "request_time" not in ev_raw and "log_integrity" not in ev_raw
