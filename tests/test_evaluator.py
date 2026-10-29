"""分层短路聚合 + pipeline 接线 + DeepEval 集成"""

import json

import pytest

from agenteval.evaluator import (CNTFunnelMetric, aggregate, aggregate_run, check, evidence_to_test_case, get_config,
                                 infer, prepare)
from agenteval.judge import parse_judge_output
from agenteval.preprocessing import build_evidence, preprocess_trace
from agenteval.rubrics import render_prompt


def agg(text):
    layers, status, notes = parse_judge_output(text)
    return aggregate("t", 0, "test", layers, status, notes, text)


class TestShortCircuit:
    def test_all_pass(self, jj):
        r = agg(jj())
        assert r.final_label == "pass" and r.failed_layer is None and r.short_circuit_at is None

    def test_serial_zero_short_circuits_rest(self, jj):
        r = agg(jj(TCC="0", CNT="0"))
        assert r.final_label == "fail" and r.failed_layer == "TCC"
        assert r.layers["CNT"].verdict == "N/A" and r.layers["CNT"].source == "short_circuit"
        assert r.layers["RED"].verdict == "1" and r.layers["RED"].source == "judge"  # RED 不受短路影响

    def test_ctx_excluded(self, jj):
        r = agg(jj(CTX="剔除"))
        assert r.final_label == "excluded"
        assert all(r.layers[l].verdict == "N/A" for l in ["OOSR", "OOC", "TCC", "CNT"])

    def test_red_parallel(self, jj):
        r = agg(jj(RED="0"))
        assert r.final_label == "fail" and r.failed_layer == "RED"
        r = agg(jj(OOSR="0", RED="0"))
        assert r.failed_layer == "OOSR" and r.layers["RED"].verdict == "0"

    def test_untriggered_na_does_not_short_circuit(self, jj):
        r = agg(jj(OOSR="N/A"))
        assert r.final_label == "pass" and r.layers["TCC"].source == "judge"

    def test_parse_failed(self):
        r = agg("garbage")
        assert r.final_label is None and r.parse_status == "failed"


class TestPrompt:
    def test_versions(self, openai_raw):
        ev = build_evidence(preprocess_trace(openai_raw), 1)
        v2, nh = (render_prompt(ev, v) for v in ["v2", "v2_no_hallu"])
        assert "取证总原则" in v2 and "边界样本示例" in v2
        assert "取证总原则" not in nh and "边界样本示例" not in nh and "六层 Rubric" in nh
        assert '"trace_id": "t1"' in v2
        for bad in ["v1", "v9"]:
            with pytest.raises(ValueError):
                render_prompt(ev, bad)


class TestPipeline:
    def _write(self, tmp_path, openai_raw):
        p = tmp_path / "traces.jsonl"
        rows = []
        for i in range(5):
            r = json.loads(json.dumps(openai_raw))
            r["id"] = f"t{i}"
            rows.append(json.dumps(r, ensure_ascii=False))
        p.write_text("\n".join(rows), encoding="utf-8")
        return str(p)

    def test_prepare_infer_check_aggregate(self, tmp_path, openai_raw, scripted, jj):
        traces = self._write(tmp_path, openai_raw)
        run = str(tmp_path / "run")
        assert prepare(traces, get_config("v2_improved"), run) == 5
        llm = scripted(jj(), {"t1": jj(CNT="0"), "t2": "garbage"})
        stats = infer(run, llm, workers=2)
        assert stats["called"] == 5
        assert llm.calls == 6  # t2 解析失败触发一次重试
        assert check(run)["ok"]
        res = {r.trace_id: r for r in aggregate_run(run)}
        assert res["t0"].final_label == "pass" and res["t1"].failed_layer == "CNT"
        assert res["t2"].final_label is None
        # 断点续跑：已完成的样本不再调用
        llm2 = scripted(jj())
        assert infer(run, llm2)["called"] == 0 and llm2.calls == 0

    def test_check_node_blocks_aggregate(self, tmp_path, openai_raw, scripted, jj):
        traces = self._write(tmp_path, openai_raw)
        run = tmp_path / "run"
        prepare(traces, get_config("v2_improved"), str(run))
        infer(str(run), scripted(jj()))
        lines = (run / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        (run / "raw_outputs.jsonl").write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")  # 模拟推理丢数据
        assert not check(str(run))["ok"]
        with pytest.raises(RuntimeError):
            aggregate_run(str(run))

class FixedScore:
    """测试替身：GEval 的步骤生成与打分都返回固定 JSON；calls 记录调用次数"""

    def __init__(self, score=3, fail_ids=()):
        from deepeval.models import DeepEvalBaseLLM

        outer = self

        class _M(DeepEvalBaseLLM):
            def __init__(self):
                super().__init__("fixed")

            def load_model(self):
                return None

            def generate(self, prompt, schema=None, **kw):
                outer.calls += 1
                if any(f'"id": "{i}"' in prompt for i in fail_ids):
                    raise RuntimeError("api down")
                return json.dumps({"steps": ["检查回复"], "score": score, "reason": "x"})

            async def a_generate(self, prompt, schema=None, **kw):
                return self.generate(prompt)

            def get_model_name(self):
                return "fixed"

        self.calls = 0
        self.llm = _M()


class TestBaselineGEval:
    @staticmethod
    def _write_n(tmp_path, openai_raw, n):
        p = tmp_path / "traces.jsonl"
        rows = []
        for i in range(n):
            r = json.loads(json.dumps(openai_raw))
            r["id"] = f"t{i}"
            rows.append(json.dumps(r, ensure_ascii=False))
        p.write_text("\n".join(rows), encoding="utf-8")
        return str(p)

    def test_test_case_is_raw_log(self, openai_raw):
        from agenteval.evaluator import baseline_test_case
        tc = baseline_test_case(openai_raw)
        assert tc.input == "订单B多少钱" and tc.actual_output.startswith("订单B")
        ctx = json.loads(tc.context[0])
        assert ctx == openai_raw  # 原始日志原样传入：噪声、历史工具结果都还在
        assert "_sys_prompt" in tc.context[0]

    def test_run_resume_and_check(self, tmp_path, openai_raw):
        from agenteval.evaluator import BASELINE, run_pipeline
        traces = self._write_n(tmp_path, openai_raw, 3)
        bad = FixedScore(fail_ids=("t1",))
        with pytest.raises(RuntimeError):  # t1 调用失败 → Check 节点拦截
            run_pipeline(BASELINE, traces, None, str(tmp_path), bad.llm, workers=1)
        steps = json.loads((tmp_path / BASELINE / "geval_steps.json").read_text(encoding="utf-8"))["steps"]
        assert steps == ["检查回复"]

        good = FixedScore(score=8)
        res = run_pipeline(BASELINE, traces, None, str(tmp_path), good.llm, workers=1)
        assert good.calls == 1  # 只补跑失败的 t1；步骤已冻结，不再生成
        by_id = {r.trace_id: r for r in res}
        assert by_id["t0"].final_label == "fail" and by_id["t0"].score == pytest.approx(0.3)  # 3/10 < 0.5
        assert by_id["t1"].final_label == "pass" and by_id["t1"].layers == {}


class TestDeepEvalIntegration:
    def test_metric(self, openai_raw, scripted, jj):
        ev = build_evidence(preprocess_trace(openai_raw), 1)
        tc = evidence_to_test_case(ev)
        assert tc.tools_called[0].name == "query_order" and tc.actual_output.startswith("订单B")
        m = CNTFunnelMetric(scripted(jj(TCC="0")))
        assert m.measure(tc) == 0.0 and not m.is_successful() and "TCC" in m.reason
        m = CNTFunnelMetric(scripted(jj(CTX="剔除")))
        m.measure(tc)
        assert m.skipped and m.case_result.final_label == "excluded"
