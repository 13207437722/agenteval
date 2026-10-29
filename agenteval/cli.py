"""
命令行入口：python -m agenteval.cli <命令>

离线命令（不需要 API Key）：
  build-dataset   生成 200 条评测集 + 构造标签
  inspect         查看单条 trace 从原始日志 → 清洗 → 证据 → Prompt 的全过程
  clean-stats     统计四层清洗在全量数据上的处理情况
  annotate        人工逐条标注（生成 data/human_labels.jsonl）

需要真实 Judge（.env 中配置 JUDGE_API_KEY / JUDGE_MODEL）：
  run             按配置批量评测（--config v2_improved / baseline_geval / 消融配置）
  temperature     Temperature 梯度实验
  experiment-all  一键跑完 Temperature/基线/v2/消融/一致率/对比/归因/回放 全部实验

基于已有运行结果的分析（不调用 LLM）：
  consistency / compare / attribution / replay-report
"""

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

DATA = "data/agent_traces_v1.0.jsonl"
CONSTRUCTION = "data/labels_construction.jsonl"
HUMAN = "data/human_labels.jsonl"
RUNS = "results/runs"


def _default_labels() -> str:
    """人工标签覆盖全部样本时优先使用人工标签，否则用构造标签（报告中会注明来源）"""
    if Path(HUMAN).exists():
        with open(HUMAN, encoding="utf-8") as f:
            n = sum(1 for l in f if l.strip())
        with open(CONSTRUCTION, encoding="utf-8") as f:
            total = sum(1 for l in f if l.strip())
        if n >= total:
            return HUMAN
        print(f"[提示] 人工标签只有 {n}/{total} 条，暂用构造标签 {CONSTRUCTION}")
    return CONSTRUCTION


def _tags() -> dict:
    with open(CONSTRUCTION, encoding="utf-8") as f:
        return {r["trace_id"]: r.get("tags", []) for r in map(json.loads, filter(str.strip, f))}


# ============== 离线命令 ==============

def cmd_build_dataset(a):
    from agenteval.dataset import build_dataset
    stats = build_dataset(a.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_inspect(a):
    from agenteval.evaluator import BASELINE, get_config, load_targets
    from agenteval.preprocessing import build_evidence, load_raw_traces, preprocess_trace
    from agenteval.rubrics import render_prompt
    raw = next((r for r in load_raw_traces(a.input)
                if a.trace_id in (r.get("id"), r.get("session_id"), r.get("traceId"))), None)
    if raw is None:
        sys.exit(f"找不到 {a.trace_id}")
    print("=" * 30, "原始日志", "=" * 30)
    print(json.dumps(raw, ensure_ascii=False, indent=1)[:4000])
    if a.config == BASELINE:
        from agenteval.evaluator import GEVAL_CRITERIA, baseline_test_case
        tc = baseline_test_case(raw, load_targets(CONSTRUCTION).get(a.trace_id))
        print("=" * 30, "基线 GEval 测试用例（context = 上面的原始日志，原样传入）", "=" * 30)
        print(f"input: {tc.input}\nactual_output: {tc.actual_output}\ncontext: {len(tc.context[0])} 字符")
        if a.prompt:
            print("=" * 30, "GEval criteria", "=" * 30)
            print(GEVAL_CRITERIA)
        return
    cfg = get_config(a.config)
    trace = preprocess_trace(copy.deepcopy(raw), cfg.clean)
    print("=" * 30, "清洗记录", "=" * 30)
    print("\n".join(trace.cleaning_log))
    turn = load_targets(CONSTRUCTION).get(trace.trace_id, len(trace.turns) - 1)
    ev = build_evidence(trace, turn, cfg.isolate, cfg.time_anchor, cfg.glossary, cfg.integrity)
    print("=" * 30, f"Judge 证据（{a.config}，目标轮 {turn}）", "=" * 30)
    print(json.dumps(ev, ensure_ascii=False, indent=1))
    if a.prompt:
        print("=" * 30, "Prompt", "=" * 30)
        print(render_prompt(ev, cfg.prompt))


def cmd_clean_stats(a):
    from agenteval.preprocessing import CleanConfig, build_evidence, load_raw_traces, preprocess_trace
    from agenteval.rubrics import render_prompt
    raws = load_raw_traces(a.input)
    fmt, events = Counter(), Counter()
    raw_chars, clean_chars, hist_calls = [], [], 0
    for raw in raws:
        t = preprocess_trace(copy.deepcopy(raw), CleanConfig())
        fmt[t.source_format] += 1
        for m in t.cleaning_log[1:]:
            events[m.split("：")[0].split(" ")[0]] += 1
        last = len(t.turns) - 1
        hist_calls += sum(len(x.tool_calls) for x in t.turns[:last])
        raw_chars.append(len(json.dumps(raw, ensure_ascii=False)))
        clean_chars.append(len(render_prompt(build_evidence(t, last), "v2")))
    lines = ["# 四层清洗统计（离线，全量数据）\n", f"样本数：{len(raws)}\n", "## 日志格式分布\n"]
    lines += [f"- {k}: {v}" for k, v in fmt.items()]
    lines += ["\n## 清洗动作命中次数（按 trace 计）\n"] + [f"- {k}: {v}" for k, v in events.most_common()]
    lines += [f"\n## 跨轮隔离\n- 历史轮中被隔离掉的工具调用总数：{hist_calls}",
              f"\n## Judge 输入长度（字符）\n- 基线：原始日志原样作为 GEval context（不含 GEval 模板）：平均 {sum(raw_chars) / len(raw_chars):.0f}，最大 {max(raw_chars)}",
              f"- v2 清洗证据 + v2 Prompt：平均 {sum(clean_chars) / len(clean_chars):.0f}，最大 {max(clean_chars)}"]
    text = "\n".join(lines) + "\n"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


def cmd_annotate(a):
    """逐条人工标注：展示清洗后的证据，录入 pass/fail/excluded 与失败层"""
    from agenteval.evaluator import load_targets
    from agenteval.preprocessing import build_evidence, load_raw_traces, preprocess_trace
    construction = {}
    with open(CONSTRUCTION, encoding="utf-8") as f:
        for r in map(json.loads, filter(str.strip, f)):
            construction[r["trace_id"]] = r
    done = set()
    if Path(a.out).exists():
        with open(a.out, encoding="utf-8") as f:
            done = {json.loads(l)["trace_id"] for l in f if l.strip()}
    targets = load_targets(CONSTRUCTION)
    todo = [r for r in load_raw_traces(DATA) if (r.get("id") or r.get("session_id") or r.get("traceId")) not in done]
    print(f"已标注 {len(done)} 条，待标注 {len(todo)} 条。输入 q 退出（进度自动保存）。")
    layers = {"1": "CTX", "2": "OOSR", "3": "OOC", "4": "TCC", "5": "CNT", "6": "RED"}
    with open(a.out, "a", encoding="utf-8") as out:
        for raw in todo:
            t = preprocess_trace(copy.deepcopy(raw))
            turn = targets.get(t.trace_id, len(t.turns) - 1)
            ev = build_evidence(t, turn)
            print("\n" + "=" * 70)
            print(f"[{t.trace_id}]  提问时间 {ev.get('request_time')}  日志完整性 {ev.get('log_integrity')}")
            for i, h in enumerate(ev["conversation_history"]):
                print(f"  历史{i} 用户：{h['user']}\n        助手：{h['assistant']}")
            cur = ev["current_turn"]
            print(f"▶ 用户：{cur['user_query']}")
            for tc in cur["tool_calls"]:
                print(f"  工具 {tc['tool_name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
                print(f"     → {json.dumps(tc['result'], ensure_ascii=False) if tc['error'] is None else 'ERROR ' + tc['error']}")
            print(f"▶ 助手：{cur['agent_reply']}")
            if a.show_suggestion:
                c = construction[t.trace_id]
                print(f"  [构造参考] {c['label']} {c['failed_layer'] or ''} {c['reason']}")
            ans = input("判定 p=pass f=fail e=excluded s=跳过 q=退出 > ").strip().lower()
            if ans == "q":
                break
            if ans not in ("p", "f", "e"):
                continue
            label = {"p": "pass", "f": "fail", "e": "excluded"}[ans]
            layer = "CTX" if label == "excluded" else None
            if label == "fail":
                layer = layers.get(input("失败层 1=CTX 2=OOSR 3=OOC 4=TCC 5=CNT 6=RED > ").strip(), None)
            reason = input("理由（可空）> ").strip()
            row = {"trace_id": t.trace_id, "turn_id": turn, "label": label, "failed_layer": layer,
                   "anomaly_type": construction[t.trace_id]["anomaly_type"], "reason": reason,
                   "label_source": "human", "annotator": a.annotator,
                   "annotated_at": datetime.now().isoformat(timespec="seconds")}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()


# ============== 在线命令（真实 Judge） ==============

def _llm(temp):
    from agenteval.judge import judge_from_env
    return judge_from_env(temperature=temp)


def cmd_run(a):
    from agenteval.evaluator import BASELINE, RunConfig, get_config, run_pipeline
    from agenteval.experiments import consistency_metrics, load_labels
    temp = a.temperature if a.temperature is not None else (
        RunConfig.temperature if a.config == BASELINE else get_config(a.config).temperature)
    labels_path = a.labels or _default_labels()
    res = run_pipeline(a.config, a.input, labels_path, a.out, _llm(temp), workers=a.workers,
                       run_name=a.run_name, temperature=a.temperature, limit=a.limit)
    m = consistency_metrics(res, {k: v for k, v in load_labels(labels_path).items()
                                  if k in {(r.trace_id, r.turn_id) for r in res}})
    print(f"完成 {len(res)} 条：一致率 {m['consistency']:.1%}，Kappa {m['kappa']}，解析失败 {m['parse_failed']}")


def cmd_temperature(a):
    from agenteval.evaluator import load_results, run_pipeline
    from agenteval.experiments import load_labels, stratified_sample, temperature_metrics, temperature_report, write_report
    labels_path = a.labels or _default_labels()
    labels = load_labels(labels_path)
    ids = set(stratified_sample(labels, a.n))
    runs = {}
    for temp in a.temps:
        runs[temp] = []
        for rep in range(1, a.repeats + 1):
            name = f"temp_{temp}_r{rep}"
            print(f"== {name}")
            runs[temp].append(run_pipeline(a.config, a.input, labels_path, a.out, _llm(temp), workers=a.workers,
                                           run_name=name, temperature=temp, only_ids=ids))
    rows = temperature_metrics(runs, labels)
    write_report(temperature_report(rows, labels), "results/temperature_report.md")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    valid = [r for r in rows if r["stability"] is not None]
    best = max(valid, key=lambda r: (round(r["stability"], 3), r["mean_consistency"] or 0, -r["parse_failed"]))
    return best["temperature"]


# ============== 分析命令 ==============

def cmd_consistency(a):
    from agenteval.evaluator import load_results
    from agenteval.experiments import consistency_report, load_labels, write_report
    labels = load_labels(a.labels or _default_labels())
    text = consistency_report(load_results(a.run), labels, Path(a.run).name)
    write_report(text, a.output)
    print(text)


def cmd_compare(a):
    from agenteval.evaluator import load_results
    from agenteval.experiments import comparison_report, load_labels, write_report
    labels = load_labels(a.labels or _default_labels())
    runs = {name: load_results(str(Path(a.root) / name)) for name in a.runs}
    text = comparison_report(runs, labels, a.baseline, a.improved)
    write_report(text, a.output)
    print(text)


def cmd_attribution(a):
    from agenteval.attribution import analyze_run, attribution_report
    from agenteval.evaluator import load_results
    from agenteval.experiments import load_labels, write_report
    labels = load_labels(a.labels or _default_labels())
    res = load_results(a.run)
    attrs = analyze_run(a.run, res, labels, _tags())
    name = Path(a.run).name
    write_report(attribution_report(attrs, res, name), a.output or f"results/attribution_{name}.md")
    with open(Path(a.run) / "attribution.jsonl", "w", encoding="utf-8") as f:
        for x in attrs:
            f.write(x.model_dump_json() + "\n")
    print(Counter(x.primary_cause for x in attrs if x.judge_label != x.human_label))


def cmd_replay(a):
    """用新配置重跑旧版本的 Bad Case + Guard Case，并输出状态迁移报告"""
    from agenteval.evaluator import load_results, run_pipeline
    from agenteval.experiments import load_labels, write_report
    from agenteval.regression import collect_bad_cases, compare_runs, pick_guard_cases, replay_report, save_case_set
    labels_path = a.labels or _default_labels()
    labels = load_labels(labels_path)
    old = load_results(a.old)
    bad, guard = collect_bad_cases(old, labels), pick_guard_cases(old, labels, a.guard)
    save_case_set(bad, f"results/regression/bad_cases_{Path(a.old).name}.json", {"source_run": a.old})
    name = f"replay_{Path(a.old).name}_to_{a.config}"
    new = run_pipeline(a.config, a.input, labels_path, a.out, _llm(a.temperature), workers=a.workers,
                       run_name=name, temperature=a.temperature, only_ids=set(bad) | set(guard))
    trans = compare_runs(old, new, labels)
    text = replay_report(trans, Path(a.old).name, a.config, bad, guard, old, new, labels)
    write_report(text, f"results/{name}.md")
    print(text)


def cmd_experiment_all(a):
    """一键复现全部实验（需要 API Key）"""
    from agenteval.evaluator import BASELINE
    ns = argparse.Namespace
    temperature = a.temperature
    if not a.skip_temperature:
        # 先做 Temperature 梯度实验定参数，再用选出的 temperature 跑正式实验
        best = cmd_temperature(ns(config="v2_improved", temps=[0.0, 0.1, 0.3, 0.7, 1.0], repeats=3, n=50,
                                  input=a.input, labels=a.labels, out=RUNS, workers=a.workers))
        temperature = temperature if temperature is not None else best
        print(f"\n######## 选定 temperature = {temperature}")
    common = dict(input=a.input, labels=a.labels, out=RUNS, workers=a.workers, temperature=temperature,
                  limit=None, run_name=None)
    for cfg in [BASELINE, "v2_improved"] + (a.ablations or []):
        print(f"\n######## run {cfg}")
        cmd_run(ns(config=cfg, **common))
    cmd_consistency(ns(run=f"{RUNS}/v2_improved", labels=a.labels, output="results/consistency_report.md"))
    cmd_compare(ns(root=RUNS, runs=[BASELINE, "v2_improved"] + (a.ablations or []), baseline=BASELINE,
                   improved="v2_improved", labels=a.labels, output="results/comparison_report.md"))
    # 四层归因针对 v2 的分层输出；GEval 基线只有一个总分，没有可归因的分层证据
    cmd_attribution(ns(run=f"{RUNS}/v2_improved", labels=a.labels, output=None))
    cmd_replay(ns(old=f"{RUNS}/{BASELINE}", config="v2_improved", guard=30, **{k: common[k] for k in
                  ("input", "labels", "out", "workers", "temperature")}))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="agenteval", description="AgentEval：基于 Agent Trace 的自动化评测系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("build-dataset", help="生成 200 条评测集")
    s.add_argument("--out", default="data")
    s.set_defaults(fn=cmd_build_dataset)

    s = sub.add_parser("inspect", help="查看单条 trace 的清洗与证据重建过程")
    s.add_argument("--trace-id", required=True)
    s.add_argument("--config", default="v2_improved")
    s.add_argument("--input", default=DATA)
    s.add_argument("--prompt", action="store_true", help="同时打印完整 Prompt")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("clean-stats", help="四层清洗统计")
    s.add_argument("--input", default=DATA)
    s.add_argument("--out", default="results/cleaning_stats.md")
    s.set_defaults(fn=cmd_clean_stats)

    s = sub.add_parser("annotate", help="人工标注")
    s.add_argument("--out", default=HUMAN)
    s.add_argument("--annotator", default="me")
    s.add_argument("--show-suggestion", action="store_true", help="显示构造标签作参考（会引入锚定偏差，默认关闭）")
    s.set_defaults(fn=cmd_annotate)

    def online(s):
        s.add_argument("--config", default="v2_improved")
        s.add_argument("--input", default=DATA)
        s.add_argument("--labels", default=None)
        s.add_argument("--out", default=RUNS)
        s.add_argument("--workers", type=int, default=8)
        s.add_argument("--temperature", type=float, default=None)
        s.add_argument("--limit", type=int, default=None)

    s = sub.add_parser("run", help="批量评测")
    online(s)
    s.add_argument("--run-name", default=None)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("temperature", help="Temperature 梯度实验")
    online(s)
    s.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.7, 1.0])
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--n", type=int, default=50, help="分层抽样的样本数")
    s.set_defaults(fn=cmd_temperature)

    s = sub.add_parser("consistency", help="一致率报告")
    s.add_argument("--run", required=True)
    s.add_argument("--labels", default=None)
    s.add_argument("--output", default="results/consistency_report.md")
    s.set_defaults(fn=cmd_consistency)

    s = sub.add_parser("compare", help="多配置误判率对比")
    s.add_argument("--root", default=RUNS)
    s.add_argument("--runs", nargs="+", default=["baseline_geval", "v2_improved"])
    s.add_argument("--baseline", default="baseline_geval")
    s.add_argument("--improved", default="v2_improved")
    s.add_argument("--labels", default=None)
    s.add_argument("--output", default="results/comparison_report.md")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("attribution", help="四层错误归因")
    s.add_argument("--run", required=True)
    s.add_argument("--labels", default=None)
    s.add_argument("--output", default=None)
    s.set_defaults(fn=cmd_attribution)

    s = sub.add_parser("replay", help="异常样本回放与回归验证")
    online(s)
    s.add_argument("--old", required=True, help="旧版本运行目录，如 results/runs/baseline_geval")
    s.add_argument("--guard", type=int, default=30)
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("experiment-all", help="一键跑全部实验")
    s.add_argument("--input", default=DATA)
    s.add_argument("--labels", default=None)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--temperature", type=float, default=None)
    s.add_argument("--ablations", nargs="*", default=["abl_no_isolation", "abl_no_noise_mask",
                                                       "abl_no_semantic", "abl_no_hallu"])
    s.add_argument("--skip-temperature", action="store_true")
    s.set_defaults(fn=cmd_experiment_all)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
