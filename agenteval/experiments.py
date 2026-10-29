"""
实验与指标：Judge-Human 一致率、误判率对比、Temperature 梯度实验

指标口径（实验前规定，全部从结果文件现场计算，不存在任何硬编码数值）：
- 全样本一致率：全部有标签样本（200 条），标签取 pass / fail / excluded 三类，Judge 最终标签与人工标签相同才算一致
- 可判样本一致率：只看人工标签为 pass / fail 的样本（约 185 条）；Judge 在这些样本上输出 excluded（误拒评）算不一致
- N/A 识别：人工标为 excluded 的样本中 Judge 也输出 excluded 的比例（召回），以及可判样本上的误拒评数；
  不支持拒评的基线（GEval）召回为 0，单独列出，不并入 pass/fail
- ERROR：调用失败或输出无法解析的样本，在两种一致率中都计为不一致（不从分母剔除，避免美化指标），并单独计数
- 误判率 = 1 - 一致率；误判率相对降低 = (Baseline 误判率 - Improved 误判率) / Baseline 误判率，两种口径分别给出，
  对外引用时必须注明口径
- Cohen's Kappa：扣除随机一致后的一致性（淘宝文章 3.6 节提到的学术口径）
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agenteval.models import CaseResult, GroundTruth

LABELS = ["pass", "fail", "excluded"]


def load_labels(path: str) -> Dict[Tuple[str, int], GroundTruth]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                gt = GroundTruth.model_validate(row)
                out[(gt.trace_id, gt.turn_id)] = gt
    return out


def label_sources(labels: Dict[Tuple[str, int], GroundTruth]) -> Counter:
    return Counter(gt.label_source for gt in labels.values())


def cohen_kappa(pairs: List[Tuple[str, str]]) -> Optional[float]:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def consistency_metrics(results: List[CaseResult], labels: Dict[Tuple[str, int], GroundTruth],
                        include_excluded: bool = True) -> Dict:
    """Judge 与标签的一致性；include_excluded=False 时只看人工标签为 pass/fail 的样本"""
    by_key = {(r.trace_id, r.turn_id): r for r in results}
    pairs, rows = [], []
    for key, gt in labels.items():
        if not include_excluded and gt.label == "excluded":
            continue
        r = by_key.get(key)
        judge = r.final_label if r else None
        pairs.append((gt.label, judge or "parse_failed"))
        rows.append((gt, r))

    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    confusion = defaultdict(Counter)
    for h, j in pairs:
        confusion[h][j] += 1

    # 按异常类型统计判对率
    per_type = defaultdict(lambda: [0, 0])
    for (gt, r), (h, j) in zip(rows, pairs):
        k = gt.anomaly_type or "normal"
        per_type[k][1] += 1
        per_type[k][0] += h == j

    # 失败层定位准确率：人工与 Judge 都判 fail 时，失败层是否一致
    both_fail = [(gt, r) for gt, r in rows if r and gt.label == "fail" and r.final_label == "fail"]
    layer_hit = sum(gt.failed_layer == r.failed_layer for gt, r in both_fail)

    return {
        "n": n,
        "agree": agree,
        "consistency": agree / n if n else None,
        "misjudgment": 1 - agree / n if n else None,
        "kappa": cohen_kappa(pairs),
        "parse_failed": sum(j == "parse_failed" for _, j in pairs),
        "confusion": {h: dict(c) for h, c in confusion.items()},
        "per_type": {k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1]} for k, v in sorted(per_type.items())},
        "layer_localization": {"both_fail": len(both_fail), "same_layer": layer_hit,
                               "acc": layer_hit / len(both_fail) if both_fail else None},
    }


def relative_reduction(base: float, improved: float) -> Optional[float]:
    return (base - improved) / base if base else None


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


def _source_note(labels) -> str:
    src = label_sources(labels)
    if src.get("construction"):
        return (f"> ⚠️ 标签来源：construction {src.get('construction', 0)} 条 / human {src.get('human', 0)} 条。"
                f"construction 是数据构造时写入的期望标签，**不等于人工标注**；完成人工复核（annotate 命令）后再对外报告"
                f"\"人工标注一致率\"。\n")
    return f"> 标签来源：人工标注 {src.get('human', 0)} 条。\n"


def na_metrics(results: List[CaseResult], labels: Dict[Tuple[str, int], GroundTruth]) -> Dict:
    """N/A（CTX 剔除）识别表现与 ERROR 计数"""
    pred = {(r.trace_id, r.turn_id): r.final_label for r in results}
    gold_na = [k for k, gt in labels.items() if gt.label == "excluded"]
    judgeable = [k for k, gt in labels.items() if gt.label != "excluded"]
    hit = sum(pred.get(k) == "excluded" for k in gold_na)
    return {"na_total": len(gold_na), "na_hit": hit, "na_recall": hit / len(gold_na) if gold_na else None,
            "false_na": sum(pred.get(k) == "excluded" for k in judgeable), "judgeable": len(judgeable),
            "errors": sum(pred.get(k) is None for k in labels)}


def summary_metrics(results: List[CaseResult], labels) -> Dict:
    return {"full": consistency_metrics(results, labels),
            "judgeable": consistency_metrics(results, labels, include_excluded=False),
            "na": na_metrics(results, labels)}


def _na_cell(na: Dict, supports_na: bool) -> str:
    if not supports_na:
        return f"不支持（0/{na['na_total']}）"
    return f"{na['na_hit']}/{na['na_total']} = {_pct(na['na_recall'])}"


def consistency_report(results: List[CaseResult], labels, run_name: str) -> str:
    s = summary_metrics(results, labels)
    m, m2, na = s["full"], s["judgeable"], s["na"]
    lines = [f"# Judge-Human 一致性报告：{run_name}\n", _source_note(labels),
             "| 指标 | 全样本 | 可判样本（人工标签为 pass/fail） |", "|---|---|---|",
             f"| 样本数 | {m['n']} | {m2['n']} |",
             f"| 一致率 | {_pct(m['consistency'])} | {_pct(m2['consistency'])} |",
             f"| 误判率 | {_pct(m['misjudgment'])} | {_pct(m2['misjudgment'])} |",
             f"| Cohen's Kappa | {_fmt(m['kappa'])} | {_fmt(m2['kappa'])} |",
             f"\n- N/A 识别（人工 excluded 中 Judge 也剔除）：{_na_cell(na, True)}",
             f"- 误拒评（可判样本上 Judge 输出 excluded）：{na['false_na']}/{na['judgeable']}",
             f"- ERROR（调用/解析失败，已计为不一致）：{na['errors']}",
             f"- 失败层定位准确率（双方都判 fail 时失败层一致）：{m['layer_localization']['same_layer']}/"
             f"{m['layer_localization']['both_fail']} = {_pct(m['layer_localization']['acc'])}\n",
             "## 混淆矩阵（行=人工标签，列=Judge）\n", "| 人工 \\ Judge | pass | fail | excluded | ERROR |",
             "|---|---|---|---|---|"]
    for h in LABELS:
        c = m["confusion"].get(h, {})
        lines.append(f"| {h} | {c.get('pass', 0)} | {c.get('fail', 0)} | {c.get('excluded', 0)} | {c.get('parse_failed', 0)} |")
    lines += ["\n## 分类型判对率\n", "| 类型 | 判对 / 总数 | 判对率 |", "|---|---|---|"]
    for k, v in m["per_type"].items():
        lines.append(f"| {k} | {v['correct']}/{v['total']} | {_pct(v['acc'])} |")
    return "\n".join(lines) + "\n"


def threshold_sensitivity(results: List[CaseResult], labels, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)) -> List[Dict]:
    """连续分基线在不同阈值下的可判样本一致率（只作参考，主结果固定用事先规定的阈值）"""
    rows = []
    for th in thresholds:
        relabeled = [r.model_copy(update={"final_label": None if r.score is None else ("pass" if r.score >= th else "fail")})
                     for r in results]
        rows.append({"threshold": th,
                     "consistency": consistency_metrics(relabeled, labels, include_excluded=False)["consistency"]})
    return rows


def comparison_report(runs: Dict[str, List[CaseResult]], labels, baseline: str, improved: str) -> str:
    """Baseline（原始日志 + 通用 GEval）vs Improved（证据治理 + 六层 Rubric），以及 v2 的消融配置"""
    lines = ["# 误判率对比实验\n", _source_note(labels),
             "| 配置 | 全样本一致率 | 可判样本一致率 | Kappa（可判） | N/A 识别 | 误拒评 | ERROR |",
             "|---|---|---|---|---|---|---|"]
    ms = {}
    for name, res in runs.items():
        s = summary_metrics(res, labels)
        ms[name] = s
        supports_na = any(r.layers for r in res)  # GEval 结果没有分层输出，也就没有剔除能力
        lines.append(f"| {name} | {_pct(s['full']['consistency'])}（{s['full']['agree']}/{s['full']['n']}） "
                     f"| {_pct(s['judgeable']['consistency'])}（{s['judgeable']['agree']}/{s['judgeable']['n']}） "
                     f"| {_fmt(s['judgeable']['kappa'])} | {_na_cell(s['na'], supports_na)} "
                     f"| {s['na']['false_na']} | {s['na']['errors']} |")
    if baseline in ms and improved in ms:
        b, i = ms[baseline], ms[improved]
        r_full = relative_reduction(b["full"]["misjudgment"], i["full"]["misjudgment"])
        r_judge = relative_reduction(b["judgeable"]["misjudgment"], i["judgeable"]["misjudgment"])
        lines += [f"\n**误判率相对降低（{baseline} → {improved}）**：",
                  f"- 全样本口径：({_pct(b['full']['misjudgment'])} - {_pct(i['full']['misjudgment'])}) / "
                  f"{_pct(b['full']['misjudgment'])} = **{_pct(r_full)}**"
                  f"（基线不支持拒评，人工 excluded 样本在此口径下必然记为基线误判）",
                  f"- 可判样本口径：({_pct(b['judgeable']['misjudgment'])} - {_pct(i['judgeable']['misjudgment'])}) / "
                  f"{_pct(b['judgeable']['misjudgment'])} = **{_pct(r_judge)}**（同一批可判样本上比较判断准确性）"]
    scored = runs.get(baseline)
    if scored and any(r.score is not None for r in scored):
        lines += ["\n## 基线阈值敏感性（仅供参考）\n",
                  "主结果固定使用 GEval 默认阈值 0.5（实验前规定）。下表只用于检查结论是否依赖阈值，不据此回调基线：\n",
                  "| 阈值 | 可判样本一致率 |", "|---|---|"]
        lines += [f"| {r['threshold']} | {_pct(r['consistency'])} |" for r in threshold_sensitivity(scored, labels)]
    lines += ["\n## 结论边界\n",
              "- 基线与改进方案的判定规则不同（GEval 连续分 + 阈值 vs 六层短路），优劣只以人工标签为准，不比较通过率高低",
              "- 结论只适用于“原始日志 + DeepEval G-Eval 通用整体质量评测”这一基线，"
              "不代表优于 DeepEval 的 Agent 专用指标或其他评测系统"]
    return "\n".join(lines) + "\n"


# ============== Temperature 梯度实验 ==============


def stratified_sample(labels: Dict[Tuple[str, int], GroundTruth], n: int, seed: int = 7) -> List[str]:
    """按异常类型分层抽样，保证每类都有代表"""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for gt in labels.values():
        groups[gt.anomaly_type or "normal"].append(gt.trace_id)
    total = sum(len(v) for v in groups.values())
    picked = []
    for k in sorted(groups):
        ids = sorted(groups[k])
        take = max(1, round(n * len(ids) / total))
        picked += rng.sample(ids, min(take, len(ids)))
    return sorted(picked)[:n] if len(picked) > n else sorted(picked)


def temperature_metrics(runs: Dict[float, List[List[CaseResult]]], labels) -> List[Dict]:
    """
    runs[temp] = [第1次运行结果, 第2次..., ...]
    - 稳定性：同一样本多次运行最终标签完全一致的比例
    - 准确率：每次运行一致率的均值
    """
    rows = []
    for temp in sorted(runs):
        reps = runs[temp]
        maps = [{(r.trace_id, r.turn_id): r.final_label for r in rep} for rep in reps]
        keys = set().union(*maps)
        stable = sum(len({m.get(k) for m in maps}) == 1 and None not in {m.get(k) for m in maps} for k in keys)
        accs = [consistency_metrics(rep, {k: v for k, v in labels.items() if k in keys})["consistency"] for rep in reps]
        fails = sum(r.final_label is None for rep in reps for r in rep)
        rows.append({"temperature": temp, "repeats": len(reps), "samples": len(keys),
                     "stability": stable / len(keys) if keys else None,
                     "mean_consistency": sum(accs) / len(accs) if accs else None,
                     "min_consistency": min(accs) if accs else None, "parse_failed": fails})
    return rows


def temperature_report(rows: List[Dict], labels) -> str:
    lines = ["# Temperature 梯度实验\n", _source_note(labels),
             "同一批样本在每个 temperature 下重复运行多次：稳定性 = 多次运行最终标签完全一致的样本占比。\n",
             "| temperature | 重复次数 | 样本数 | 稳定性 | 平均一致率 | 最低一致率 | 解析失败次数 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['temperature']} | {r['repeats']} | {r['samples']} | {_pct(r['stability'])} | "
                     f"{_pct(r['mean_consistency'])} | {_pct(r['min_consistency'])} | {r['parse_failed']} |")
    valid = [r for r in rows if r["stability"] is not None]
    if valid:
        best = max(valid, key=lambda r: (round(r["stability"], 3), r["mean_consistency"] or 0, -r["parse_failed"]))
        lines.append(f"\n按\"稳定性优先、其次一致率\"选择：**temperature = {best['temperature']}**")
    return "\n".join(lines) + "\n"


def write_report(text: str, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
