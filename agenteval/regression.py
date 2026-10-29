"""
异常样本回放与回归验证

用途：评测规则（清洗逻辑 / Prompt / Rubric）迭代后，验证"修好了旧问题、没引入新问题"。

流程：
1. 从旧版本运行结果中收集 Bad Case（Judge 与人工标签不一致的样本）
2. 再从旧版本判对的样本中分层抽取一批 Guard Case（护栏样本）
3. 用新配置只重跑 Bad + Guard 样本（不必全量重跑）
4. 逐条比较状态迁移：
   fixed      旧错 → 新对（迭代生效）
   still_bad  旧错 → 新错（未收敛，进入下一轮迭代）
   regressed  旧对 → 新错（新规则引入的回归，必须关注）
   stable     旧对 → 新对
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agenteval.models import CaseResult, GroundTruth


def collect_bad_cases(results: List[CaseResult], labels: Dict[Tuple[str, int], GroundTruth]) -> List[str]:
    return sorted(r.trace_id for r in results
                  if (r.trace_id, r.turn_id) in labels and r.final_label != labels[(r.trace_id, r.turn_id)].label)


def pick_guard_cases(results: List[CaseResult], labels, n: int = 30, seed: int = 11) -> List[str]:
    """从旧版本判对的样本中按标签类型分层抽取护栏样本"""
    good = [r for r in results if (r.trace_id, r.turn_id) in labels
            and r.final_label == labels[(r.trace_id, r.turn_id)].label]
    by_type: Dict[str, List[str]] = {}
    for r in good:
        gt = labels[(r.trace_id, r.turn_id)]
        by_type.setdefault(gt.anomaly_type or "normal", []).append(r.trace_id)
    rng = random.Random(seed)
    picked = []
    total = max(1, len(good))
    for k in sorted(by_type):
        ids = sorted(by_type[k])
        picked += rng.sample(ids, min(len(ids), max(1, round(n * len(ids) / total))))
    return sorted(picked)


def compare_runs(old: List[CaseResult], new: List[CaseResult], labels) -> Dict[str, List[str]]:
    """逐条比较两个版本相对人工标签的对错迁移"""
    old_map = {(r.trace_id, r.turn_id): r for r in old}
    trans: Dict[str, List[str]] = {"fixed": [], "still_bad": [], "regressed": [], "stable": []}
    for r in new:
        key = (r.trace_id, r.turn_id)
        if key not in labels or key not in old_map:
            continue
        gt = labels[key].label
        was_ok, now_ok = old_map[key].final_label == gt, r.final_label == gt
        state = {(False, True): "fixed", (False, False): "still_bad",
                 (True, False): "regressed", (True, True): "stable"}[(was_ok, now_ok)]
        trans[state].append(r.trace_id)
    return trans


def replay_report(trans: Dict[str, List[str]], old_name: str, new_name: str, bad: List[str], guard: List[str],
                  old: List[CaseResult], new: List[CaseResult], labels) -> str:
    old_map = {r.trace_id: r for r in old}
    new_map = {r.trace_id: r for r in new}
    n_bad = len(bad)
    fixed = len(trans["fixed"])
    lines = [f"# 异常样本回放：{old_name} → {new_name}\n",
             f"- Bad Case（旧版本判错）：{n_bad} 条；Guard Case（旧版本判对的护栏样本）：{len(guard)} 条",
             f"- 修复：{fixed}/{n_bad}" + (f"（{fixed / n_bad * 100:.1f}%）" if n_bad else ""),
             f"- 未收敛：{len(trans['still_bad'])} 条",
             f"- 回归（护栏样本由对变错）：{len(trans['regressed'])}/{len(guard)} 条\n",
             "## 按异常类型\n", "| 类型 | fixed | still_bad | regressed | stable |", "|---|---|---|---|---|"]
    by_type = {}
    for state, ids in trans.items():
        for tid in ids:
            gt = next(v for k, v in labels.items() if k[0] == tid)
            by_type.setdefault(gt.anomaly_type or "normal", Counter())[state] += 1
    for k in sorted(by_type):
        c = by_type[k]
        lines.append(f"| {k} | {c['fixed']} | {c['still_bad']} | {c['regressed']} | {c['stable']} |")
    for state in ["still_bad", "regressed"]:
        if trans[state]:
            lines += [f"\n## {state} 明细\n", "| trace | 人工 | 旧 Judge | 新 Judge | 新版失败层 |", "|---|---|---|---|---|"]
            for tid in trans[state]:
                gt = next(v for k, v in labels.items() if k[0] == tid)
                o, n = old_map.get(tid), new_map.get(tid)
                lines.append(f"| {tid} | {gt.label} | {o.final_label if o else '-'} | {n.final_label if n else '-'} "
                             f"| {n.failed_layer if n else '-'} |")
    return "\n".join(lines) + "\n"


def save_case_set(ids: List[str], path: str, meta: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"trace_ids": ids, **(meta or {})}, ensure_ascii=False, indent=2), encoding="utf-8")
