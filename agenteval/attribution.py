"""
四层错误归因（参考淘宝文章 3.6 节"单点 Case 修复归因"与 4.1 节）

归因对象：Judge 与人工标签不一致的样本 + 双方都判 fail 的样本。
按 数据层 → Prompt 层 → Rubric 层 → Agent 层 的顺序定位根因，先排除评测链路自身的问题，
剩下的才算 Agent 真实缺陷：

1. 数据层：喂给 Judge 的证据本身有问题（残留提示词/噪声调用、未做跨轮隔离、缺时间锚点、状态码未映射、截断）
         → 修数据清洗逻辑，影响该类所有数据
2. Prompt 层：数据是对的，但 Judge 没按规则判（证据引用不是原文、推测性措辞、输出非标、该剔除未剔除、失败层错位）
         → 补判定规则或 Few-shot
3. Rubric 层：批量统计发现某层在某类场景上系统性判严/判松，说明标准本身与预期不一致
         → 改 Rubric 定义并同步 Prompt
4. Agent 层：排除以上三类后，人工也判失败的样本 → Agent 真实质量问题，反馈给 Agent 开发
   排除以上三类后仍不一致、且人工判通过 → unresolved，需要人工复核
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agenteval.models import SERIAL_LAYERS, AttributionResult, CaseResult, GroundTruth
from agenteval.preprocessing import NOISE_TEXT_PATTERNS, NOISE_TOOL_NAMES, STATUS_GLOSSARY

SPECULATIVE = re.compile(r"可能|似乎|推测|大概|也许|应该是|猜测|估计")
TIME_EXPR = re.compile(r"今天|明天|后天|\d+个?工作日|\d+月\d+日")
RUBRIC_MIN_CLUSTER = 3  # 同一层、同一场景、同一方向的不一致达到该数量才认定为 Rubric 系统性问题


# ============== 数据层 ==============

def _walk(value, keys: List[str], strings: List[str]):
    if isinstance(value, dict):
        for k, v in value.items():
            keys.append(k)
            _walk(v, keys, strings)
    elif isinstance(value, list):
        for v in value:
            _walk(v, keys, strings)
    elif isinstance(value, str):
        strings.append(value)


def check_data_layer(evidence: Dict, result: CaseResult) -> List[str]:
    issues = []
    cur = evidence["current_turn"]
    calls = list(cur["tool_calls"]) + [tc for h in evidence["conversation_history"] for tc in h.get("tool_calls", [])]

    keys, strings = [], []
    for tc in calls:
        _walk(tc["result"], keys, strings)
    if any(tc["tool_name"] in NOISE_TOOL_NAMES for tc in calls):
        issues.append("证据中残留日志/埋点类噪声工具调用")
    if any(k.startswith("_") or k == "debug_info" for k in keys) or any(p.search(s) for p in NOISE_TEXT_PATTERNS for s in strings):
        issues.append("工具返回中残留透传的系统提示词/开发注释")
    if any("tool_calls" in h for h in evidence["conversation_history"]):
        issues.append("证据未做跨轮隔离：历史轮工具返回进入了 Judge 输入")
    if TIME_EXPR.search(cur["agent_reply"]) and not evidence.get("request_time"):
        issues.append("回复含相对时间表述，但证据缺少提问时间锚点")
    if any(s in STATUS_GLOSSARY for s in strings) and not evidence.get("status_glossary") \
            and not any(k.endswith("_cn") for k in keys):
        issues.append("工具返回含系统状态码，但未做用户语义映射")
    if any("[已截断" in s for s in strings) and result.failed_layer == "CNT":
        issues.append("CNT 判失败且证据中存在被截断字段，需确认关键信息未被截掉")
    return issues


# ============== Prompt 层 ==============

def _quote_grounded(quote: str, source: str) -> bool:
    """证据引用是否来自输入原文：去空白后，引用中任一 8 字片段出现在输入里即视为有据"""
    q = re.sub(r"\s+", "", quote)
    s = re.sub(r"\s+", "", source)
    if len(q) < 8:
        return q in s
    return any(q[i:i + 8] in s for i in range(0, len(q) - 7, 4))


def check_prompt_layer(evidence: Dict, prompt: str, result: CaseResult, gt: GroundTruth) -> List[str]:
    issues = []
    if result.parse_status == "failed":
        return ["Judge 输出无法解析（结构化输出约束失效）"]
    if result.parse_status == "normalized":
        issues.append(f"Judge 输出非标取值，经归一化处理：{'; '.join(result.normalize_notes[:2])}")

    decisive = result.failed_layer or result.short_circuit_at
    if decisive:
        lr = result.layers[decisive]
        if not lr.evidence_quote.strip():
            issues.append(f"{decisive} 层判定未给出证据引用")
        elif not _quote_grounded(lr.evidence_quote, json.dumps(evidence, ensure_ascii=False)):
            issues.append(f"{decisive} 层的证据引用不是输入原文（疑似幻觉）")
        if SPECULATIVE.search(lr.reason):
            issues.append(f"{decisive} 层判定理由含推测性措辞")

    marks = evidence.get("log_integrity") or []
    if gt.label == "excluded" and result.final_label != "excluded" and marks:
        issues.append("日志完整性已标记证据缺失，Judge 仍强行判分（未遵守禁止推断）")
    if result.final_label == "excluded" and gt.label != "excluded" and not marks:
        issues.append("无证据缺失标记，Judge 却剔除了样本")
    if gt.label == "fail" and result.final_label == "fail" and gt.failed_layer and gt.failed_layer != result.failed_layer:
        issues.append(f"失败层错位：人工判 {gt.failed_layer}，Judge 判 {result.failed_layer}（子项判定边界模糊）")
    return issues


# ============== Rubric 层（批量统计） ==============

def rubric_clusters(items: List[Tuple[CaseResult, GroundTruth, List[str]]]) -> Dict[Tuple[str, str, str], int]:
    """
    统计不一致样本的 (层, 方向, 场景标签) 簇
      方向 strict：Judge 判 fail、人工判 pass（标准过严）
      方向 loose ：Judge 判 pass、人工判 fail（标准过松）
    """
    clusters = Counter()
    for res, gt, tags in items:
        if res.final_label == "fail" and gt.label == "pass":
            for tag in tags or ["-"]:
                clusters[(res.failed_layer, "strict", tag)] += 1
        elif res.final_label == "pass" and gt.label == "fail":
            for tag in tags or ["-"]:
                clusters[(gt.failed_layer, "loose", tag)] += 1
    return clusters


def check_rubric_layer(result: CaseResult, gt: GroundTruth, tags: List[str], clusters) -> List[str]:
    issues = []
    if result.final_label == "fail" and gt.label == "pass":
        direction, layer = "strict", result.failed_layer
    elif result.final_label == "pass" and gt.label == "fail":
        direction, layer = "loose", gt.failed_layer
    else:
        return issues
    for tag in tags or ["-"]:
        n = clusters.get((layer, direction, tag), 0)
        if n >= RUBRIC_MIN_CLUSTER:
            word = "判严" if direction == "strict" else "判松"
            issues.append(f"{layer} 层在 [{tag}] 场景系统性{word}（共 {n} 例），Rubric 标准与预期不一致")
    return issues


# ============== 主流程 ==============

def analyze_run(run_dir: str, results: List[CaseResult], labels: Dict[Tuple[str, int], GroundTruth],
                tags_by_trace: Optional[Dict[str, List[str]]] = None) -> List[AttributionResult]:
    tags_by_trace = tags_by_trace or {}
    evid = {}
    with open(Path(run_dir) / "evidence.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            evid[(row["trace_id"], row["turn_id"])] = row

    targets = []
    for r in results:
        gt = labels.get((r.trace_id, r.turn_id))
        if gt is None:
            continue
        if r.final_label != gt.label or (gt.label == "fail" and r.final_label == "fail"):
            targets.append((r, gt, tags_by_trace.get(r.trace_id, [])))
    clusters = rubric_clusters(targets)

    out = []
    for r, gt, tags in targets:
        row = evid[(r.trace_id, r.turn_id)]
        data = check_data_layer(row["evidence"], r)
        prompt = check_prompt_layer(row["evidence"], row["prompt"], r, gt)
        rubric = check_rubric_layer(r, gt, tags, clusters)
        agent = None
        if r.final_label == gt.label:
            # 判定一致且都为 fail：评测链路无误，确认 Agent 真实缺陷
            primary = "agent"
            agent = f"Agent 真实缺陷（{gt.anomaly_type or gt.failed_layer}）：{gt.reason}"
        elif data:
            primary = "data"
        elif prompt:
            primary = "prompt"
        elif rubric:
            primary = "rubric"
        else:
            primary = "unresolved"
        out.append(AttributionResult(trace_id=r.trace_id, turn_id=r.turn_id, judge_label=r.final_label,
                                     human_label=gt.label, data_issues=data, prompt_issues=prompt,
                                     rubric_issues=rubric, agent_issue=agent, primary_cause=primary))
    return out


FIX_HINT = {
    "data": "修改清洗逻辑（噪声正则、跨轮隔离、时间锚点、状态映射），影响该类全部数据",
    "prompt": "补充判定规则或 Few-shot 边界样本，强化证据引用与禁止推断约束",
    "rubric": "修改 Rubric 定义并同步更新 Prompt，记录到 RUBRIC_CHANGELOG",
    "agent": "反馈给 Agent 开发侧修复（模型/工具/话术）",
    "unresolved": "规则未能定位，需人工复核该样本（可能是标注本身存疑）",
}


def attribution_report(attrs: List[AttributionResult], results: List[CaseResult], run_name: str) -> str:
    lines = [f"# 四层错误归因报告：{run_name}\n"]
    # 宏观：按漏斗层拆解 Judge 判失败的分布（淘宝文章 4.1"低在哪一层"）
    layer_dist = Counter(r.failed_layer for r in results if r.final_label == "fail")
    lines += ["## 宏观：Judge 判失败样本的漏斗层分布\n", "| 层 | 失败数 |", "|---|---|"]
    for layer in SERIAL_LAYERS[1:] + ["RED"]:
        lines.append(f"| {layer} | {layer_dist.get(layer, 0)} |")
    lines.append(f"\n被 CTX 剔除：{sum(r.final_label == 'excluded' for r in results)} 条\n")

    disagree = [a for a in attrs if a.judge_label != a.human_label]
    lines += ["## 微观：不一致样本的根因分布\n", f"不一致样本 {len(disagree)} 条；双方都判 fail（确认 Agent 缺陷）"
              f" {sum(a.judge_label == a.human_label for a in attrs)} 条。\n",
              "| 根因 | 数量 | 修复方向 |", "|---|---|---|"]
    cnt = Counter(a.primary_cause for a in disagree)
    for k in ["data", "prompt", "rubric", "unresolved"]:
        lines.append(f"| {k} | {cnt.get(k, 0)} | {FIX_HINT[k]} |")

    issue_cnt = Counter()
    for a in disagree:
        for s in a.data_issues + a.prompt_issues + a.rubric_issues:
            issue_cnt[re.sub(r"[（(].*?[）)]|\d+", "", s)] += 1
    lines += ["\n## 高频问题\n", "| 问题 | 次数 |", "|---|---|"]
    for s, n in issue_cnt.most_common(15):
        lines.append(f"| {s} | {n} |")

    lines += ["\n## 不一致样本明细\n", "| trace | 人工 | Judge | 根因 | 依据 |", "|---|---|---|---|---|"]
    for a in sorted(disagree, key=lambda x: x.trace_id):
        why = (a.data_issues or a.prompt_issues or a.rubric_issues or ["-"])[0]
        lines.append(f"| {a.trace_id} | {a.human_label} | {a.judge_label} | {a.primary_cause} | {why} |")
    return "\n".join(lines) + "\n"
