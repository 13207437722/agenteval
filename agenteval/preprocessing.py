"""
Trace 预处理：四层数据清洗 + 证据重建（参考淘宝文章 3.1 节）

    原始异构日志
      │ 第一层 normalize_trace      格式统一：三种日志格式 → AgentTrace；时间格式归一；
      │                             JSON 字符串解析；超时重试去重；超长文本截断
      │ 第二层 align_semantics      语义对齐：补充请求时间锚点；系统状态枚举 → 用户语义
      │ 第三层 mask_noise           屏蔽干扰：剔除日志/埋点类噪声工具；正则剔除工具返回中透传的系统提示词
      │ 第四层 isolate_turn_evidence 跨轮隔离：只保留当前轮 tool_calls，历史轮只保留对话文本
      ▼
    build_evidence → 喂给 Judge 的证据字典

每一层都会往 trace.cleaning_log 写记录，数据层归因时用来判断"数据是否已被正确处理"。
"""

import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from agenteval.models import AgentTrace, ToolCall, Turn

# 系统状态码 → 用户语义（淘宝文章 3.1 第二层："已揽收"≠"已发货"）
STATUS_GLOSSARY = {
    "WAIT_PICKUP": "待揽收（商家已打包，快递员尚未取件）",
    "COLLECTED": "已揽收（快递员已取件，尚未离开揽收网点）",
    "IN_TRANSIT": "运输中",
    "DELIVERING": "派送中",
    "SIGNED": "已签收",
    "SHIPPED": "已发货",
    "REFUND_PROCESSING": "退款处理中",
    "REFUND_SUCCESS": "退款成功",
    "REFUND_REJECTED": "退款被拒绝",
    "WAIT_SELLER_AGREE": "等待商家处理退款申请",
}

# 日志/埋点/健康检查类工具：不是 Agent 业务行为，对评测无意义
NOISE_TOOL_NAMES = {"log_debug", "log_info", "trace_span", "health_check", "ping"}
# 工具返回中透传的内部字段（以下划线开头或调试字段）
NOISE_RESULT_KEYS = {"debug_info"}
# 工具返回字符串中透传的生产链路提示词
NOISE_TEXT_PATTERNS = [
    re.compile(r"<<SYS>>.*?<</SYS>>", re.S),
    re.compile(r"【(?:内部指令|系统prompt|系统提示)】[^\n]*"),
    re.compile(r"#\s*DEV NOTE:[^\n]*"),
]

MAX_FIELD_CHARS = 300  # 单个字符串字段截断阈值（防止超长卡片撑爆上下文）


@dataclass
class CleanConfig:
    """清洗开关：Baseline 关闭清洗，Improved 全开，消融实验逐项关闭"""

    dedupe_retry: bool = True
    truncate: bool = True
    semantic_align: bool = True
    mask_noise: bool = True


RAW_ONLY = CleanConfig(dedupe_retry=False, truncate=False, semantic_align=False, mask_noise=False)


# ============== 工具函数 ==============

def parse_time(value: Any) -> Optional[str]:
    """把 ISO / epoch 毫秒 / 'YYYY/MM/DD HH:MM:SS' 统一成 'YYYY-MM-DDTHH:MM:SS'"""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%dT%H:%M:%S")
    s = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return None


def _maybe_json(value: Any) -> Any:
    """日志里参数/返回值常被序列化成字符串，尝试还原"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _split_error(result: Any, error: Optional[str]):
    """openai 格式把错误塞在 tool content 的 {"error": ...} 里"""
    if error is None and isinstance(result, dict) and set(result) == {"error"}:
        return None, str(result["error"])
    return result, error


# ============== 第一层：格式统一 ==============

def _from_openai(raw: Dict[str, Any]) -> List[Turn]:
    turns, pending = [], {}
    cur: Optional[Dict[str, Any]] = None
    for m in raw["messages"]:
        role = m.get("role")
        if role == "user":
            cur = {"user": m.get("content", ""), "time": parse_time(m.get("timestamp")), "calls": []}
        elif role == "assistant" and m.get("tool_calls") and cur is not None:
            for call in m["tool_calls"]:
                fn = call.get("function", {})
                tc = ToolCall(tool_name=fn.get("name", "unknown"), arguments=_maybe_json(fn.get("arguments")) or {})
                pending[call.get("id")] = tc
                cur["calls"].append(tc)
        elif role == "tool":
            tc = pending.pop(m.get("tool_call_id"), None)
            if tc is not None:
                tc.result, tc.error = _split_error(_maybe_json(m.get("content")), None)
        elif role == "assistant" and cur is not None:
            turns.append(Turn(turn_id=len(turns), user_query=cur["user"], agent_reply=m.get("content") or "",
                              tool_calls=cur["calls"], request_time=cur["time"]))
            cur = None
    return turns


def _from_event_log(raw: Dict[str, Any]) -> List[Turn]:
    turns = []
    cur: Optional[Dict[str, Any]] = None
    for ev in sorted(raw["events"], key=lambda e: e.get("ts", 0)):
        t = ev.get("type")
        if t == "USER_MSG":
            cur = {"user": ev.get("text", ""), "time": parse_time(ev.get("ts")), "calls": []}
        elif t == "TOOL_INVOKE" and cur is not None:
            cur["calls"].append(ToolCall(tool_name=ev.get("tool", "unknown"), arguments=ev.get("params") or {},
                                         timestamp=parse_time(ev.get("ts"))))
        elif t == "TOOL_RETURN" and cur is not None:
            # 匹配同名、尚未有返回的最早一次调用
            for tc in cur["calls"]:
                if tc.tool_name == ev.get("tool") and tc.result is None and tc.error is None:
                    tc.result, tc.error = ev.get("output"), ev.get("err_msg")
                    break
        elif t == "BOT_MSG" and cur is not None:
            turns.append(Turn(turn_id=len(turns), user_query=cur["user"], agent_reply=ev.get("text", ""),
                              tool_calls=cur["calls"], request_time=cur["time"]))
            cur = None
    return turns


def _from_turn_table(raw: Dict[str, Any], log: List[str]) -> List[Turn]:
    rows = sorted(raw["turnList"], key=lambda r: r.get("turnNo", 0))
    if rows and rows[0].get("turnNo", 1) > 1:
        log.append(f"历史轮次缺失：首条记录轮次号为 {rows[0]['turnNo']}")
    turns = []
    for r in rows:
        calls = [ToolCall(tool_name=c.get("toolName", "unknown"), arguments=_maybe_json(c.get("toolArgs")) or {},
                          result=_maybe_json(c.get("toolResult")), error=c.get("errMsg"))
                 for c in r.get("toolCalls") or []]
        turns.append(Turn(turn_id=len(turns), user_query=r.get("qry", ""), agent_reply=r.get("resp", ""),
                          tool_calls=calls, request_time=parse_time(r.get("reqTime"))))
    return turns


def _truncate(value: Any, log: List[str]) -> Any:
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        log.append(f"超长字段截断：{len(value)} → {MAX_FIELD_CHARS} 字")
        return value[:MAX_FIELD_CHARS] + f"…[已截断，原长{len(value)}字]"
    if isinstance(value, dict):
        return {k: _truncate(v, log) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, log) for v in value]
    return value


def _dedupe_retries(calls: List[ToolCall], log: List[str]) -> List[ToolCall]:
    """超时重试：失败调用后紧跟同名同参的调用时，只保留最终那次"""
    kept = []
    for i, tc in enumerate(calls):
        nxt = calls[i + 1] if i + 1 < len(calls) else None
        if tc.error and "TIMEOUT" in tc.error and nxt and nxt.tool_name == tc.tool_name and nxt.arguments == tc.arguments:
            log.append(f"去除超时重试的失败调用：{tc.tool_name}")
            continue
        kept.append(tc)
    return kept


def normalize_trace(raw: Dict[str, Any], config: CleanConfig = CleanConfig()) -> AgentTrace:
    """第一层：把三种异构日志格式统一成 AgentTrace"""
    log: List[str] = []
    if "messages" in raw:
        fmt, turns = "openai", _from_openai(raw)
    elif "events" in raw:
        fmt, turns = "event_log", _from_event_log(raw)
    elif "turnList" in raw:
        fmt, turns = "turn_table", _from_turn_table(raw, log)
    elif "turns" in raw:
        fmt, turns = "standard", [Turn(**t) for t in raw["turns"]]
    else:
        raise ValueError(f"无法识别的 trace 格式，字段：{sorted(raw)}")

    if not turns:
        raise ValueError("trace 中没有完整的对话轮次")

    for t in turns:
        if config.dedupe_retry:
            t.tool_calls = _dedupe_retries(t.tool_calls, log)
        if config.truncate:
            for tc in t.tool_calls:
                tc.result = _truncate(tc.result, log)

    trace_id = raw.get("id") or raw.get("session_id") or raw.get("traceId") or raw.get("trace_id") or "unknown"
    skill = raw.get("skill") or raw.get("skill_code") or raw.get("bizSkill") or "unknown"
    log.insert(0, f"格式识别：{fmt}，共 {len(turns)} 轮")
    return AgentTrace(trace_id=trace_id, skill=skill, turns=turns, source_format=fmt, cleaning_log=log)


# ============== 第二层：语义对齐 ==============

def _annotate_status(value: Any, used: set) -> Any:
    """在状态码旁边补一个 *_cn 字段，给 Judge 明确的用户语义"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[k] = _annotate_status(v, used)
            if isinstance(v, str) and v in STATUS_GLOSSARY:
                out[f"{k}_cn"] = STATUS_GLOSSARY[v]
                used.add(v)
        return out
    if isinstance(value, list):
        return [_annotate_status(v, used) for v in value]
    return value


def align_semantics(trace: AgentTrace) -> AgentTrace:
    """第二层：时间锚点 + 状态语义映射"""
    used: set = set()
    for t in trace.turns:
        if not t.request_time:
            # 提问时间缺失时，用本轮第一次工具调用时间兜底
            fallback = next((tc.timestamp for tc in t.tool_calls if tc.timestamp), None)
            t.request_time = fallback
            trace.cleaning_log.append(f"第{t.turn_id}轮缺少提问时间，" + ("用工具调用时间兜底" if fallback else "无法补全"))
        for tc in t.tool_calls:
            tc.result = _annotate_status(tc.result, used)
    if used:
        trace.cleaning_log.append(f"状态语义映射：{sorted(used)}")
    trace.metadata["status_codes"] = sorted(used)
    return trace


# ============== 第三层：屏蔽干扰信号 ==============

def _strip_noise(value: Any, counter: List[int]) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k.startswith("_") or k in NOISE_RESULT_KEYS:
                counter[0] += 1
                continue
            out[k] = _strip_noise(v, counter)
        return out
    if isinstance(value, list):
        return [_strip_noise(v, counter) for v in value]
    if isinstance(value, str):
        for p in NOISE_TEXT_PATTERNS:
            value, n = p.subn("", value)
            counter[0] += n
        return value
    return value


def mask_noise(trace: AgentTrace) -> AgentTrace:
    """第三层：剔除噪声工具调用 + 正则剔除透传的系统提示词"""
    dropped, counter = 0, [0]
    for t in trace.turns:
        before = len(t.tool_calls)
        t.tool_calls = [tc for tc in t.tool_calls if tc.tool_name not in NOISE_TOOL_NAMES]
        dropped += before - len(t.tool_calls)
        for tc in t.tool_calls:
            tc.result = _strip_noise(tc.result, counter)
    if dropped:
        trace.cleaning_log.append(f"剔除噪声工具调用 {dropped} 次")
    if counter[0]:
        trace.cleaning_log.append(f"剔除透传提示词/调试字段 {counter[0]} 处")
    return trace


def preprocess_trace(raw: Dict[str, Any], config: CleanConfig = CleanConfig()) -> AgentTrace:
    """前三层清洗（第四层跨轮隔离在构造证据时按轮执行）"""
    trace = normalize_trace(raw, config)
    if config.semantic_align:
        trace = align_semantics(trace)
    if config.mask_noise:
        trace = mask_noise(trace)
    return trace


# ============== 第四层：跨轮证据隔离 ==============

def _tool_view(tc: ToolCall) -> Dict[str, Any]:
    return {"tool_name": tc.tool_name, "arguments": tc.arguments, "result": tc.result, "error": tc.error}


def isolate_turn_evidence(trace: AgentTrace, turn_index: int, isolate: bool = True) -> Dict[str, Any]:
    """
    第四层：跨轮证据隔离（核心亮点）

    问题：把所有轮次的 tool_calls 都传给 Judge，模型会拿前一轮的工具返回
         （比如上一单的金额）当作当前轮的事实依据，跨轮串味导致误判。
    方案：当前轮保留完整 tool_calls；历史轮只保留 user/assistant 文本。

    isolate=False 用于 Baseline/消融：历史轮也带上全部工具调用。
    """
    if not 0 <= turn_index < len(trace.turns):
        raise ValueError(f"turn_index {turn_index} 超出范围 [0, {len(trace.turns)})")
    cur = trace.turns[turn_index]
    history = []
    for t in trace.turns[:turn_index]:
        h = {"user": t.user_query, "assistant": t.agent_reply}
        if not isolate:
            h["tool_calls"] = [_tool_view(tc) for tc in t.tool_calls]
        history.append(h)
    return {
        "trace_id": trace.trace_id,
        "turn_id": cur.turn_id,
        "skill": trace.skill,
        "current_turn": {
            "user_query": cur.user_query,
            "agent_reply": cur.agent_reply,
            "tool_calls": [_tool_view(tc) for tc in cur.tool_calls],
        },
        "conversation_history": history,
    }


def build_evidence(trace: AgentTrace, turn_index: int, isolate: bool = True, time_anchor: bool = True,
                   glossary: bool = True, integrity: bool = True) -> Dict[str, Any]:
    """组装 Judge 的完整输入证据（第四层隔离 + 时间锚点 + 状态语义表 + 日志完整性标记）"""
    ev = isolate_turn_evidence(trace, turn_index, isolate=isolate)
    if time_anchor:
        ev["request_time"] = trace.turns[turn_index].request_time
    if glossary:
        codes = trace.metadata.get("status_codes", [])
        ev["status_glossary"] = {c: STATUS_GLOSSARY[c] for c in codes}
    if integrity:
        marks = [m for m in trace.cleaning_log if m.startswith("历史轮次缺失")]
        for tc in trace.turns[turn_index].tool_calls:
            if tc.error and "LOG_TRUNCATED" in tc.error:
                marks.append(f"工具 {tc.tool_name} 的返回结果未落日志")
        ev["log_integrity"] = marks
    return ev


# ============== IO ==============

def load_raw_traces(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_traces(path: str, config: CleanConfig = CleanConfig()) -> List[AgentTrace]:
    return [preprocess_trace(copy.deepcopy(r), config) for r in load_raw_traces(path)]


def save_traces(traces: List[AgentTrace], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(t.model_dump_json() + "\n")
