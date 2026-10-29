"""
Pydantic 数据模型

数据流中的四类对象：
1. Trace 侧：ToolCall / Turn / AgentTrace —— 清洗后统一的会话结构
2. Judge 侧：LayerJudgment / JudgeOutput —— 约束 LLM 输出的 JSON Schema
3. 结果侧：LayerResult / CaseResult —— 短路聚合后的单条评测结果
4. 标注与归因：GroundTruth / AttributionResult
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# 六层漏斗：前 5 层串行短路，RED 并联恒评（淘宝文章 3.4 节）
LAYERS = ["CTX", "OOSR", "OOC", "TCC", "CNT", "RED"]
SERIAL_LAYERS = ["CTX", "OOSR", "OOC", "TCC", "CNT"]

# 单层取值口径：1 通过 / 0 不通过 / N/A（CTX 层的"剔除"，或被短路跳过）
Verdict = Literal["1", "0", "N/A"]
# 样本最终标签：excluded = CTX 证据不足被剔除，不进入通过率分母
FinalLabel = Literal["pass", "fail", "excluded"]


# ============== Trace 侧 ==============

class ToolCall(BaseModel):
    """一次工具调用"""

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None


class Turn(BaseModel):
    """一轮对话：用户提问 + 本轮工具调用 + Agent 回复"""

    turn_id: int = Field(..., ge=0)
    user_query: str
    agent_reply: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    request_time: Optional[str] = Field(None, description="用户提问时间（ISO 格式），时间锚点")


class AgentTrace(BaseModel):
    """清洗后的完整多轮 Trace"""

    trace_id: str
    skill: str = Field("unknown", description="承接本会话的 skill 标识")
    turns: List[Turn] = Field(..., min_length=1)
    source_format: str = Field("unknown", description="原始日志格式（openai/event_log/turn_table）")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cleaning_log: List[str] = Field(default_factory=list, description="清洗过程记录，供数据层归因使用")


# ============== Judge 侧（结构化输出约束） ==============

class LayerJudgment(BaseModel):
    """Judge 对单层的输出：先引用证据，再给判定（反向验证指令）"""

    evidence_quote: str = Field("", description="从输入中摘录的原文证据")
    verdict: str = Field(..., description="原始判定值，归一化前可能是非标值")
    reason: str = ""


class JudgeOutput(BaseModel):
    """Judge 输出的 JSON Schema：六层逐维度输出，避免维度间串味"""

    CTX: LayerJudgment
    OOSR: LayerJudgment
    OOC: LayerJudgment
    TCC: LayerJudgment
    CNT: LayerJudgment
    RED: LayerJudgment


# ============== 结果侧 ==============

class LayerResult(BaseModel):
    """单层最终结果（已归一化、已应用短路）"""

    layer: str
    verdict: Verdict
    reason: str = ""
    evidence_quote: str = ""
    source: Literal["judge", "short_circuit", "parse_failed"] = "judge"


class CaseResult(BaseModel):
    """单条样本（一个 trace 的目标轮）的评测结果"""

    trace_id: str
    turn_id: int
    config: str
    layers: Dict[str, LayerResult] = Field(default_factory=dict)
    final_label: Optional[FinalLabel] = Field(None, description="解析失败时为 None")
    failed_layer: Optional[str] = Field(None, description="判 0 的层（串行短路位置或 RED）")
    short_circuit_at: Optional[str] = None
    parse_status: Literal["ok", "normalized", "failed"] = "ok"
    normalize_notes: List[str] = Field(default_factory=list)
    raw_output: str = ""
    score: Optional[float] = Field(None, description="仅基线 GEval 使用：0~1 连续分")


# ============== 标注与归因 ==============

class GroundTruth(BaseModel):
    """样本标签

    label_source 严格区分：
    - construction：构造数据时按设计意图写入的期望标签（不是人工标注）
    - human：人工逐条复核后的标签（一致率报告应基于此）
    """

    trace_id: str
    turn_id: int
    label: FinalLabel
    failed_layer: Optional[str] = None
    anomaly_type: Optional[str] = None
    reason: str = ""
    label_source: Literal["construction", "human"] = "construction"
    annotator: Optional[str] = None
    annotated_at: Optional[str] = None


class AttributionResult(BaseModel):
    """四层归因结果"""

    trace_id: str
    turn_id: int
    judge_label: Optional[FinalLabel]
    human_label: FinalLabel
    data_issues: List[str] = Field(default_factory=list)
    prompt_issues: List[str] = Field(default_factory=list)
    rubric_issues: List[str] = Field(default_factory=list)
    agent_issue: Optional[str] = None
    primary_cause: Literal["data", "prompt", "rubric", "agent", "unresolved"]
