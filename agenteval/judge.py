"""
Judge：LLM 调用 + 结构化输出的三层防线（参考淘宝文章 3.5 节第 6 点）

    第一层 Prompt 约束    rubrics.py 中强制 JSON Schema、枚举取值
    第二层 输出校验器     extract_json + Pydantic(JudgeOutput) 校验，拦截非标输出
    第三层 标签归一化     normalize_verdict：把"通过/pass/剔除/NA/空值"等非标值映射到 1/0/N/A

LLM 调用复用 DeepEval 的模型抽象（DeepEvalBaseLLM），接任意 OpenAI 兼容接口
（OpenAI / DeepSeek / 通义千问 / 本地 vLLM 等），这样同一个模型对象既能给自研 Judge 用，
也能直接传给 DeepEval 的 GEval。
"""

import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from deepeval.models import DeepEvalBaseLLM
from pydantic import ValidationError

from agenteval.models import LAYERS, JudgeOutput

# ============== LLM 封装（复用 DeepEval 模型抽象） ==============


class JudgeLLM(DeepEvalBaseLLM):
    """OpenAI 兼容接口的 Judge 模型"""

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None,
                 temperature: float = 0.1, json_mode: bool = True, max_retries: int = 3, timeout: float = 120):
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.timeout = timeout
        super().__init__(model)

    def load_model(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def generate(self, prompt: str, schema=None, **kwargs) -> str:
        """返回原始文本；失败按指数退避重试。schema 参数只为兼容 DeepEval 调用约定，解析在本模块完成"""
        params = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                  "temperature": self.temperature}
        if self.json_mode:
            params["response_format"] = {"type": "json_object"}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.model.chat.completions.create(**params)
                return resp.choices[0].message.content or ""
            except Exception as e:  # 网络、限流、超时都重试
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败（重试 {self.max_retries} 次）：{last_err}")

    async def a_generate(self, prompt: str, schema=None, **kwargs) -> str:
        return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self.model_name

    def supports_temperature(self) -> bool:
        return True


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def judge_from_env(temperature: float = 0.1) -> JudgeLLM:
    """
    从环境变量 / .env 创建 Judge：
      JUDGE_API_KEY（或 OPENAI_API_KEY）、JUDGE_MODEL、JUDGE_BASE_URL（可选）、JUDGE_JSON_MODE（默认 1）
    没有配置时直接报错——本项目不允许用 Mock 代替真实 Judge 做实验。
    """
    load_env()
    key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("JUDGE_MODEL")
    if not key or not model:
        raise RuntimeError("未配置 Judge 模型：请在 .env 中设置 JUDGE_API_KEY 与 JUDGE_MODEL（可选 JUDGE_BASE_URL）")
    return JudgeLLM(model=model, api_key=key, base_url=os.getenv("JUDGE_BASE_URL") or None,
                    temperature=temperature, json_mode=os.getenv("JUDGE_JSON_MODE", "1") == "1")


# ============== 第二层：输出校验 ==============

def extract_json(text: str) -> Optional[dict]:
    """从模型输出中提取 JSON 对象（兼容 ```json 代码块、前后多余文字）"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ============== 第三层：标签归一化 ==============

_PASS = {"1", "1.0", "pass", "通过", "合格", "true", "yes", "是"}
_FAIL = {"0", "0.0", "fail", "不通过", "不合格", "false", "no", "否"}
_NA = {"n/a", "na", "剔除", "不适用", "不可评", "skip", "none", "null", "", "-"}


def normalize_verdict(raw, layer: str) -> Tuple[Optional[str], Optional[str]]:
    """
    归一化单层判定，返回 (verdict, note)。verdict 为 None 表示无法识别（解析失败）。

    取值口径（淘宝文章 3.4 节）：
      - CTX 只有 1 / 剔除(N/A)；CTX 给出 0 视为"证据不足"→ 剔除
      - RED 恒评，只有 1 / 0；RED 给出 N/A 无法使用 → 解析失败
      - OOSR/OOC/TCC/CNT：1 / 0 / N/A（N/A 只应出现在被短路时，由聚合层处理）
    """
    s = str(raw).strip().lower() if raw is not None else ""
    if s in _PASS:
        v = "1"
    elif s in _FAIL:
        v = "0"
    elif s in _NA:
        v = "N/A"
    else:
        return None, f"{layer}: 无法识别的取值 {raw!r}"

    note = None if str(raw).strip() in {"1", "0", "N/A", "剔除"} else f"{layer}: {raw!r} → {v}"
    if layer == "CTX" and v == "0":
        v, note = "N/A", f"CTX: '0' 按证据不足处理为剔除"
    if layer == "RED" and v == "N/A":
        return None, "RED: 并联层不允许 N/A"
    return v, note


def parse_judge_output(text: str) -> Tuple[Optional[Dict[str, dict]], str, List[str]]:
    """
    解析 Judge 输出。返回 (layers, status, notes)：
      layers = {layer: {"verdict", "reason", "evidence_quote"}}；失败时为 None
      status = ok / normalized / failed
    """
    obj = extract_json(text)
    if obj is None:
        return None, "failed", ["输出中找不到合法 JSON"]
    # 兼容大小写不一致的键名
    obj = {str(k).upper(): v for k, v in obj.items()}
    try:
        out = JudgeOutput.model_validate(obj)
    except ValidationError as e:
        return None, "failed", [f"Schema 校验失败：{e.errors()[0]['loc']} {e.errors()[0]['msg']}"]

    layers, notes = {}, []
    for layer in LAYERS:
        j = getattr(out, layer)
        v, note = normalize_verdict(j.verdict, layer)
        if note:
            notes.append(note)
        if v is None:
            return None, "failed", notes
        layers[layer] = {"verdict": v, "reason": j.reason, "evidence_quote": j.evidence_quote}
    return layers, ("normalized" if notes else "ok"), notes
