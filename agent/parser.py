"""把 LLM 响应解析成"要么调工具、要么给最终答案"的结构化结果。

对应锁定决策 #2：provider 原生 ``tool_calls`` 为主，``ResponseParser`` 再回退解析
``content`` 里的 JSON（ReAct 协议），让不支持 function-calling 的模型也能驱动循环。

回退时识别的规范 JSON（system prompt 会据此约束模型输出）::

    {"thought": "...", "action": "工具名", "action_input": {...}}   # 调工具
    {"thought": "...", "final_answer": "..."}                        # 给最终答案

同时容忍 ``{"tool_calls": [{"name", "arguments"}]}`` 这种多调用形式，以及把 JSON
包在 ```json 围栏或夹在散文里的情况。解析不出可识别结构时，优雅降级为最终答案
（把原始文本当作回复），绝不抛异常打断循环。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from llm.base import LLMResponse, ToolCall

__all__ = ["ParsedResponse", "ResponseParser"]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_NAME_KEYS = ("action", "tool", "name", "function")
_ARGS_KEYS = ("action_input", "arguments", "args", "input")
_FINAL_KEYS = ("final_answer", "answer", "output", "response")
_THOUGHT_KEYS = ("thought", "reasoning", "thinking")


class ParsedResponse(BaseModel):
    """解析后的单轮 LLM 输出。"""

    model_config = {"arbitrary_types_allowed": True}

    thought: str | None = Field(default=None, description="思考/推理文本")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="要执行的工具调用")
    final_answer: str | None = Field(default=None, description="给用户的最终回复")

    @property
    def is_final(self) -> bool:
        """没有工具调用即视为最终答案。"""
        return not self.tool_calls


class ResponseParser:
    """LLMResponse -> ParsedResponse 的解析器（无状态）。"""

    def parse(self, response: LLMResponse) -> ParsedResponse:
        # 1) 原生 function-calling 优先
        if response.tool_calls:
            return ParsedResponse(
                thought=response.content,
                tool_calls=list(response.tool_calls),
                final_answer=None,
            )

        content = response.content or ""

        # 2) 无原生调用 -> 尝试从 content 里解析 ReAct JSON
        data = self._extract_json(content)
        if isinstance(data, dict):
            parsed = self._interpret_json(data)
            if parsed is not None:
                return parsed

        # 3) 兜底：把纯文本当作最终答案
        return ParsedResponse(thought=None, tool_calls=[], final_answer=content)

    # ---- JSON 提取 ----
    def _extract_json(self, text: str) -> Any:
        if not text.strip():
            return None

        candidates: list[str] = []
        fence = _FENCE_RE.search(text)
        if fence:
            candidates.append(fence.group(1).strip())
        candidates.append(text)

        for candidate in candidates:
            obj = self._raw_decode_first_object(candidate)
            if obj is not None:
                return obj
        return None

    def _raw_decode_first_object(self, text: str) -> Any:
        """从文本中第一个 '{' 起尝试解析出一个 JSON 对象。"""
        decoder = json.JSONDecoder()
        start = text.find("{")
        while start != -1:
            try:
                obj, _ = decoder.raw_decode(text[start:])
                return obj
            except (json.JSONDecodeError, ValueError):
                start = text.find("{", start + 1)
        return None

    # ---- JSON 语义解释 ----
    def _interpret_json(self, data: dict[str, Any]) -> ParsedResponse | None:
        thought = self._first_str(data, _THOUGHT_KEYS)

        # 多调用形式：{"tool_calls": [...]}
        raw_calls = data.get("tool_calls")
        if isinstance(raw_calls, list) and raw_calls:
            calls = [c for c in (self._to_tool_call(x) for x in raw_calls) if c is not None]
            if calls:
                return ParsedResponse(thought=thought, tool_calls=calls, final_answer=None)

        # 单调用形式：{"action": ..., "action_input": {...}}
        name = self._first_str(data, _NAME_KEYS)
        if name:
            arguments = self._coerce_arguments(self._first_present(data, _ARGS_KEYS))
            call = ToolCall(id=self._new_id(), name=name, arguments=arguments)
            return ParsedResponse(thought=thought, tool_calls=[call], final_answer=None)

        # 最终答案形式：{"final_answer": ...}
        if any(k in data for k in _FINAL_KEYS):
            answer = self._first_present(data, _FINAL_KEYS)
            answer_text = answer if isinstance(answer, str) else json.dumps(
                answer, ensure_ascii=False
            )
            return ParsedResponse(thought=thought, tool_calls=[], final_answer=answer_text)

        # 认不出的 JSON 结构 -> 交给上层降级为纯文本最终答案
        return None

    def _to_tool_call(self, item: Any) -> ToolCall | None:
        if not isinstance(item, dict):
            return None
        # 兼容 OpenAI 形状 {"id","function":{"name","arguments"}}
        inner = item.get("function") if isinstance(item.get("function"), dict) else item
        name = self._first_str(inner, _NAME_KEYS) or self._first_str(item, _NAME_KEYS)
        if not name:
            return None
        arguments = self._coerce_arguments(
            self._first_present(inner, _ARGS_KEYS)
            if self._first_present(inner, _ARGS_KEYS) is not None
            else self._first_present(item, _ARGS_KEYS)
        )
        call_id = item.get("id") or inner.get("id") or self._new_id()
        return ToolCall(id=str(call_id), name=name, arguments=arguments)

    # ---- 小工具 ----
    @staticmethod
    def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]
        return None

    @staticmethod
    def _first_str(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        value = ResponseParser._first_present(data, keys)
        return str(value) if value is not None else None

    @staticmethod
    def _coerce_arguments(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _new_id() -> str:
        return f"call_{uuid.uuid4().hex[:8]}"
