"""OpenAI 兼容协议的 LLM 适配器。

覆盖所有 OpenAI-compatible 端点（OpenAI 本身、DeepSeek、火山方舟等），只需换
``base_url`` / ``api_key`` / ``model``。

要点：
- ``openai`` SDK 懒加载，且 ``client`` 可注入——测试里传入桩对象即可离线验证
  请求构造与响应归一化，绝不触网。
- 把 SDK 返回的对象统一归一化为 :class:`~llm.base.LLMResponse`，tool_calls 里的
  arguments（JSON 字符串）在此解析成 dict。
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall

__all__ = ["OpenAIClient"]


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """把 provider 返回的 arguments 归一化为 dict，非法时降级为空 dict。"""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class OpenAIClient(LLMClient):
    """基于 ``openai.AsyncOpenAI`` 的适配器。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        if client is not None:
            # 注入模式：测试用桩对象，跳过 SDK 构造，不触网
            self._client = client
        else:
            # 先校验参数再导入 SDK：未装 openai 时也能给出清晰的缺 Key 报错
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "api_key is required (pass api_key= or set OPENAI_API_KEY)"
                )

            from openai import AsyncOpenAI  # 懒加载

            self._client = AsyncOpenAI(
                api_key=resolved_key, base_url=base_url, timeout=timeout
            )

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": list(messages),
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
        }
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = kwargs.pop("tool_choice", "auto")
        request.update(kwargs)

        response = await self._client.chat.completions.create(**request)
        return self._normalize(response)

    def _normalize(self, response: Any) -> LLMResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return LLMResponse(content=None, raw=response)

        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)

        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            tool_calls.append(
                ToolCall(
                    id=str(getattr(tc, "id", "") or f"call_{len(tool_calls)}"),
                    name=str(getattr(fn, "name", "") or ""),
                    arguments=_parse_arguments(getattr(fn, "arguments", None)),
                )
            )

        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(raw_usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(raw_usage, "total_tokens", 0) or 0,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(response, "model", None),
            usage=usage,
            raw=response,
        )
