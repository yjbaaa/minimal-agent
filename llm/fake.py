"""测试/离线用的 Fake LLM 客户端。

按脚本回放预设响应，并记录每次请求，供断言"上下文里带了什么、工具 schema 有没有
传进去、多轮是否串联"。绝不触网、绝不依赖真实 provider（见项目锁定决策）。

脚本项可以是：
- :class:`LLMResponse` 实例；
- dict（作为 ``LLMResponse(**dict)`` 的关键字参数）；
- callable ``(messages, tools) -> LLMResponse | dict``，用于按输入动态决定。

配合模块级的 :func:`final` / :func:`tool_call` / :func:`tool_calls` 便捷构造器使用。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Union

from .base import LLMClient, LLMResponse, Message, ToolCall

__all__ = [
    "FakeLLMClient",
    "final",
    "tool_call",
    "tool_calls",
]

ScriptItem = Union[
    LLMResponse,
    dict[str, Any],
    Callable[[Sequence[Message], Sequence[dict[str, Any]] | None], Union[LLMResponse, dict[str, Any]]],
]


def final(content: str, **kwargs: Any) -> LLMResponse:
    """构造一个"最终回复"响应（无工具调用）。"""
    return LLMResponse(content=content, finish_reason="stop", **kwargs)


def tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    content: str | None = None,
    id: str | None = None,
) -> LLMResponse:
    """构造一个含单个原生工具调用的响应。"""
    return LLMResponse(
        content=content,
        tool_calls=[ToolCall(id=id or f"call_{name}_0", name=name, arguments=arguments or {})],
        finish_reason="tool_calls",
    )


def tool_calls(calls: list[ToolCall], *, content: str | None = None) -> LLMResponse:
    """构造一个含多个原生工具调用的响应。"""
    return LLMResponse(content=content, tool_calls=calls, finish_reason="tool_calls")


class FakeLLMClient(LLMClient):
    """回放脚本响应的假客户端。"""

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        model: str = "fake-model",
    ) -> None:
        self._script: list[ScriptItem] = list(script or [])
        self.model = model
        self.requests: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def last_request(self) -> dict[str, Any]:
        return self.requests[-1]

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.requests.append(
            {"messages": list(messages), "tools": list(tools) if tools else None, "kwargs": kwargs}
        )

        if self._script:
            item = self._script.pop(0)
        else:
            # 脚本耗尽：返回一个终止性最终回复，避免上层循环卡死
            return final("(no more scripted responses)")

        if callable(item):
            item = item(messages, tools)
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(**item)
