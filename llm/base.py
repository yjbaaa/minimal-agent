"""LLM 客户端的抽象接口与归一化响应模型。

设计边界（阶段 2）：
- ``complete`` 接收 OpenAI wire-format 的消息字典列表（``{"role","content",...}``），
  这样本层不耦合阶段 3 才会引入的结构化 Message 类型；上下文层负责拼这些 dict。
- 所有 provider 的原始响应都归一化成 :class:`LLMResponse`，上游（解析器/循环）
  只认这一个形状，换 provider 只需换适配器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["Message", "ToolCall", "LLMResponse", "LLMClient"]

# OpenAI wire-format 的一条消息
Message = dict[str, Any]


class ToolCall(BaseModel):
    """一次工具调用的归一化表示（provider 无关）。"""

    id: str = Field(..., description="调用 id，用于把 tool 结果回填给对应调用")
    name: str = Field(..., description="工具名")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="已解析为 dict 的调用参数"
    )


class LLMResponse(BaseModel):
    """归一化后的 LLM 响应。"""

    model_config = {"arbitrary_types_allowed": True}

    content: str | None = Field(default=None, description="助手文本（可能含思考过程）")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="原生 function-calling 解析出的工具调用"
    )
    finish_reason: str | None = Field(default=None, description="结束原因，如 stop/tool_calls")
    model: str | None = Field(default=None, description="实际使用的模型名")
    usage: dict[str, int] | None = Field(default=None, description="token 用量")
    raw: Any = Field(default=None, description="provider 原始响应对象，便于调试")


class LLMClient(ABC):
    """LLM 客户端接口。具体 provider 实现 :meth:`complete`。"""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送一次对话补全请求，返回归一化的 :class:`LLMResponse`。"""
        raise NotImplementedError
