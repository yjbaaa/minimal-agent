"""LLM 客户端层：接口、OpenAI 兼容适配器、离线假客户端。"""

from __future__ import annotations

from .base import LLMClient, LLMResponse, Message, ToolCall
from .fake import FakeLLMClient, final, tool_call, tool_calls

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "ToolCall",
    "FakeLLMClient",
    "final",
    "tool_call",
    "tool_calls",
    "OpenAIClient",
]


def __getattr__(name: str):
    # 懒加载 OpenAIClient，避免未安装 openai SDK 时 import llm 就失败
    if name == "OpenAIClient":
        from .openai_client import OpenAIClient

        return OpenAIClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
