"""单个会话的上下文消息流。

Context 内部保留**全量**历史（不破坏原始记录），压缩只发生在构建发给 LLM 的
视图时——这样截断是可逆的，测试也能同时观察"全量历史"与"压缩后视图"。
"""

from __future__ import annotations

from typing import Any

from agent.schema import (
    AnyMessage,
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from llm.base import ToolCall

from .compression import CompressionResult, compress_messages, estimate_tokens_total

__all__ = ["Context"]


class Context:
    """有序消息流 + system prompt + 压缩配置。"""

    def __init__(
        self,
        system_prompt: str | None = None,
        *,
        max_history_turns: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._messages: list[AnyMessage] = []
        if system_prompt:
            self._messages.append(SystemMessage(content=system_prompt))
        self.max_history_turns = max_history_turns
        self.max_tokens = max_tokens

    # ---- 写入 ----
    def add(self, message: AnyMessage) -> None:
        self._messages.append(message)

    def add_user(self, content: str) -> UserMessage:
        msg = UserMessage(content=content)
        self.add(msg)
        return msg

    def add_system(self, content: str) -> SystemMessage:
        msg = SystemMessage(content=content)
        self.add(msg)
        return msg

    def add_assistant(
        self, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> AssistantMessage:
        msg = AssistantMessage(content=content, tool_calls=tool_calls or [])
        self.add(msg)
        return msg

    def add_tool(
        self, content: str, tool_call_id: str, name: str | None = None
    ) -> ToolMessage:
        msg = ToolMessage(content=content, tool_call_id=tool_call_id, name=name)
        self.add(msg)
        return msg

    # ---- 读取 ----
    @property
    def messages(self) -> list[AnyMessage]:
        """全量历史（副本）。"""
        return list(self._messages)

    @property
    def system_prompt(self) -> str | None:
        for msg in self._messages:
            if isinstance(msg, SystemMessage):
                return msg.content
        return None

    def __len__(self) -> int:
        return len(self._messages)

    def estimated_tokens(self) -> int:
        return estimate_tokens_total(self._messages)

    # ---- 压缩视图 ----
    def compress(self) -> CompressionResult:
        """对全量历史应用压缩策略，返回压缩结果（不改内部状态）。"""
        return compress_messages(
            self._messages,
            max_history_turns=self.max_history_turns,
            max_tokens=self.max_tokens,
        )

    def compressed_messages(self) -> list[AnyMessage]:
        return self.compress().messages

    def build_messages(self) -> list[dict[str, Any]]:
        """构建发给 LLM 的 OpenAI wire-format 消息列表（压缩后视图）。"""
        return [m.to_dict() for m in self.compressed_messages()]
