"""Agent 运行时层。

阶段 2：响应解析器。阶段 3：结构化消息类型。阶段 4：ReAct 循环 Agent 与 trace。
"""

from __future__ import annotations

from .loop import Agent, render_step, render_trace
from .parser import ParsedResponse, ResponseParser
from .schema import (
    AgentResult,
    AnyMessage,
    AssistantMessage,
    StepTrace,
    SystemMessage,
    ToolCallTrace,
    ToolMessage,
    UserMessage,
)

__all__ = [
    "Agent",
    "render_step",
    "render_trace",
    "ParsedResponse",
    "ResponseParser",
    "AgentResult",
    "StepTrace",
    "ToolCallTrace",
    "AnyMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
]
