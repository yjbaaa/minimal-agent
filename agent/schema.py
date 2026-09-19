"""结构化消息类型。

标准消息流：System / User / Assistant / Tool（对应项目文档 (3)）。每种消息都能
``to_dict()`` 成 OpenAI wire-format，供上下文层拼装请求、供解析器/循环消费。

复用 :class:`llm.base.ToolCall` 作为 AssistantMessage 的工具调用载体，避免重复定义。
"""

from __future__ import annotations

import json
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from llm.base import ToolCall

__all__ = [
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "AnyMessage",
]


class BaseMessage(BaseModel):
    """所有消息的基类。"""

    role: str

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def text(self) -> str:
        """用于 token 估算的文本近似（角色 + 正文 + 工具调用参数）。"""
        return self.content or ""  # type: ignore[attr-defined]


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"
    content: str = Field(..., description="系统提示词")

    def to_dict(self) -> dict[str, Any]:
        return {"role": "system", "content": self.content}


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"
    content: str = Field(..., description="用户输入")

    def to_dict(self) -> dict[str, Any]:
        return {"role": "user", "content": self.content}


class AssistantMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    content: str | None = Field(default=None, description="助手文本/思考，可为空")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="本轮发起的工具调用")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in self.tool_calls
            ]
        return payload

    def text(self) -> str:
        base = self.content or ""
        for tc in self.tool_calls:
            base += " " + tc.name + " " + json.dumps(tc.arguments, ensure_ascii=False)
        return base.strip()


class ToolMessage(BaseMessage):
    role: Literal["tool"] = "tool"
    content: str = Field(..., description="工具执行结果（或标准错误文本）")
    tool_call_id: str = Field(..., description="对应 AssistantMessage 里的 tool_call id")
    name: str | None = Field(default=None, description="工具名，便于调试/日志")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


AnyMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


# ---------- ReAct 循环的 Trace / 结果模型 ----------


class ToolCallTrace(BaseModel):
    """一次工具调用的执行轨迹。"""

    tool_call_id: str = Field(..., description="对应 AssistantMessage 里的 tool_call id")
    name: str = Field(..., description="工具名")
    arguments: dict[str, Any] = Field(default_factory=dict, description="调用参数")
    ok: bool = Field(..., description="工具是否执行成功")
    output: Any = Field(default=None, description="成功时的返回值")
    error: str | None = Field(default=None, description="失败时的标准错误文本")
    elapsed_ms: float = Field(default=0.0, description="执行耗时（毫秒）")


class StepTrace(BaseModel):
    """一次循环迭代（一次 LLM 调用 + 其触发的工具执行）的轨迹。"""

    step: int = Field(..., description="从 1 开始的迭代序号")
    thought: str | None = Field(default=None, description="本轮思考/推理文本")
    tool_calls: list[ToolCallTrace] = Field(default_factory=list, description="本轮执行的工具调用")
    is_final: bool = Field(default=False, description="本轮是否产出最终答案")
    final_answer: str | None = Field(default=None, description="最终答案（is_final 时有值）")


class AgentResult(BaseModel):
    """Agent 处理一次用户输入的完整结果。"""

    model_config = {"arbitrary_types_allowed": True}

    session_id: str | None = Field(default=None, description="所属会话 id")
    final_answer: str | None = Field(default=None, description="返回给用户的答案")
    steps: list[StepTrace] = Field(default_factory=list, description="全部迭代轨迹")
    stopped_reason: Literal["final_answer", "max_steps"] = Field(
        ..., description="结束原因：正常给出答案 / 触发最大步数上限"
    )

    @property
    def num_steps(self) -> int:
        return len(self.steps)

