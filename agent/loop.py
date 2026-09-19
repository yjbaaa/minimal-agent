"""ReAct 循环运行时：把 LLM、解析器、工具注册表、会话上下文串成 Agent。

单轮时序（对应项目文档 (1)）：
1. 用户输入入上下文，拼装 messages（system + history）+ tools schema；
2. 调用 LLM，拿到归一化响应；
3. ResponseParser 解析出 thought / tool_calls / final_answer；
4. 若需调工具：分发执行，记录 trace，把结果（或标准错误文本）作为 ToolMessage
   回灌上下文——错误也回灌，让模型有机会自我修正；
5. 循环，直到产出最终答案或达到 ``max_steps`` 上限（优雅退出，不 crash）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from llm.base import LLMClient
from session.manager import Session

from .parser import ResponseParser
from .schema import AgentResult, StepTrace, ToolCallTrace

__all__ = ["Agent", "render_trace", "render_step"]

Tracer = Callable[[str], None]


class Agent:
    """最小可用的 ReAct Agent 运行时。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        parser: ResponseParser | None = None,
        max_steps: int = 5,
        tracer: Tracer | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm = llm
        self.parser = parser or ResponseParser()
        self.max_steps = max_steps
        self.tracer = tracer

    async def run(self, user_input: str, session: Session) -> AgentResult:
        """处理一次用户输入，返回带完整 trace 的 :class:`AgentResult`。"""
        session.context.add_user(user_input)
        steps: list[StepTrace] = []

        tools = session.registry.schemas() or None

        for step_index in range(1, self.max_steps + 1):
            messages = session.context.build_messages()
            response = await self.llm.complete(messages, tools=tools)
            parsed = self.parser.parse(response)

            # 助手消息入上下文（工具调用轮存 thought，收尾轮存最终答案）
            assistant_content = parsed.final_answer if parsed.is_final else parsed.thought
            session.context.add_assistant(
                content=assistant_content, tool_calls=parsed.tool_calls
            )

            if parsed.is_final:
                step = StepTrace(
                    step=step_index,
                    thought=parsed.thought,
                    tool_calls=[],
                    is_final=True,
                    final_answer=parsed.final_answer,
                )
                steps.append(step)
                self._emit(step)
                return AgentResult(
                    session_id=session.session_id,
                    final_answer=parsed.final_answer,
                    steps=steps,
                    stopped_reason="final_answer",
                )

            # 执行本轮所有工具调用，结果/错误回灌上下文
            call_traces: list[ToolCallTrace] = []
            for call in parsed.tool_calls:
                result = await session.registry.call(call.name, call.arguments)
                session.context.add_tool(
                    content=result.to_text(), tool_call_id=call.id, name=call.name
                )
                call_traces.append(
                    ToolCallTrace(
                        tool_call_id=call.id,
                        name=call.name,
                        arguments=call.arguments,
                        ok=result.ok,
                        output=result.output,
                        error=result.error,
                        elapsed_ms=result.elapsed_ms,
                    )
                )

            step = StepTrace(
                step=step_index,
                thought=parsed.thought,
                tool_calls=call_traces,
                is_final=False,
                final_answer=None,
            )
            steps.append(step)
            self._emit(step)

        # 触发上限：优雅退出，给一个可读的兜底答案
        fallback = (
            f"已达到最大迭代步数（max_steps={self.max_steps}），停止推理。"
            "请重试或缩小问题范围。"
        )
        return AgentResult(
            session_id=session.session_id,
            final_answer=fallback,
            steps=steps,
            stopped_reason="max_steps",
        )

    def _emit(self, step: StepTrace) -> None:
        if self.tracer is not None:
            self.tracer(render_step(step))


def render_step(step: StepTrace) -> str:
    """把一步 trace 渲染成人类可读文本（Step / Thought / Tool Call / Time / Output）。"""
    lines = [f"── Step {step.step} " + "─" * 30]
    if step.thought:
        lines.append(f"Thought: {step.thought}")

    for call in step.tool_calls:
        args = json.dumps(call.arguments, ensure_ascii=False)
        lines.append(f"Tool Call: {call.name}({args})")
        status = "ok" if call.ok else "error"
        lines.append(f"Execution Time: {call.elapsed_ms:.3f} ms [{status}]")
        if call.ok:
            output = call.output
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, default=str)
            lines.append(f"Tool Output: {output}")
        else:
            lines.append(f"Tool Error: {call.error}")

    if step.is_final:
        lines.append(f"Final Answer: {step.final_answer}")
    return "\n".join(lines)


def render_trace(result: AgentResult) -> str:
    """把整个 AgentResult 的 trace 渲染成一段文本。"""
    parts = [render_step(step) for step in result.steps]
    parts.append(f"══ stopped_reason: {result.stopped_reason} | steps: {result.num_steps} ══")
    return "\n".join(parts)
