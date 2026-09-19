"""ReAct 循环测试（阶段 4）：全用 FakeLLMClient 脚本化，不碰真实 LLM。

覆盖：纯文字单轮、单工具调用链、多轮追问串联上下文、工具报错后自我修正、
死循环触发 max_steps 优雅退出、trace 输出、上下文消息顺序、tools schema 传递。
"""

from __future__ import annotations

import asyncio
import json

from agent.loop import Agent, render_trace
from llm.fake import FakeLLMClient, final, tool_call
from session.manager import SessionManager


def _run(agent, text, session):
    return asyncio.run(agent.run(text, session))


def _flatten_messages(messages):
    """把 wire-format 消息列表拼成一段可断言的文本。"""
    return json.dumps(messages, ensure_ascii=False)


# ---------- 纯文字单轮 ----------

def test_pure_text_single_turn():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient([final("你好，我是助手")])
    agent = Agent(fake, max_steps=5)

    result = _run(agent, "你好", session)
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "你好，我是助手"
    assert result.num_steps == 1
    assert result.steps[0].is_final is True
    assert result.steps[0].tool_calls == []


# ---------- 单工具调用链 ----------

def test_single_tool_call_chain():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient(
        [
            tool_call("calculator", {"expression": "1+1"}, content="我来算"),
            final("答案是 2"),
        ]
    )
    agent = Agent(fake, max_steps=5)

    result = _run(agent, "1+1 等于几", session)
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "答案是 2"
    assert result.num_steps == 2

    # 第一步执行了 calculator，成功，输出 "2"
    call = result.steps[0].tool_calls[0]
    assert call.name == "calculator"
    assert call.ok is True
    assert call.output == "2"
    assert call.elapsed_ms >= 0

    # tools schema 确实传给了 LLM
    assert len(fake.requests[0]["tools"]) == 3


def test_context_message_ordering():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient(
        [tool_call("calculator", {"expression": "2*3"}), final("6")]
    )
    agent = Agent(fake)
    _run(agent, "2*3", session)

    roles = [m.role for m in session.context.messages]
    # user -> assistant(带 tool_calls) -> tool -> assistant(最终)
    assert roles == ["user", "assistant", "tool", "assistant"]

    # 第二次 LLM 请求里，assistant(tool_calls) 必须在 tool 消息之前
    wire = fake.requests[1]["messages"]
    wire_roles = [m["role"] for m in wire]
    assert wire_roles.index("assistant") < wire_roles.index("tool")
    assert wire[wire_roles.index("assistant")]["tool_calls"][0]["function"]["name"] == "calculator"


# ---------- 多轮追问串联上下文 ----------

def test_multi_turn_followup_chains_context():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient(
        [
            tool_call("calculator", {"expression": "2+3"}),
            final("结果是 5"),
            final("上一轮算的是 5"),  # 第二轮：纯追问
        ]
    )
    agent = Agent(fake)

    r1 = _run(agent, "2+3 等于几", session)
    assert r1.final_answer == "结果是 5"

    r2 = _run(agent, "那上一轮结果是多少", session)
    assert r2.final_answer == "上一轮算的是 5"
    assert r2.num_steps == 1  # 第二轮无需工具

    # 第二轮 LLM 请求应包含第一轮完整历史（user/工具结果/assistant）
    last_request = _flatten_messages(fake.requests[-1]["messages"])
    assert "2+3 等于几" in last_request
    assert "结果是 5" in last_request
    assert '"content": "5"' in last_request  # 工具输出 5 也在上下文里
    assert "那上一轮结果是多少" in last_request


# ---------- 工具报错 -> 自我修正 ----------

def test_tool_error_is_fed_back_for_self_correction():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient(
        [
            tool_call("calculator", {"expression": "1/0"}),  # 会报错
            final("除数为 0 无法计算，请换个表达式"),  # 模型看到错误后收尾
        ]
    )
    agent = Agent(fake)

    result = _run(agent, "算 1/0", session)
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "除数为 0 无法计算，请换个表达式"

    # 第一步工具失败，错误被记录
    call = result.steps[0].tool_calls[0]
    assert call.ok is False
    assert "division by zero" in call.error

    # 错误作为 tool message 回灌进上下文，供下一轮自我修正
    tool_msgs = [m for m in session.context.messages if m.role == "tool"]
    assert tool_msgs[0].content.startswith("Error:")
    assert "division by zero" in tool_msgs[0].content


# ---------- 死循环 -> max_steps 优雅退出 ----------

def test_max_steps_graceful_exit():
    mgr = SessionManager()
    session = mgr.create("s1")
    # 模型每轮都坚持调工具、从不收尾 -> 触发上限
    fake = FakeLLMClient(
        [
            tool_call("search", {"query": "无限循环"}),
            tool_call("search", {"query": "无限循环"}),
            tool_call("search", {"query": "无限循环"}),
        ]
    )
    agent = Agent(fake, max_steps=3)

    result = _run(agent, "帮我一直搜", session)
    assert result.stopped_reason == "max_steps"
    assert result.num_steps == 3
    assert result.final_answer is not None
    assert "最大迭代步数" in result.final_answer
    # 每轮都执行了工具
    assert all(step.tool_calls for step in result.steps)


def test_max_steps_counts_llm_iterations():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient(
        [
            tool_call("search", {"query": "a"}),
            tool_call("search", {"query": "b"}),
            final("终于结束"),
        ]
    )
    agent = Agent(fake, max_steps=5)
    result = _run(agent, "查两次再总结", session)
    assert result.stopped_reason == "final_answer"
    assert result.num_steps == 3
    assert fake.call_count == 3


# ---------- trace 输出 ----------

def test_tracer_emits_readable_steps():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient(
        [tool_call("calculator", {"expression": "6*7"}, content="算一下"), final("42")]
    )
    lines: list[str] = []
    agent = Agent(fake, tracer=lines.append)

    _run(agent, "6*7", session)
    text = "\n".join(lines)
    assert "Step 1" in text
    assert "Thought: 算一下" in text
    assert "Tool Call: calculator" in text
    assert "Tool Output: 42" in text
    assert "Final Answer: 42" in text


def test_render_trace_includes_stop_reason():
    mgr = SessionManager()
    session = mgr.create("s1")
    fake = FakeLLMClient([final("done")])
    agent = Agent(fake)
    result = _run(agent, "x", session)
    text = render_trace(result)
    assert "stopped_reason: final_answer" in text
    assert "Final Answer: done" in text
