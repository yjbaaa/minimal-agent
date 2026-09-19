"""上下文与压缩测试（阶段 3）：消息 to_dict、轮切分、滑动窗口、token 预算。"""

from __future__ import annotations

import json

from agent.schema import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from llm.base import ToolCall
from session.compression import (
    compress_messages,
    estimate_tokens,
    split_turns,
)
from session.context import Context


# ---------- 消息 to_dict ----------

def test_message_to_dict_shapes():
    assert SystemMessage(content="你是助手").to_dict() == {
        "role": "system",
        "content": "你是助手",
    }
    assert UserMessage(content="hi").to_dict() == {"role": "user", "content": "hi"}

    tool_msg = ToolMessage(content="4", tool_call_id="c1", name="calculator").to_dict()
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "4",
        "name": "calculator",
    }


def test_assistant_to_dict_with_tool_calls():
    msg = AssistantMessage(
        content="算一下",
        tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})],
    )
    d = msg.to_dict()
    assert d["role"] == "assistant"
    assert d["content"] == "算一下"
    assert d["tool_calls"][0]["type"] == "function"
    assert d["tool_calls"][0]["function"]["name"] == "calculator"
    # arguments 序列化成 JSON 字符串（OpenAI wire-format 要求）
    assert json.loads(d["tool_calls"][0]["function"]["arguments"]) == {"expression": "2+2"}


def test_assistant_to_dict_without_tool_calls_omits_field():
    d = AssistantMessage(content="just text").to_dict()
    assert "tool_calls" not in d


# ---------- token 估算 ----------

def test_estimate_tokens_cjk_vs_ascii():
    # 4 个中文字 ~ 4 token；16 个 ascii ~ 4 token
    assert estimate_tokens("中文字符测试") == 6
    assert estimate_tokens("a" * 16) == 4
    assert estimate_tokens("") == 0


# ---------- 轮切分 ----------

def _turn_fixture():
    return [
        SystemMessage(content="sys"),
        UserMessage(content="u1"),
        AssistantMessage(content="a1"),
        UserMessage(content="u2"),
        AssistantMessage(
            content=None, tool_calls=[ToolCall(id="c", name="search", arguments={})]
        ),
        ToolMessage(content="r", tool_call_id="c"),
        AssistantMessage(content="a2"),
    ]


def test_split_turns_groups_by_user_message():
    msgs = _turn_fixture()
    body = msgs[1:]  # 去掉 system
    turns = split_turns(body)
    assert len(turns) == 2
    assert turns[0][0].content == "u1"  # type: ignore[attr-defined]
    assert len(turns[0]) == 2  # u1 + a1
    assert len(turns[1]) == 4  # u2 + assistant(toolcall) + tool + a2


# ---------- 压缩 ----------

def test_compress_keeps_system_and_last_n_turns():
    msgs = _turn_fixture()
    result = compress_messages(msgs, max_history_turns=1)
    roles = [m.role for m in result.messages]
    # 保留 system + 最后一轮（u2 起 4 条）
    assert roles[0] == "system"
    assert result.messages[1].content == "u2"  # type: ignore[attr-defined]
    assert len(result.messages) == 5
    assert result.truncated is True
    assert result.dropped_turns == 1


def test_compress_no_limit_keeps_everything():
    msgs = _turn_fixture()
    result = compress_messages(msgs)
    assert len(result.messages) == len(msgs)
    assert result.truncated is False
    assert result.dropped_messages == 0


def test_compress_token_budget_drops_oldest_turns():
    msgs = _turn_fixture()
    full_tokens = compress_messages(msgs).tokens_after
    # 预算收紧到只够 system + 最近一轮
    result = compress_messages(msgs, max_tokens=full_tokens - 1)
    assert result.messages[0].role == "system"
    assert result.messages[1].content == "u2"  # type: ignore[attr-defined]
    assert result.tokens_after <= full_tokens


def test_compress_always_keeps_at_least_last_turn():
    msgs = _turn_fixture()
    # 预算极小，也不能把最近一轮丢掉、更不能丢 system
    result = compress_messages(msgs, max_tokens=1)
    assert result.messages[0].role == "system"
    assert any(m.role == "user" and m.content == "u2" for m in result.messages)  # type: ignore[union-attr]


# ---------- Context ----------

def test_context_is_nondestructive_and_builds_wire_messages():
    ctx = Context("你是助手", max_history_turns=1)
    ctx.add_user("u1")
    ctx.add_assistant("a1")
    ctx.add_user("u2")
    ctx.add_assistant("a2")

    # 全量历史保留
    assert len(ctx.messages) == 5
    # 压缩视图只保留 system + 最近一轮
    wire = ctx.build_messages()
    assert wire[0] == {"role": "system", "content": "你是助手"}
    assert [m["content"] for m in wire[1:]] == ["u2", "a2"]
    assert ctx.system_prompt == "你是助手"
    assert ctx.estimated_tokens() > 0
