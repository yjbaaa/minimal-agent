"""ResponseParser 测试（阶段 2）：原生 tool_calls 优先 + content JSON 兜底。"""

from __future__ import annotations

import json

from agent.parser import ResponseParser
from llm.base import LLMResponse, ToolCall

parser = ResponseParser()


# ---------- 原生 function-calling ----------

def test_native_tool_calls_take_priority():
    resp = LLMResponse(
        content="我先算一下",
        tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})],
        finish_reason="tool_calls",
    )
    out = parser.parse(resp)
    assert out.is_final is False
    assert out.thought == "我先算一下"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "calculator"
    assert out.tool_calls[0].arguments == {"expression": "1+1"}


def test_native_multiple_tool_calls():
    resp = LLMResponse(
        tool_calls=[
            ToolCall(id="c1", name="search", arguments={"query": "天气"}),
            ToolCall(id="c2", name="calculator", arguments={"expression": "2*3"}),
        ]
    )
    out = parser.parse(resp)
    assert [c.name for c in out.tool_calls] == ["search", "calculator"]


def test_plain_content_is_final_answer():
    out = parser.parse(LLMResponse(content="这是最终回复", finish_reason="stop"))
    assert out.is_final is True
    assert out.final_answer == "这是最终回复"
    assert out.tool_calls == []


# ---------- JSON 兜底 ----------

def test_json_action_form():
    content = '{"thought": "需要计算", "action": "calculator", "action_input": {"expression": "6*7"}}'
    out = parser.parse(LLMResponse(content=content))
    assert out.is_final is False
    assert out.thought == "需要计算"
    call = out.tool_calls[0]
    assert call.name == "calculator"
    assert call.arguments == {"expression": "6*7"}
    assert call.id  # 自动合成 id


def test_json_final_answer_form():
    content = '{"thought": "信息够了", "final_answer": "结果是 42"}'
    out = parser.parse(LLMResponse(content=content))
    assert out.is_final is True
    assert out.thought == "信息够了"
    assert out.final_answer == "结果是 42"


def test_json_in_code_fence():
    content = '好的，我来查一下：\n```json\n{"action": "search", "action_input": {"query": "ReAct"}}\n```'
    out = parser.parse(LLMResponse(content=content))
    assert out.tool_calls[0].name == "search"
    assert out.tool_calls[0].arguments == {"query": "ReAct"}


def test_json_embedded_in_prose():
    content = '让我调用工具 {"action": "todo_manager", "action_input": {"action": "add", "content": "写周报"}} 完成后再回复。'
    out = parser.parse(LLMResponse(content=content))
    assert out.tool_calls[0].name == "todo_manager"
    assert out.tool_calls[0].arguments == {"action": "add", "content": "写周报"}


def test_json_tool_calls_list_form():
    payload = {
        "tool_calls": [
            {"name": "search", "arguments": {"query": "a"}},
            {"function": {"name": "calculator", "arguments": json.dumps({"expression": "1+1"})}},
        ]
    }
    out = parser.parse(LLMResponse(content=json.dumps(payload)))
    assert [c.name for c in out.tool_calls] == ["search", "calculator"]
    assert out.tool_calls[1].arguments == {"expression": "1+1"}


def test_action_input_as_json_string():
    content = json.dumps(
        {"action": "calculator", "action_input": json.dumps({"expression": "9-4"})}
    )
    out = parser.parse(LLMResponse(content=content))
    assert out.tool_calls[0].arguments == {"expression": "9-4"}


def test_malformed_json_degrades_to_final_answer():
    content = '{"action": "calculator", "action_input": '  # 坏掉的 JSON
    out = parser.parse(LLMResponse(content=content))
    assert out.is_final is True
    assert out.final_answer == content  # 原样当最终答案，不崩


def test_unrecognized_json_object_degrades_to_final_answer():
    content = '{"foo": "bar"}'  # 合法 JSON 但非 ReAct 结构
    out = parser.parse(LLMResponse(content=content))
    assert out.is_final is True
    assert out.final_answer == content


def test_empty_content_is_final():
    out = parser.parse(LLMResponse(content=None))
    assert out.is_final is True
    assert out.final_answer == ""
