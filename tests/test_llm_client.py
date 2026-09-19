"""LLM 客户端测试（阶段 2）：OpenAI 适配器归一化 + Fake 客户端回放。

OpenAI 适配器用桩 client 注入，验证请求构造与响应归一化，全程离线不触网。
"""

from __future__ import annotations

import asyncio

from llm.base import LLMResponse, ToolCall
from llm.fake import FakeLLMClient, final, tool_call
from llm.openai_client import OpenAIClient


# ---------- OpenAI 响应桩对象 ----------

class _StubFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _StubFn(name, arguments)


class _StubMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StubChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _StubUsage:
    def __init__(self, p, c, t):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t


class _StubResponse:
    def __init__(self, choices, model="stub-model", usage=None):
        self.choices = choices
        self.model = model
        self.usage = usage


class _StubCompletions:
    def __init__(self, response, sink):
        self._response = response
        self._sink = sink

    async def create(self, **kwargs):
        self._sink.append(kwargs)
        return self._response


class _StubChat:
    def __init__(self, completions):
        self.completions = completions


class _StubClient:
    def __init__(self, response):
        self.captured: list[dict] = []
        self.chat = _StubChat(_StubCompletions(response, self.captured))


def _make_client(response):
    stub = _StubClient(response)
    return OpenAIClient(model="deepseek-chat", client=stub), stub


# ---------- OpenAI 适配器 ----------

def test_adapter_normalizes_native_tool_calls():
    resp = _StubResponse(
        choices=[
            _StubChoice(
                _StubMessage(
                    content="算一下",
                    tool_calls=[_StubToolCall("c1", "calculator", '{"expression": "2+2"}')],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=_StubUsage(10, 5, 15),
    )
    client, stub = _make_client(resp)
    out = asyncio.run(client.complete([{"role": "user", "content": "2+2=?"}]))

    assert isinstance(out, LLMResponse)
    assert out.content == "算一下"
    assert out.finish_reason == "tool_calls"
    assert out.tool_calls[0].name == "calculator"
    assert out.tool_calls[0].arguments == {"expression": "2+2"}  # JSON 字符串已解析
    assert out.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert out.model == "stub-model"


def test_adapter_bad_arguments_json_degrades_to_empty_dict():
    resp = _StubResponse(
        choices=[
            _StubChoice(
                _StubMessage(tool_calls=[_StubToolCall("c1", "search", "{not json")]),
                finish_reason="tool_calls",
            )
        ]
    )
    client, _ = _make_client(resp)
    out = asyncio.run(client.complete([{"role": "user", "content": "x"}]))
    assert out.tool_calls[0].arguments == {}


def test_adapter_plain_text_response():
    resp = _StubResponse(choices=[_StubChoice(_StubMessage(content="你好"))])
    client, _ = _make_client(resp)
    out = asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert out.content == "你好"
    assert out.tool_calls == []


def test_adapter_passes_tools_and_tool_choice_auto():
    resp = _StubResponse(choices=[_StubChoice(_StubMessage(content="ok"))])
    client, stub = _make_client(resp)
    tools = [{"type": "function", "function": {"name": "calculator"}}]
    asyncio.run(client.complete([{"role": "user", "content": "hi"}], tools=tools))

    sent = stub.captured[0]
    assert sent["model"] == "deepseek-chat"
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"


def test_adapter_omits_tools_when_none():
    resp = _StubResponse(choices=[_StubChoice(_StubMessage(content="ok"))])
    client, stub = _make_client(resp)
    asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    sent = stub.captured[0]
    assert "tools" not in sent
    assert "tool_choice" not in sent


def test_adapter_requires_key_when_not_injected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        OpenAIClient(model="deepseek-chat")
    except ValueError as exc:
        assert "api_key" in str(exc)
    else:
        raise AssertionError("expected ValueError without api_key")


# ---------- Fake 客户端 ----------

def test_fake_replays_script_in_order_and_records():
    fake = FakeLLMClient(
        [
            tool_call("calculator", {"expression": "1+1"}),
            final("结果是 2"),
        ]
    )
    r1 = asyncio.run(fake.complete([{"role": "user", "content": "1+1"}], tools=[]))
    r2 = asyncio.run(fake.complete([{"role": "user", "content": "..."}]))

    assert r1.tool_calls[0].name == "calculator"
    assert r2.content == "结果是 2"
    assert fake.call_count == 2
    assert fake.requests[0]["messages"][0]["content"] == "1+1"


def test_fake_dict_script_item():
    fake = FakeLLMClient([{"content": "hi", "finish_reason": "stop"}])
    out = asyncio.run(fake.complete([{"role": "user", "content": "x"}]))
    assert out.content == "hi"


def test_fake_callable_script_item():
    def responder(messages, tools):
        return final(f"echo:{messages[-1]['content']}")

    fake = FakeLLMClient([responder])
    out = asyncio.run(fake.complete([{"role": "user", "content": "ping"}]))
    assert out.content == "echo:ping"


def test_fake_exhausted_returns_terminal_final():
    fake = FakeLLMClient([])
    out = asyncio.run(fake.complete([{"role": "user", "content": "x"}]))
    assert out.tool_calls == []
    assert out.content == "(no more scripted responses)"
