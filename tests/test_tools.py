"""工具层端到端测试（阶段 1）。

约定（见项目锁定决策）：不引入 pytest-asyncio，异步用例统一用 asyncio.run 包裹；
不触碰任何真实 LLM。覆盖注册/schema 推导、3 个工具的输入输出与错误边界、
以及分发的异常捕获 / 超时 / 未知工具 / 参数错误。
"""

from __future__ import annotations

import asyncio

from tools import build_default_registry
from tools.calculator import calculator
from tools.registry import (
    ToolRegistry,
    build_tool_spec,
    tool,
)
from tools.search import search
from tools.todo import TodoStore, create_todo_tool


# ---------- schema 推导 ----------

def test_spec_from_type_hints_and_docstring():
    spec = build_tool_spec(calculator)
    assert spec.name == "calculator"
    assert spec.parameters["type"] == "object"
    assert spec.parameters["properties"]["expression"]["type"] == "string"
    assert spec.parameters["required"] == ["expression"]
    assert "description" in spec.parameters["properties"]["expression"]
    assert spec.description  # 来自 docstring 摘要


def test_spec_optional_param_has_default_and_not_required():
    spec = build_tool_spec(search)
    props = spec.parameters["properties"]
    assert props["query"]["type"] == "string"
    assert props["top_k"]["type"] == "integer"
    assert props["top_k"]["default"] == 3
    assert spec.parameters["required"] == ["query"]


def test_spec_todo_action_required_content_optional():
    spec = build_tool_spec(create_todo_tool(TodoStore()))
    props = spec.parameters["properties"]
    assert spec.parameters["required"] == ["action"]
    assert props["content"]["default"] == ""


def test_to_openai_tool_format():
    payload = build_tool_spec(calculator).to_openai_tool()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "calculator"
    assert payload["function"]["parameters"]["properties"]["expression"]


# ---------- 注册表 ----------

def test_default_registry_has_three_tools():
    registry, _ = build_default_registry()
    assert sorted(registry.names()) == ["calculator", "search", "todo_manager"]
    assert len(registry.schemas()) == 3


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    registry.register(calculator)
    try:
        registry.register(calculator)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("expected ValueError on duplicate registration")


# ---------- calculator ----------

def test_calculator_basic_and_functions():
    registry, _ = build_default_registry()

    def calc(expr):
        return asyncio.run(registry.call("calculator", {"expression": expr}))

    assert calc("2*(3+4)/5").output == "2.8"
    assert calc("10 // 3").output == "3"
    assert calc("10 % 3").output == "1"
    assert calc("2 ** 10").output == "1024"
    assert calc("sqrt(16)").output == "4"
    assert calc("-5 + 3").output == "-2"


def test_calculator_errors_are_captured():
    registry, _ = build_default_registry()

    div0 = asyncio.run(registry.call("calculator", {"expression": "1/0"}))
    assert div0.ok is False
    assert "division by zero" in div0.error

    bad_name = asyncio.run(registry.call("calculator", {"expression": "foo(1)"}))
    assert bad_name.ok is False

    empty = asyncio.run(registry.call("calculator", {"expression": "   "}))
    assert empty.ok is False

    # 危险表达式：属性访问/推导式等被 AST 白名单拒绝，不会执行
    danger = asyncio.run(registry.call("calculator", {"expression": "().__class__"}))
    assert danger.ok is False


# ---------- search ----------

def test_search_returns_ranked_results():
    registry, _ = build_default_registry()
    res = asyncio.run(registry.call("search", {"query": "react agent reasoning"}))
    assert res.ok is True
    assert isinstance(res.output, list) and res.output
    assert res.output[0]["title"].startswith("ReAct")
    assert res.output[0]["score"] >= res.output[-1]["score"]


def test_search_top_k_and_no_match():
    registry, _ = build_default_registry()
    limited = asyncio.run(registry.call("search", {"query": "天气", "top_k": 1}))
    assert len(limited.output) == 1

    none = asyncio.run(registry.call("search", {"query": "zzzz-no-match"}))
    assert none.ok is True and none.output == []


def test_search_invalid_inputs():
    registry, _ = build_default_registry()
    assert asyncio.run(registry.call("search", {"query": ""})).ok is False
    assert asyncio.run(registry.call("search", {"query": "a", "top_k": 0})).ok is False


# ---------- todo_manager ----------

def test_todo_add_list_done_clear():
    store = TodoStore()
    registry, _ = build_default_registry(todo_store=store)

    def todo(action, content=""):
        return asyncio.run(
            registry.call("todo_manager", {"action": action, "content": content})
        )

    added = todo("add", "写周报")
    assert added.ok and added.output["item"]["content"] == "写周报"
    assert added.output["item"]["id"] == 1

    todo("add", "跑测试")
    listed = todo("list")
    assert len(listed.output["items"]) == 2

    done = todo("done", "1")
    assert done.output["item"]["done"] is True

    cleared = todo("clear")
    assert cleared.output["removed"] == 2
    assert todo("list").output["items"] == []


def test_todo_invalid_action_and_bad_id():
    registry, _ = build_default_registry()
    bad_action = asyncio.run(registry.call("todo_manager", {"action": "delete"}))
    assert bad_action.ok is False
    assert "invalid action" in bad_action.error

    bad_id = asyncio.run(registry.call("todo_manager", {"action": "done", "content": "x"}))
    assert bad_id.ok is False

    missing_id = asyncio.run(registry.call("todo_manager", {"action": "done", "content": "99"}))
    assert missing_id.ok is False

    empty_add = asyncio.run(registry.call("todo_manager", {"action": "add", "content": "  "}))
    assert empty_add.ok is False


# ---------- 分发：异常 / 超时 / 未知工具 / 参数 ----------

def test_unknown_tool_returns_error():
    registry, _ = build_default_registry()
    res = asyncio.run(registry.call("no_such_tool", {}))
    assert res.ok is False
    assert "Unknown tool" in res.error
    assert "calculator" in res.error  # 错误里列出可用工具，便于 LLM 自我修正


def test_missing_argument_captured_as_error():
    registry, _ = build_default_registry()
    res = asyncio.run(registry.call("calculator", {}))  # 缺 expression
    assert res.ok is False
    assert "Invalid arguments" in res.error


def test_tool_timeout_is_captured():
    @tool
    async def slow(x: int) -> int:
        """一个会超时的工具。

        Args:
            x: 输入。
        """
        await asyncio.sleep(1.0)
        return x

    registry = ToolRegistry(default_timeout=0.05)
    registry.register(slow)
    res = asyncio.run(registry.call("slow", {"x": 1}))
    assert res.ok is False
    assert "timed out" in res.error


def test_result_to_text_serializes_output():
    registry, _ = build_default_registry()
    ok = asyncio.run(registry.call("calculator", {"expression": "1+1"}))
    assert ok.to_text() == "2"

    err = asyncio.run(registry.call("calculator", {"expression": "1/0"}))
    assert err.to_text().startswith("Error:")

    listed = asyncio.run(registry.call("search", {"query": "天气"}))
    assert "title" in listed.to_text()
