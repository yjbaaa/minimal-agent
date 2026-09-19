"""工具层：内置工具与注册表。

对外暴露 :func:`build_default_registry`，一行装配 calculator / search /
todo_manager 三个内置工具，供测试与后续 main.py 使用。
"""

from __future__ import annotations

from .calculator import calculator
from .registry import (
    RegisteredTool,
    ToolCallResult,
    ToolRegistry,
    ToolSpec,
    build_tool_spec,
    tool,
)
from .search import search
from .todo import TodoItem, TodoStore, create_todo_tool

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "RegisteredTool",
    "ToolCallResult",
    "build_tool_spec",
    "tool",
    "calculator",
    "search",
    "TodoStore",
    "TodoItem",
    "create_todo_tool",
    "build_default_registry",
]


def build_default_registry(
    *, default_timeout: float = 10.0, todo_store: TodoStore | None = None
) -> tuple[ToolRegistry, TodoStore]:
    """构建含 3 个内置工具的注册表。

    返回 (registry, todo_store)。``todo_store`` 可外部传入以便复用/隔离，
    不传则新建一个空 store。
    """
    store = todo_store or TodoStore()
    registry = ToolRegistry(default_timeout=default_timeout)
    registry.register(calculator)
    registry.register(search)
    registry.register(create_todo_tool(store))
    return registry, store
