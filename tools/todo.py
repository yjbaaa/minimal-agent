"""todo_manager 工具：可读写的待办清单。

状态保存在 :class:`TodoStore` 里。工具通过工厂 :func:`create_todo_tool` 绑定到
某个 store 实例——这样阶段 3 做 session 隔离时，每个会话可持有独立的 store，
待办不会跨会话串味。内存存储（见项目锁定决策 in-memory dict/list）。
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from .registry import tool

__all__ = ["TodoItem", "TodoStore", "create_todo_tool"]

_VALID_ACTIONS = ("add", "list", "done", "clear")


class TodoItem(BaseModel):
    """一条待办。"""

    id: int = Field(..., description="自增 id，用于标记完成")
    content: str = Field(..., description="待办内容")
    done: bool = Field(default=False, description="是否已完成")


class TodoStore:
    """待办状态的内存存储。"""

    def __init__(self) -> None:
        self._items: list[TodoItem] = []
        self._next_id = 1

    def add(self, content: str) -> TodoItem:
        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")
        item = TodoItem(id=self._next_id, content=content, done=False)
        self._items.append(item)
        self._next_id += 1
        return item

    def list(self) -> list[TodoItem]:
        return list(self._items)

    def done(self, item_id: int) -> TodoItem:
        for item in self._items:
            if item.id == item_id:
                item.done = True
                return item
        raise ValueError(f"todo id {item_id} not found")

    def clear(self) -> int:
        removed = len(self._items)
        self._items.clear()
        return removed


def create_todo_tool(store: TodoStore) -> Callable[..., object]:
    """返回一个绑定到 ``store`` 的 todo_manager 工具函数（已被 @tool 装饰）。"""

    @tool
    async def todo_manager(action: str, content: str = "") -> dict[str, object]:
        """管理一个待办清单：新增、查看、标记完成或清空。

        Args:
            action: 操作类型，取值 "add"（新增）、"list"（列出全部）、
                "done"（标记完成，需在 content 传入待办 id）、"clear"（清空）。
            content: add 时是待办正文；done 时是待办 id（字符串）；其余可留空。

        Returns:
            描述操作结果的字典，如 {"action": ..., "ok": True, ...}。
        """
        if action not in _VALID_ACTIONS:
            raise ValueError(
                f"invalid action '{action}', must be one of {', '.join(_VALID_ACTIONS)}"
            )

        if action == "add":
            item = store.add(content)
            return {"action": "add", "ok": True, "item": item.model_dump()}

        if action == "list":
            return {
                "action": "list",
                "ok": True,
                "items": [it.model_dump() for it in store.list()],
            }

        if action == "done":
            try:
                item_id = int(str(content).strip())
            except (TypeError, ValueError):
                raise ValueError(
                    f"content must be a todo id integer for 'done', got '{content}'"
                ) from None
            item = store.done(item_id)
            return {"action": "done", "ok": True, "item": item.model_dump()}

        # clear
        removed = store.clear()
        return {"action": "clear", "ok": True, "removed": removed}

    return todo_manager
