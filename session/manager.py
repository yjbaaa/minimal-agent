"""Session 与 SessionManager：会话隔离与内存状态管理。

每个 Session 持有**独立**的 Context 与 TodoStore，并绑定一个用该 store 构造的
ToolRegistry——因此窗口 A 的待办/上下文与窗口 B 完全隔离，不会串话（对应文档 (3)
与测试要求）。存储用内存 dict（见项目锁定决策）。
"""

from __future__ import annotations

import time
from typing import Any

from tools import build_default_registry
from tools.registry import ToolRegistry
from tools.todo import TodoStore

from .context import Context

__all__ = ["Session", "SessionManager"]


class Session:
    """一个独立会话的全部状态。"""

    def __init__(
        self,
        session_id: str,
        *,
        system_prompt: str | None = None,
        max_history_turns: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.created_at = time.time()
        self.todo_store = TodoStore()
        self.context = Context(
            system_prompt,
            max_history_turns=max_history_turns,
            max_tokens=max_tokens,
        )
        # 注册表绑定本会话独有的 todo_store -> 待办状态天然隔离
        self.registry: ToolRegistry = build_default_registry(todo_store=self.todo_store)[0]


class SessionManager:
    """按 session_id 管理会话，内存存储。"""

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        max_history_turns: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        self._system_prompt = system_prompt
        self._max_history_turns = max_history_turns
        self._max_tokens = max_tokens

    def create(self, session_id: str, **overrides: Any) -> Session:
        """新建会话；session_id 已存在则报错。"""
        if session_id in self._sessions:
            raise ValueError(f"session '{session_id}' already exists")
        session = Session(
            session_id,
            system_prompt=overrides.get("system_prompt", self._system_prompt),
            max_history_turns=overrides.get("max_history_turns", self._max_history_turns),
            max_tokens=overrides.get("max_tokens", self._max_tokens),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = self.create(session_id)
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)
