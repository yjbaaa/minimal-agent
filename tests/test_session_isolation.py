"""Session 隔离测试（阶段 3）：窗口 A 与窗口 B 并行，待办/上下文互不串味。"""

from __future__ import annotations

import asyncio

from session.manager import Session, SessionManager


def test_get_or_create_reuses_same_session():
    mgr = SessionManager(system_prompt="你是助手")
    s1 = mgr.get_or_create("win-1")
    s2 = mgr.get_or_create("win-1")
    assert s1 is s2
    assert len(mgr) == 1
    assert mgr.get("win-1") is s1


def test_default_config_propagates_to_sessions():
    mgr = SessionManager(system_prompt="sys", max_history_turns=3, max_tokens=500)
    s = mgr.create("a")
    assert s.context.system_prompt == "sys"
    assert s.context.max_history_turns == 3
    assert s.context.max_tokens == 500


def test_todo_state_isolated_between_sessions():
    mgr = SessionManager()
    a = mgr.get_or_create("A")
    b = mgr.get_or_create("B")

    def add(sess: Session, text: str):
        return asyncio.run(
            sess.registry.call("todo_manager", {"action": "add", "content": text})
        )

    def list_todos(sess: Session):
        return asyncio.run(sess.registry.call("todo_manager", {"action": "list"}))

    add(a, "查天气")
    add(a, "记待办")
    add(b, "写周报")

    a_items = list_todos(a).output["items"]
    b_items = list_todos(b).output["items"]

    assert [it["content"] for it in a_items] == ["查天气", "记待办"]
    assert [it["content"] for it in b_items] == ["写周报"]
    # 底层 store 也是不同对象
    assert a.todo_store is not b.todo_store


def test_todo_ids_are_independent_per_session():
    mgr = SessionManager()
    a = mgr.get_or_create("A")
    b = mgr.get_or_create("B")

    asyncio.run(a.registry.call("todo_manager", {"action": "add", "content": "x"}))
    res_b = asyncio.run(b.registry.call("todo_manager", {"action": "add", "content": "y"}))
    # B 的第一条 id 从 1 开始，不受 A 影响
    assert res_b.output["item"]["id"] == 1


def test_context_isolated_between_sessions():
    mgr = SessionManager(system_prompt="sys")
    a = mgr.get_or_create("A")
    b = mgr.get_or_create("B")

    a.context.add_user("窗口A的消息")
    a.context.add_assistant("A的回复")
    b.context.add_user("窗口B的消息")

    a_texts = [m.text() for m in a.context.messages]
    b_texts = [m.text() for m in b.context.messages]

    assert any("窗口A" in t for t in a_texts)
    assert not any("窗口B" in t for t in a_texts)
    assert any("窗口B" in t for t in b_texts)
    assert not any("窗口A" in t for t in b_texts)


def test_parallel_conversation_no_cross_contamination():
    """模拟两个窗口交替对话，验证状态各自独立累积。"""
    mgr = SessionManager(system_prompt="你是助手")

    def turn(sess_id: str, user_text: str, assistant_text: str):
        sess = mgr.get_or_create(sess_id)
        sess.context.add_user(user_text)
        asyncio.run(sess.registry.call("todo_manager", {"action": "add", "content": user_text}))
        sess.context.add_assistant(assistant_text)

    # 交错进行
    turn("A", "A-task-1", "A-reply-1")
    turn("B", "B-task-1", "B-reply-1")
    turn("A", "A-task-2", "A-reply-2")

    a = mgr.get("A")
    b = mgr.get("B")

    a_todos = asyncio.run(a.registry.call("todo_manager", {"action": "list"})).output["items"]
    b_todos = asyncio.run(b.registry.call("todo_manager", {"action": "list"})).output["items"]

    assert [t["content"] for t in a_todos] == ["A-task-1", "A-task-2"]
    assert [t["content"] for t in b_todos] == ["B-task-1"]
    # 上下文条数：A 有 system + 2*(user+assistant) = 5；B = system + 2 = 3
    assert len(a.context.messages) == 5
    assert len(b.context.messages) == 3


def test_delete_session():
    mgr = SessionManager()
    mgr.create("tmp")
    assert "tmp" in mgr
    assert mgr.delete("tmp") is True
    assert "tmp" not in mgr
    assert mgr.delete("tmp") is False


def test_create_duplicate_raises():
    mgr = SessionManager()
    mgr.create("dup")
    try:
        mgr.create("dup")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected ValueError on duplicate session")
