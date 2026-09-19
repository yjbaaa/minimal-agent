"""minimal-agent CLI 入口。

两种运行方式：
- 真实 LLM：配置 API Key 后走 OpenAI 兼容端点（DeepSeek / 火山方舟 / OpenAI）。
- 离线 demo：``--demo`` 用 FakeLLMClient 回放脚本，无需 Key、不触网，
  用于快速验证"工具层 -> LLM -> 解析器 -> 会话 -> ReAct 循环"整条链路跑通。

示例::

    python main.py --demo --once "帮我算 (12+8)*3 并记一条待办"
    python main.py --model deepseek-chat --base-url https://api.deepseek.com
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from agent.loop import Agent, render_trace
from llm.base import LLMClient, ToolCall
from llm.fake import FakeLLMClient, final, tool_call, tool_calls
from session.manager import SessionManager

DEFAULT_SYSTEM_PROMPT = (
    "你是一个善用工具的 AI 助手。可用工具：calculator（安全数学计算）、"
    "search（知识库检索）、todo_manager（增删查待办）。"
    "需要外部能力时调用工具，拿到结果后再作答；信息足够时直接给最终答案。"
    "若工具返回错误，请据此调整参数重试或换一种方式。"
)


def _demo_client() -> FakeLLMClient:
    """离线 demo 的脚本化客户端：一次多工具调用 -> 检索 -> 收尾，覆盖三个工具。"""
    return FakeLLMClient(
        [
            tool_calls(
                [
                    ToolCall(id="d1", name="calculator", arguments={"expression": "(12+8)*3"}),
                    ToolCall(
                        id="d2",
                        name="todo_manager",
                        arguments={"action": "add", "content": "提交周报"},
                    ),
                ],
                content="先算数，再记一条待办",
            ),
            tool_call("search", {"query": "ReAct agent"}, content="顺便查下 ReAct"),
            final("算好了：(12+8)*3 = 60；已记录待办『提交周报』；并检索到 ReAct 相关资料。"),
        ],
        model="fake-demo-model",
    )


def build_llm(args: argparse.Namespace) -> LLMClient:
    if args.demo:
        return _demo_client()

    from llm.openai_client import OpenAIClient

    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("ARK_API_KEY")
    )
    if not api_key:
        sys.exit(
            "缺少 API Key：请设置环境变量 OPENAI_API_KEY（或 DEEPSEEK_API_KEY / ARK_API_KEY），"
            "或用 --api-key 传入；也可加 --demo 离线体验。"
        )
    return OpenAIClient(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_completion_tokens,
    )


def build_agent(llm: LLMClient, args: argparse.Namespace) -> Agent:
    return Agent(
        llm,
        max_steps=args.max_steps,
        tracer=(print if not args.quiet else None),
    )


def build_manager(args: argparse.Namespace) -> SessionManager:
    return SessionManager(
        system_prompt=args.system,
        max_history_turns=args.max_history_turns,
        max_tokens=args.max_context_tokens,
    )


async def handle_once(agent: Agent, manager: SessionManager, args: argparse.Namespace) -> None:
    session = manager.get_or_create(args.session)
    result = await agent.run(args.once, session)
    if args.quiet:
        print(render_trace(result))
    print(f"\n[final] {result.final_answer}")
    print(f"[stopped_reason] {result.stopped_reason} | [steps] {result.num_steps}")


async def handle_repl(agent: Agent, manager: SessionManager, args: argparse.Namespace) -> None:
    session_id = args.session
    print("minimal-agent 交互模式（:exit 退出，:sessions 列出会话，:session <id> 切换）")
    while True:
        try:
            line = input(f"[{session_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":exit", ":quit"):
            break
        if line == ":sessions":
            print("sessions:", manager.list_ids() or "(none)")
            continue
        if line.startswith(":session "):
            session_id = line[len(":session ") :].strip() or session_id
            print(f"switched to session '{session_id}'")
            continue

        session = manager.get_or_create(session_id)
        result = await agent.run(line, session)
        print(f"\n[final] {result.final_answer}")
        print(f"[stopped_reason] {result.stopped_reason} | [steps] {result.num_steps}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="minimal-agent：手写 ReAct Agent MVP")
    parser.add_argument("--once", help="单次提问后退出（非交互）")
    parser.add_argument("--demo", action="store_true", help="离线 demo，无需 API Key、不触网")
    parser.add_argument("--session", default="default", help="会话 id（默认 default）")
    parser.add_argument("--quiet", action="store_true", help="关闭流式 trace（改为结束时整体打印）")

    parser.add_argument("--model", default="deepseek-chat", help="模型名")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容端点 base_url")
    parser.add_argument("--api-key", default=None, help="API Key（不传则读环境变量）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)

    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="system prompt")
    parser.add_argument("--max-steps", type=int, default=5, help="单次输入的 ReAct 最大迭代步数")
    parser.add_argument("--max-history-turns", type=int, default=None, help="上下文滑动窗口轮数")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="上下文 token 预算")

    args = parser.parse_args(argv)
    if args.demo and not args.once:
        args.once = "帮我算 (12+8)*3 并记一条待办，再查下 ReAct"
    return args


async def amain(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    llm = build_llm(args)
    agent = build_agent(llm, args)
    manager = build_manager(args)
    if args.once:
        await handle_once(agent, manager, args)
    else:
        await handle_repl(agent, manager, args)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
