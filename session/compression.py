"""基础上下文压缩：token 估算 + 滑动窗口截断。

策略（对应项目文档 (3)）：
- **保留头部 System Prompt**：开头的连续 system 消息永不丢弃。
- **按"轮"切分**：每条 UserMessage 起一轮，其后跟随的 assistant/tool 消息属于同一轮。
- **滑动窗口**：``max_history_turns`` 只保留最近 N 轮。
- **token 预算**：``max_tokens`` 超阈值时从最旧一轮开始丢弃，直到达标；
  无论如何至少保留最近一轮，避免把上下文清空。

token 估算不引入 tokenizer 依赖：CJK 字符按每字 ~1 token，其余按 ~4 字符/token 估算。
这是保守近似，仅用于触发压缩，不追求与真实计费一致。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from pydantic import BaseModel, Field

from agent.schema import AnyMessage, SystemMessage

__all__ = [
    "estimate_tokens",
    "estimate_message_tokens",
    "estimate_tokens_total",
    "split_turns",
    "compress_messages",
    "CompressionResult",
]

_PER_MESSAGE_OVERHEAD = 4  # 每条消息的固定开销近似（role/分隔符等）


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK 统一表意
        or 0x3400 <= code <= 0x4DBF  # 扩展 A
        or 0x3000 <= code <= 0x303F  # CJK 标点
        or 0xFF00 <= code <= 0xFFEF  # 全角
        or 0x3040 <= code <= 0x30FF  # 日文假名
        or 0xAC00 <= code <= 0xD7AF  # 韩文
    )


def estimate_tokens(text: str) -> int:
    """粗略估算一段文本的 token 数（无 tokenizer 依赖）。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


def estimate_message_tokens(message: AnyMessage) -> int:
    """估算单条消息的 token 数（含角色、正文、工具调用参数）。"""
    total = _PER_MESSAGE_OVERHEAD + estimate_tokens(message.role)
    total += estimate_tokens(message.text())
    return total


def estimate_tokens_total(messages: Sequence[AnyMessage]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def split_head_and_body(
    messages: Sequence[AnyMessage],
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """拆出头部连续 system 消息与其后的正文历史。"""
    head: list[AnyMessage] = []
    idx = 0
    while idx < len(messages) and isinstance(messages[idx], SystemMessage):
        head.append(messages[idx])
        idx += 1
    return head, list(messages[idx:])


def split_turns(body: Sequence[AnyMessage]) -> list[list[AnyMessage]]:
    """把正文历史按轮切分：每条 UserMessage 开启新一轮。

    首个 UserMessage 之前的游离消息（若有）单独成组，置于最前，压缩时最先被丢弃。
    """
    turns: list[list[AnyMessage]] = []
    current: list[AnyMessage] = []
    for message in body:
        if message.role == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


class CompressionResult(BaseModel):
    """一次压缩的结果与统计，便于日志/测试观察。"""

    model_config = {"arbitrary_types_allowed": True}

    messages: list[AnyMessage] = Field(..., description="压缩后保留的消息")
    dropped_messages: int = Field(default=0, description="被丢弃的消息条数")
    dropped_turns: int = Field(default=0, description="被丢弃的轮数")
    truncated: bool = Field(default=False, description="是否发生了截断")
    tokens_before: int = Field(default=0, description="压缩前估算 token")
    tokens_after: int = Field(default=0, description="压缩后估算 token")


def compress_messages(
    messages: Sequence[AnyMessage],
    *,
    max_history_turns: int | None = None,
    max_tokens: int | None = None,
) -> CompressionResult:
    """对消息流应用滑动窗口 + token 预算压缩，非破坏式（不改入参）。"""
    all_messages = list(messages)
    tokens_before = estimate_tokens_total(all_messages)

    head, body = split_head_and_body(all_messages)
    head_tokens = estimate_tokens_total(head)
    turns = split_turns(body)
    total_turns = len(turns)

    kept = turns

    # 1) 轮数窗口
    if max_history_turns is not None and max_history_turns >= 0:
        if len(kept) > max_history_turns:
            kept = kept[len(kept) - max_history_turns :]

    # 2) token 预算：从最旧轮开始丢，至少保留最近一轮
    if max_tokens is not None:
        budget_for_body = max_tokens - head_tokens
        while len(kept) > 1 and estimate_tokens_total(_flatten(kept)) > budget_for_body:
            kept = kept[1:]

    kept_messages = head + _flatten(kept)
    dropped_turns = total_turns - len(kept)
    dropped_messages = len(all_messages) - len(kept_messages)

    return CompressionResult(
        messages=kept_messages,
        dropped_messages=max(0, dropped_messages),
        dropped_turns=max(0, dropped_turns),
        truncated=dropped_messages > 0,
        tokens_before=tokens_before,
        tokens_after=estimate_tokens_total(kept_messages),
    )


def _flatten(turns: Sequence[Sequence[AnyMessage]]) -> list[AnyMessage]:
    out: list[AnyMessage] = []
    for turn in turns:
        out.extend(turn)
    return out
