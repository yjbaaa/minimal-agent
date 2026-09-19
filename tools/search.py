"""search 工具：Mock 的知识库/网页检索。

真实项目里这里会接搜索 API 或向量库；MVP 阶段用一个小型内存文档集 +
朴素关键词打分来模拟"检索到若干条结果"的行为，保证离线、可测、可复现。
"""

from __future__ import annotations

from .registry import tool

__all__ = ["search"]

# 每条文档：标题、正文摘要、用于打分的关键词、来源链接
_MOCK_DB: list[dict[str, object]] = [
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "snippet": "ReAct 让模型交替产生推理轨迹(Thought)与动作(Action)，"
        "动作结果作为观察(Observation)回灌上下文，从而边想边做。",
        "keywords": ["react", "agent", "reasoning", "tool", "thought"],
        "url": "https://arxiv.org/abs/2210.03629",
    },
    {
        "title": "什么是 Function Calling",
        "snippet": "Function Calling 让 LLM 以结构化 JSON 选择并填参调用外部函数，"
        "由运行时执行后把结果作为 tool message 返回。",
        "keywords": ["function", "tool", "calling", "json", "schema"],
        "url": "https://platform.openai.com/docs/guides/function-calling",
    },
    {
        "title": "上下文窗口与滑动窗口压缩",
        "snippet": "当对话历史逼近 token 上限时，可保留头部 system prompt 与"
        "最近 N 轮对话，做滚动窗口截断以控制上下文长度。",
        "keywords": ["context", "window", "token", "compression", "memory"],
        "url": "https://example.com/context-compression",
    },
    {
        "title": "北京今日天气",
        "snippet": "北京，晴，气温 18~26℃，东南风 2 级，空气质量良。",
        "keywords": ["天气", "北京", "weather", "气温"],
        "url": "https://example.com/weather/beijing",
    },
    {
        "title": "上海今日天气",
        "snippet": "上海，多云转小雨，气温 20~27℃，东风 3 级，空气质量优。",
        "keywords": ["天气", "上海", "weather", "气温"],
        "url": "https://example.com/weather/shanghai",
    },
]


def _score(query_terms: list[str], doc: dict[str, object]) -> float:
    haystack = " ".join(
        [
            str(doc["title"]).lower(),
            str(doc["snippet"]).lower(),
            " ".join(str(k).lower() for k in doc["keywords"]),  # type: ignore[union-attr]
        ]
    )
    score = 0.0
    for term in query_terms:
        if term and term in haystack:
            score += 1.0
    return score


@tool
async def search(query: str, top_k: int = 3) -> list[dict[str, object]]:
    """在知识库中检索与查询相关的条目（Mock 检索）。

    对查询做简单分词打分，返回相关度最高的若干条结果，每条含标题、摘要与来源。

    Args:
        query: 检索关键词，例如 "ReAct agent" 或 "北京 天气"。
        top_k: 最多返回的结果条数，默认 3。

    Returns:
        结果列表，每个元素是 {title, snippet, url, score}；无匹配时返回空列表。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    terms = query.strip().lower().split()
    scored = [
        {**{k: doc[k] for k in ("title", "snippet", "url")}, "score": _score(terms, doc)}
        for doc in _MOCK_DB
    ]
    matched = [item for item in scored if item["score"] > 0]
    matched.sort(key=lambda item: item["score"], reverse=True)
    return matched[:top_k]
