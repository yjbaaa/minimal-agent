"""工具注册表与 @tool 装饰器。

本模块是工具层的运行时核心，职责：
1. 从函数的类型提示 + docstring 自动推导出 JSON Schema，供 LLM 理解工具。
2. 提供 ``ToolRegistry``：注册、查询、导出 OpenAI tools 格式、异步分发调用。
3. 统一的异常捕获：工具执行报错/超时时，转成结构化的错误信息返回给调用方
   （最终回灌给 LLM 让其自我修正），而不是让进程崩溃。

约定：所有内置工具都是 ``async`` 函数（见项目锁定决策 async throughout），
分发时用 ``asyncio.wait_for`` 施加超时。
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
import typing
from collections.abc import Awaitable, Callable
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, Field

__all__ = [
    "ToolSpec",
    "RegisteredTool",
    "ToolCallResult",
    "ToolRegistry",
    "build_tool_spec",
    "tool",
]

# Python 类型 -> JSON Schema 类型
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_ARGS_HEADERS = {"args:", "arguments:", "parameters:", "参数:"}
_OTHER_HEADERS = {
    "returns:",
    "return:",
    "raises:",
    "yields:",
    "examples:",
    "example:",
    "note:",
    "notes:",
    "返回:",
    "示例:",
}


class ToolSpec(BaseModel):
    """工具的元数据描述，可序列化为 OpenAI function-calling 的 tool 结构。"""

    name: str = Field(..., description="工具名，LLM 通过它选择工具")
    description: str = Field(default="", description="工具用途说明")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="参数的 JSON Schema（object 类型）"
    )

    def to_openai_tool(self) -> dict[str, Any]:
        """转成 OpenAI / OpenAI 兼容协议的 ``tools`` 数组元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class RegisteredTool(BaseModel):
    """注册表里的一条记录：元数据 + 可调用体 + 单次调用超时。"""

    model_config = {"arbitrary_types_allowed": True}

    spec: ToolSpec
    func: Callable[..., Any]
    timeout: float | None = Field(
        default=None, description="该工具的超时秒数，None 表示用注册表默认值"
    )


class ToolCallResult(BaseModel):
    """一次工具调用的结果。

    ``ok=False`` 时 ``error`` 是可直接回灌给 LLM 的标准错误文本；
    ``ok=True`` 时 ``output`` 是工具返回值（任意可序列化对象）。
    """

    model_config = {"arbitrary_types_allowed": True}

    tool: str
    ok: bool
    output: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_text(self) -> str:
        """把结果渲染成给 LLM 的 Tool Message 文本。"""
        if not self.ok:
            return f"Error: {self.error}"
        if isinstance(self.output, str):
            return self.output
        import json

        try:
            return json.dumps(self.output, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(self.output)


def _json_schema_for_type(tp: Any) -> dict[str, Any]:
    """把一个 Python 类型注解转换成 JSON Schema 片段。"""
    if tp is inspect.Parameter.empty or tp is Any:
        return {}

    origin = get_origin(tp)
    # Optional[X] / X | None -> 取非 None 分支
    if origin is Union or origin is typing.Optional:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _json_schema_for_type(args[0])
        return {"oneOf": [_json_schema_for_type(a) for a in args]}

    if origin in (list, set, tuple):
        item_args = get_args(tp)
        schema: dict[str, Any] = {"type": "array"}
        if item_args:
            schema["items"] = _json_schema_for_type(item_args[0])
        return schema

    if origin is dict:
        return {"type": "object"}

    if isinstance(tp, type) and tp in _JSON_TYPES:
        return {"type": _JSON_TYPES[tp]}

    # 兜底：未知类型不强加约束
    return {}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """解析 Google 风格 docstring，返回 (摘要, {参数名: 描述})。"""
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()

    summary_lines: list[str] = []
    idx = 0
    while idx < len(lines) and lines[idx].strip():
        summary_lines.append(lines[idx].strip())
        idx += 1
    summary = " ".join(summary_lines)

    param_docs: dict[str, str] = {}
    section: str | None = None
    last_param: str | None = None
    for raw in lines[idx:]:
        stripped = raw.strip()
        lowered = stripped.lower()
        if lowered in _ARGS_HEADERS:
            section = "args"
            last_param = None
            continue
        if lowered in _OTHER_HEADERS:
            section = None
            last_param = None
            continue
        if section != "args" or not stripped:
            continue

        match = re.match(r"^(\w+)\s*(\([^)]*\))?\s*[:：]\s*(.*)$", stripped)
        if match:
            last_param = match.group(1)
            param_docs[last_param] = match.group(3).strip()
        elif last_param is not None:
            # 续行：拼接到上一个参数描述
            param_docs[last_param] = (param_docs[last_param] + " " + stripped).strip()

    return summary, param_docs


def build_tool_spec(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolSpec:
    """从函数签名 + docstring 推导出 :class:`ToolSpec`。"""
    summary, param_docs = _parse_docstring(func.__doc__)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # noqa: BLE001 - 注解无法解析时退回空提示
        hints = {}

    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        prop = _json_schema_for_type(hints.get(pname, param.annotation))
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)
        properties[pname] = prop

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return ToolSpec(
        name=name or func.__name__,
        description=description if description is not None else summary,
        parameters=parameters,
    )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """``@tool`` 装饰器：把推导出的 ToolSpec 挂到函数上（``__tool_spec__``）。

    它不直接注册到任何注册表，因此工具函数可被复用/测试；随后用
    :meth:`ToolRegistry.register` 收纳。支持 ``@tool`` 与 ``@tool(...)`` 两种写法。
    """

    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__tool_spec__ = build_tool_spec(f, name=name, description=description)  # type: ignore[attr-defined]
        return f

    if func is not None:
        return decorate(func)
    return decorate


class ToolRegistry:
    """工具的注册、查询与异步分发。"""

    def __init__(self, default_timeout: float = 10.0) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._default_timeout = default_timeout

    # ---- 注册 ----
    def register(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        override: bool = False,
    ) -> RegisteredTool:
        """把一个（通常已被 ``@tool`` 装饰的）函数登记进注册表。"""
        spec = getattr(func, "__tool_spec__", None)
        if spec is None or name is not None or description is not None:
            spec = build_tool_spec(func, name=name, description=description)
        if spec.name in self._tools and not override:
            raise ValueError(f"Tool '{spec.name}' is already registered")

        registered = RegisteredTool(spec=spec, func=func, timeout=timeout)
        self._tools[spec.name] = registered
        return registered

    def add(self, registered: RegisteredTool, *, override: bool = False) -> None:
        """直接登记一个已构造好的 :class:`RegisteredTool`。"""
        if registered.spec.name in self._tools and not override:
            raise ValueError(f"Tool '{registered.spec.name}' is already registered")
        self._tools[registered.spec.name] = registered

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
    ) -> Callable[..., Any]:
        """装饰器写法：``@registry.tool`` 直接把函数注册进本注册表。"""

        def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
            self.register(f, name=name, description=description, timeout=timeout)
            return f

        if func is not None:
            return decorate(func)
        return decorate

    # ---- 查询 ----
    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def schemas(self) -> list[dict[str, Any]]:
        """导出 OpenAI tools 数组，供拼装 LLM 请求。"""
        return [t.spec.to_openai_tool() for t in self._tools.values()]

    # ---- 分发 ----
    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> ToolCallResult:
        """执行一次工具调用，永不抛异常，全部转成 :class:`ToolCallResult`。"""
        started = time.perf_counter()
        args = arguments or {}

        registered = self._tools.get(name)
        if registered is None:
            return ToolCallResult(
                tool=name,
                ok=False,
                error=f"Unknown tool '{name}'. Available tools: {', '.join(self.names()) or '(none)'}",
                elapsed_ms=_ms(started),
            )

        effective_timeout = timeout or registered.timeout or self._default_timeout

        try:
            result = registered.func(**args)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=effective_timeout)
            return ToolCallResult(
                tool=name, ok=True, output=result, elapsed_ms=_ms(started)
            )
        except asyncio.TimeoutError:
            return ToolCallResult(
                tool=name,
                ok=False,
                error=f"Tool '{name}' timed out after {effective_timeout}s",
                elapsed_ms=_ms(started),
            )
        except TypeError as exc:
            # 参数不匹配（缺参/多参/类型名错误）
            return ToolCallResult(
                tool=name,
                ok=False,
                error=f"Invalid arguments for tool '{name}': {exc}",
                elapsed_ms=_ms(started),
            )
        except Exception as exc:  # noqa: BLE001 - 工具内任何异常都需回灌给 LLM
            return ToolCallResult(
                tool=name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=_ms(started),
            )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
