"""calculator 工具：基于 ast 的安全数学表达式求值。

不使用 ``eval``，而是解析成 AST 后只放行白名单节点（数字常量、四则/幂/取模、
一元正负、白名单函数与常量），从根源上杜绝任意代码执行。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from .registry import tool

__all__ = ["calculator"]

_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}

_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and isinstance(right, (int, float)) and right > 512:
            raise ValueError("Exponent too large (max 512)")
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"Unknown name: '{node.id}'")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            fname = getattr(node.func, "id", "<expr>")
            raise ValueError(f"Function not allowed: '{fname}'")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        args = [_eval_node(arg) for arg in node.args]
        return _FUNCS[node.func.id](*args)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _safe_eval(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{round(value, 10):g}"
    return str(value)


@tool
async def calculator(expression: str) -> str:
    """计算一个数学表达式并返回结果字符串。

    支持四则运算、整除、取模、幂、括号，以及 sqrt/sin/cos/log/abs/round 等
    函数和 pi/e 常量。表达式非法时会返回错误说明。

    Args:
        expression: 要计算的数学表达式，例如 "2*(3+4)/5" 或 "sqrt(16)+pi"。

    Returns:
        计算结果的字符串形式。
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    try:
        value = _safe_eval(expression.strip())
    except ZeroDivisionError:
        raise ValueError("division by zero") from None
    except (ValueError, TypeError, OverflowError, ArithmeticError) as exc:
        raise ValueError(f"cannot evaluate '{expression}': {exc}") from exc
    return _format_number(value)
