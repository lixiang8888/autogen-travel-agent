# -*- coding: utf-8 -*-
"""
tools.py —— 工具：search（Tavily）+ calculator
================================================

**只有两个工具，且各只挂给一个 agent**（BUILD.md §四个 Agent 的划界判据）：

    search      → researcher，全队唯一联网者
    calculator  → critic，全队唯一算账者

这个不对称是刻意的，不是省事：如果 planner 也能搜，它就会绕过 researcher 自己查，
群聊立刻退化成三个各自为战的单 agent。所以「谁能用哪个工具」是这个拓扑的地基，
改之前先回去读 docs/agents/ 下的四份说明书。

工具签名是**单个字符串入参**：AutoGen 把函数签名转成 JSON schema 喂给模型，
单参数最好写也最好读。要传多参数就传 JSON 字符串，工具自己 json.loads。

加工具三步：
    1. 写一个函数，一个字符串参数，返回字符串
    2. 在 ALL_TOOLS 里登记一行
    3. 在 persona.py 对应 AgentSpec 的 tool_names 里写上它的名字
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from llm import ENV_TAVILY, load_key

TAVILY_BASE = "https://api.tavily.com"
TAVILY_MAX_RESULTS = 5


# ---------------------------------------------------------------------------
# 1. search —— 只挂给 researcher
# ---------------------------------------------------------------------------

def search(query: str) -> str:
    """联网搜索，返回若干条网页的标题、摘要与来源链接。

    用于查询实时或你知识之外的信息：交通班次与票价、酒店价位、景点门票、
    开放时间等。搜不到时换一个更具体的关键词再试。

    Args:
        query: 搜索关键词，例如 "成都大熊猫基地 门票价格 2026"
    """
    api_key = load_key(ENV_TAVILY)
    if not api_key:
        return "错误：未配置 Tavily API key（环境变量 TAVILY_API_KEY 或 keys.py）"

    import requests   # 延迟导入：只用 calculator 的场合不必装 requests

    try:
        resp = requests.post(
            f"{TAVILY_BASE}/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": TAVILY_MAX_RESULTS,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:                       # noqa: BLE001 —— 工具层不抛异常
        return f"搜索失败：{type(exc).__name__}: {exc}。可以换个关键词重试。"

    results = data.get("results") or []
    if not results:
        return "搜索无结果，请换一个更具体的关键词重试。"

    blocks: list[str] = []
    for i, item in enumerate(results, 1):
        blocks.append(
            f"{i}. {item.get('title', '(无标题)')}\n"
            f"   {item.get('content', '').strip()}\n"
            f"   来源：{item.get('url', '')}"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# 2. calculator —— 只挂给 critic
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_POW_EXPONENT = 12


def calculator(expression: str) -> str:
    """计算一个算术表达式并返回数值结果。

    用它做所有加总、乘人数、比预算的运算——不要心算，哪怕是一眼能看出的加法。

    Args:
        expression: 算术表达式，例如 "750*2*2 + 350*3"。支持 + - * / // % ** 和括号。
    """
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree)
    except Exception as exc:                       # noqa: BLE001 —— 工具层不抛异常
        return f"计算失败：{exc}。请检查表达式，只写数字和 + - * / // % ** 与括号。"
    return f"{expression.strip()} = {_fmt(value)}"


def _eval_node(node: ast.AST) -> float:
    """白名单求值：只认数字和算术运算符，不碰任何名字、属性、函数调用。

    用 ast 而不是 eval()，是因为这个表达式的来源是模型输出——不可信输入。
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量 {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符 {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("指数过大")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("除数为零")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符 {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    raise ValueError(f"不支持的语法 {type(node).__name__}")


def _fmt(value: float) -> str:
    """整数就不显示小数点：3980.0 → 3980。"""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


# ---------------------------------------------------------------------------
# 3. 注册表
# ---------------------------------------------------------------------------

ALL_TOOLS: dict[str, Callable[[str], str]] = {
    "search": search,
    "calculator": calculator,
}


def build_tools(names: list[str] | tuple[str, ...]) -> list[Callable[[str], str]]:
    """按名字取出工具函数，交给 AutoGen 的 AssistantAgent(tools=...)。

    每次返回同一批函数对象——它们是纯函数，无状态，不像 react agent 那边的
    工具类需要每个 agent 一份实例。
    """
    tools: list[Callable[[str], str]] = []
    for name in names:
        if name not in ALL_TOOLS:
            raise KeyError(f"工具 {name!r} 没在 ALL_TOOLS 里登记。可用：{sorted(ALL_TOOLS)}")
        tools.append(ALL_TOOLS[name])
    return tools
