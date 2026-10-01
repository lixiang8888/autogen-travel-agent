# -*- coding: utf-8 -*-
"""
llm.py —— 模型客户端：DeepSeek 走 OpenAI 兼容协议
==================================================

本项目用 AutoGen 自带的 OpenAIChatCompletionClient，不自己写 HTTP 客户端——
群聊编排要的就是这个 client 对象本身。

**第一个会撞上的坑**：AutoGen 不认识 DeepSeek 的模型名。不给它显式的能力声明，
它会拒绝注册工具，报错信息还不直白——`critic` 因为要挂 handoff，会直接抛
「The model does not support function calling」。所以下面 `model_info` 里那个
`function_calling: True` 不能省。

    取舍：`family` 只能填 "unknown"，意味着 AutoGen 会走保守路径，
    部分针对特定模型族的优化拿不到。可接受。

用法：
    from llm import build_model
    model = build_model()
"""

from __future__ import annotations

import os
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

#: 模型名。**老文档里写的 `deepseek-chat` 已实测为过期**——api.deepseek.com
#: 认的是 `deepseek-flash` 和 `deepseek-v4-pro`（后者是推理模型，慢且贵）。
#: 要换模型就设环境变量 / keys.py / .env 里的 DEEPSEEK_MODEL，不用动代码。
DEFAULT_MODEL = "deepseek-flash"
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "").strip() or DEFAULT_MODEL

ENV_DEEPSEEK = "DEEPSEEK_API_KEY"
ENV_TAVILY = "TAVILY_API_KEY"
ENV_AMAP = "AMAP_API_KEY"


# ---------------------------------------------------------------------------
# 1. 取 key：环境变量 > keys.py > .env
# ---------------------------------------------------------------------------

def load_key(env_name: str, key_module_attr: str | None = None) -> str:
    """按 环境变量 > keys.py > .env 的顺序取一个 key。

    环境变量排第一是有意的——它不进版本库，不会随文件被误提交。
    """
    value = (os.environ.get(env_name) or "").strip()
    if value:
        return value

    value = _from_keys_module(key_module_attr or env_name)
    if value:
        return value

    return _from_dotenv(env_name)


def _from_keys_module(attr: str) -> str:
    """从 keys.py 读。没有这个文件也要能跑（用户可能只用环境变量）。"""
    try:
        import keys as keys_module
    except ImportError:
        return ""
    return str(getattr(keys_module, attr, "") or "").strip()


def _from_dotenv(env_name: str) -> str:
    """从项目根目录的 .env 里读一行 KEY=VALUE。

    手写十行，不引 python-dotenv——这个项目不值得多一个依赖。
    """
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.is_file():
        return ""
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == env_name:
            return val.strip().strip('"').strip("'")
    return ""


# ---------------------------------------------------------------------------
# 2. 造模型客户端
# ---------------------------------------------------------------------------

def build_model(api_key: str | None = None) -> OpenAIChatCompletionClient:
    """造一个指向 DeepSeek 的 AutoGen 模型客户端。

    Args:
        api_key: 显式指定 key。不传就按 load_key 的顺序去找。

    Raises:
        RuntimeError: 三种途径都没找到 key 时，报错信息里写清三种给法。
    """
    key = api_key if api_key is not None else load_key(ENV_DEEPSEEK)
    if not key:
        raise RuntimeError(
            "没找到 DEEPSEEK_API_KEY。三种给法，任选一种：\n"
            "  1. cp keys.example.py keys.py，然后填进去\n"
            "  2. export DEEPSEEK_API_KEY=sk-...\n"
            "  3. 在项目根目录建 .env，写一行 DEEPSEEK_API_KEY=sk-..."
        )
    # 模型名优先从配置里取，取不到才用模块级默认值
    # ——这样改模型不用动代码，也方便在阶段 1 快速试哪个名字能通。
    model_name = load_key("DEEPSEEK_MODEL") or DEEPSEEK_MODEL
    return OpenAIChatCompletionClient(
        model=model_name,
        base_url=DEEPSEEK_BASE_URL,
        api_key=key,
        model_info={
            "vision": False,
            "function_calling": True,     # ← 不写这个工具挂不上；critic 挂了 handoff，省了它直接造不出来
            "json_output": True,
            "family": "unknown",
            "structured_output": False,
        },
    )
