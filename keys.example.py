# -*- coding: utf-8 -*-
"""
keys.example.py —— keys.py 的模板（本文件会提交，里面不含任何真实密钥）
=========================================================================

用法：复制本文件为 keys.py，填入你的 key。

    cp keys.example.py keys.py

`keys.py` 已在 .gitignore 里，不会被提交。也可以不用 keys.py，改用环境变量或
项目根目录的 .env —— 三种途径的优先级见 llm.py 的 load_key()：

    环境变量  >  keys.py  >  .env

申请地址：
    DEEPSEEK_API_KEY  https://platform.deepseek.com/api_keys
    TAVILY_API_KEY    https://app.tavily.com/home
"""

# DeepSeek —— 跑三个 LLM agent（researcher / planner / critic）
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Tavily —— search 工具的后端，只挂给 researcher
TAVILY_API_KEY = "tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# ---------------------------------------------------------------------------
# 可选：模型名（不填就用 llm.py 里的默认值 deepseek-chat）
# ---------------------------------------------------------------------------

# ⚠️ 这条需要你实测确认。BUILD.md 写的是 `deepseek-chat`，但姊妹项目
# plan-solve-agent 实测 api.deepseek.com **只认 `deepseek-flash` 和
# `deepseek-v4-pro`**（见那边的 .env.example）。两种说法冲突。
#
# 阶段 1 如果报 400 且消息里出现「supported API model names are ...」，
# 就把下面这行打开、改成报错里给出的名字，不用动代码：
#
# DEEPSEEK_MODEL = "deepseek-flash"
