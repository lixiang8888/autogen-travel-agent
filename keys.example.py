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
# 可选：模型名（不填就用 llm.py 里的默认值 deepseek-flash）
# ---------------------------------------------------------------------------

# api.deepseek.com 认两个模型名：
#   deepseek-flash       默认，快而便宜 —— 本项目用这个
#   deepseek-v4-pro      推理模型，更强但慢、贵
#
# BUILD.md 里写的 `deepseek-chat` 已实测为过期，代码里没用它。
# 要换模型时把这行打开改掉即可，不用动代码：
#
# DEEPSEEK_MODEL = "deepseek-v4-pro"
