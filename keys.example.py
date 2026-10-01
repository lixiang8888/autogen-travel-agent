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
    AMAP_API_KEY      https://console.amap.com/dev/key/app
"""

# DeepSeek —— 跑三个 LLM agent（researcher / planner / critic）
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Tavily —— search 与 fetch_page 的后端，只挂给 researcher
TAVILY_API_KEY = "tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 高德 —— taxi_fare（路径规划 + 地理编码）与 hotel_options（搜索 POI）的后端，
# 两个都只挂给 researcher
#
# 申请时有两个地方容易选错，都会让 key 报出一堆看不懂的错：
#   1. 服务平台必须选「Web服务」。选成 Android/iOS 的 key 调不了 restapi，
#      报 INVALID_USER_KEY——看着像 key 坏了，其实是平台不对。
#   2. IP 白名单留空。本项目跑在 WSL2 里，出口 IP 是 Windows 侧的动态地址，
#      填了白名单过几天就会因 IP 变动而全部请求失败，且报错同样像 key 失效。
#
# 新 key 的 QPS 限制很紧，未认证时连调三四次就可能撞上「QPS 超限」。tools.py 里
# 有退避重试兜着；想彻底解决就去做个人认证（顺带把月配额提到 15 万次）。
AMAP_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# ---------------------------------------------------------------------------
# 可选：模型名（不填就用 llm.py 里的默认值 deepseek-flash）
# ---------------------------------------------------------------------------

# api.deepseek.com 认两个模型名：
#   deepseek-flash       默认，快而便宜 —— 本项目用这个
#   deepseek-v4-pro      推理模型，更强但慢、贵
#
# 老文档里写的 `deepseek-chat` 已实测为过期，代码里没用它。
# 要换模型时把这行打开改掉即可，不用动代码：
#
# DEEPSEEK_MODEL = "deepseek-v4-pro"
