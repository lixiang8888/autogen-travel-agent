# -*- coding: utf-8 -*-
"""
tools.py —— 工具：search + fetch_page（Tavily）、taxi_fare（高德）、calculator
================================================================================

**五个工具，联网的四把都只挂给一个 agent**（划界判据：工具集不同，或信息视野不同）：

    search        → researcher，全队唯一联网者
    fetch_page    → researcher，同上（搜到的摘要截断时读全文）
    taxi_fare     → researcher，同上（两地之间的驾车距离/耗时/打车估价）
    hotel_options → researcher，同上（按商圈+档位列酒店，不含房价）
    calculator    → critic，全队唯一算账者

researcher 那四把是**同一把权力的四个动作**：「搜得到」「读得全」「算得出路线」
「列得出候选」。它们是同一条信息视野的四个入口，所以一起挂给 researcher，不构成
新的划界。挂到别人身上照样会废掉拓扑——理由同下。

`taxi_fare` 为什么必须是个工具、而不是让 researcher 拿运价表自己乘：它要的
**里程数**搜不出来，只能算。见 README「踩过的坑」第 11 条。

`hotel_options` 为什么查的是**档位**而不是房价：酒店实价按日期和房型动态生成、
且在登录墙后面，`fetch_page` 打开携程酒店页只拿得到一张登录表单。能查的只有
「这一带有哪些连锁、哪家是哪一档」。见 README「踩过的坑」第 13 条。

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

from llm import ENV_AMAP, ENV_TAVILY, load_key

TAVILY_BASE = "https://api.tavily.com"
TAVILY_MAX_RESULTS = 5

#: 搜索深度。**别退回 "basic"**：实测同一条「诸暨 景点 门票价格」，basic 返回的 5 条
#: 里没有一条带票价（全是攻略聚合页和一日游产品），advanced 能带出 4 个具体票价。
#: 代价见 README「踩过的坑」第 9 条。
TAVILY_SEARCH_DEPTH = "advanced"

#: fetch_page 单页正文的字符上限。Tavily 的 advanced 提取一页常有两三千字，而
#: researcher 一轮要抓好几次，全文灌进群聊会顶爆上下文（README「设计取舍」承认
#: 上下文膨胀是既定代价，但没理由主动加码）。超长就截断，并在文末说明截过。
FETCH_MAX_CHARS = 4000


# ---------------------------------------------------------------------------
# 1. search —— 只挂给 researcher
# ---------------------------------------------------------------------------

def search(query: str) -> str:
    """联网搜索，返回若干条网页的标题、摘要与来源链接。

    用于查询实时或你知识之外的信息：交通班次与票价、酒店价位、景点门票、
    开放时间等。搜不到时换一个更具体的关键词再试。

    摘要常常在关键处被截断——拿到链接但没拿到数字时，用 fetch_page 打开那一条读全文。

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
                "search_depth": TAVILY_SEARCH_DEPTH,
                "max_results": TAVILY_MAX_RESULTS,
            },
            timeout=60,
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
# 2. fetch_page —— 只挂给 researcher
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """打开一个网页，返回正文全文。

    搜索摘要经常在价目表、班次表前面被截断——摘要里能看到「运价调整有关事项的
    通知」，但看不到通知里的数字。这时用这个工具把那一页读全。

    Args:
        url: 完整网址，从 search 结果的「来源：」后面整条复制
    """
    api_key = load_key(ENV_TAVILY)
    if not api_key:
        return "错误：未配置 Tavily API key（环境变量 TAVILY_API_KEY 或 keys.py）"

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return f"错误：{url!r} 不是完整网址。请从 search 结果的「来源：」处整条复制。"

    import requests

    try:
        resp = requests.post(
            f"{TAVILY_BASE}/extract",
            json={"api_key": api_key, "urls": [url], "extract_depth": "advanced"},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:                       # noqa: BLE001 —— 工具层不抛异常
        return f"打开网页失败：{type(exc).__name__}: {exc}。可以换一个来源再试。"

    results = data.get("results") or []
    if not results:
        failed = data.get("failed_results") or []
        reason = failed[0].get("error", "无法提取正文") if failed else "没有返回内容"
        return f"这个网页打不开：{reason}。换一个来源再试。"

    content = (results[0].get("raw_content") or "").strip()
    if not content:
        return "这个网页没有可提取的正文（可能是纯图片或脚本渲染的页面），换一个来源再试。"

    if len(content) > FETCH_MAX_CHARS:
        content = content[:FETCH_MAX_CHARS] + f"\n…（正文过长，已截断到前 {FETCH_MAX_CHARS} 字符）"
    return f"{url}\n\n{content}"


# ---------------------------------------------------------------------------
# 3. taxi_fare —— 只挂给 researcher
# ---------------------------------------------------------------------------

AMAP_BASE = "https://restapi.amap.com/v3"

#: 搜索 POI 2.0 在 v5 下，地理编码和路径规划还在 v3——高德的版本号不是统一的，
#: 拿 v3 的基址去调 /place/text 会 404。分成两个常量，别合并。
AMAP_POI_BASE = "https://restapi.amap.com/v5"

#: 高德的错误码 → 人话。模型看不懂 INVALID_USER_KEY，但看得懂「key 没配」。
#: 10009 单列出来，是因为它最容易误判：key 本身是好的，只是创建时服务平台
#: 选成了 Android/iOS——那种 key 调不了 restapi，报错却像是 key 坏了。
_AMAP_ERRORS: dict[str, str] = {
    "10001": "高德 key 无效，检查 AMAP_API_KEY 是否填对",
    "10002": "高德 key 被停用或已删除",
    "10003": "高德 key 今日调用量超限",
    "10004": "高德 key 调用过于频繁，稍后重试",
    "10009": "高德 key 的服务平台类型不对：这个 key 必须建在「Web服务」平台下，"
             "Android/iOS 平台的 key 调不了 restapi",
    "10021": "高德 QPS 超限——新 key 的并发限制很紧。等几秒重试；持续出现就去做"
             "个人认证提升配额",
    "20000": "高德服务暂时不可用",
    "20800": "高德返回了规划失败（起终点太近或无法驾车抵达）",
    "30001": "高德没能解析这个地名，换个更完整的写法（带上城市、区县，或写成「XX路XX号」）",
}

#: 这些码是「你太快了」，不是「你错了」——等一会儿原样重试即可。
#: 一次 taxi_fare 要连发三次请求（两次地理编码 + 一次路径规划），而实测新 key
#: 连续调三四次就会撞上 10021，所以重试不是防御性编程，是必需。
_AMAP_RETRY_INFOCODES = frozenset({"10004", "10021"})
_AMAP_RETRY_TIMES = 3
_AMAP_RETRY_WAIT_SEC = 1.2


def _amap_get(
    path: str, params: dict[str, str], api_key: str, base: str = AMAP_BASE
) -> tuple[dict, str]:
    """调一次高德 Web 服务。返回 (数据, 错误信息)。错误信息为空串表示成功。

    key 在这里统一注入——每个高德接口都要它，散在各调用点上一定会漏（漏了的表现
    是 10001 INVALID_USER_KEY，看着像 key 坏了，其实是没传）。
    """
    import time

    import requests

    for attempt in range(_AMAP_RETRY_TIMES):
        try:
            resp = requests.get(
                f"{base}{path}",
                params={**params, "key": api_key},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:                   # noqa: BLE001 —— 工具层不抛异常
            return {}, f"调用高德失败：{type(exc).__name__}: {exc}"

        if data.get("status") == "1":
            return data, ""

        code = str(data.get("infocode", ""))
        # 只有「太快了」值得重试；key 错了、配额用完了，重试一万次也一样
        if code in _AMAP_RETRY_INFOCODES and attempt < _AMAP_RETRY_TIMES - 1:
            time.sleep(_AMAP_RETRY_WAIT_SEC)
            continue
        return {}, _AMAP_ERRORS.get(code, f"高德返回 {data.get('info')}（infocode {code}）")

    return {}, "高德连续多次 QPS 超限，请稍后再试"


def _amap_location(place: str, city: str, api_key: str) -> tuple[str, str]:
    """把地名换成「经度,纬度」。已经是坐标的原样返回。返回 (坐标, 错误信息)。"""
    # 模型有时直接给坐标，别浪费一次地理编码
    parts = place.split(",")
    if len(parts) == 2 and all(p.strip().replace(".", "", 1).lstrip("-").isdigit() for p in parts):
        return place.strip(), ""

    params = {"address": place}
    if city:
        params["city"] = city
    data, err = _amap_get("/geocode/geo", params, api_key)
    if err:
        return "", err

    geocodes = data.get("geocodes") or []
    if not geocodes:
        hint = "" if city else "。加上 city 参数指定城市能提高命中率"
        return "", f"高德找不到「{place}」这个地方{hint}"
    return geocodes[0].get("location", ""), ""


def taxi_fare(arguments: str) -> str:
    """算两地之间开车要多久、多远，以及打车大概多少钱。

    搜索引擎只能给你当地的运价表，给不了「从 A 到 B 具体多少钱」——那要按实际
    路线算。这个工具算得出来，市内交通的花费该用它。

    注意：金额是高德按通用计价模型估的，不套用当地运价文件。有回空补贴、
    夜间加价的城市，长途会偏低——报的时候要带上「高德估算」这几个字。

    Args:
        arguments: JSON 字符串，必填 from 与 to（地名，或「经度,纬度」），
                   可选 city 用于消歧。例如
                   {"from": "诸暨站", "to": "五泄风景区", "city": "诸暨"}
    """
    import json

    # 先校参数、后查 key：参数写错是模型自己能改的错，应该优先报出来；
    # 顺带让自测能在没有高德 key 的机器上离线测这两条守卫（见 main.py 第 [3] 节）。
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return f'参数不是合法 JSON：{exc}。应形如 {{"from":"诸暨站","to":"五泄风景区","city":"诸暨"}}'
    if not isinstance(args, dict):
        return '参数应是一个 JSON 对象，形如 {"from":"诸暨站","to":"五泄风景区","city":"诸暨"}'

    origin_name = str(args.get("from", "")).strip()
    dest_name = str(args.get("to", "")).strip()
    city = str(args.get("city", "")).strip()
    if not origin_name or not dest_name:
        return '缺少 from 或 to。应形如 {"from":"诸暨站","to":"五泄风景区","city":"诸暨"}'

    api_key = load_key(ENV_AMAP)
    if not api_key:
        return "错误：未配置高德 key（环境变量 AMAP_API_KEY 或 keys.py）"

    origin, err = _amap_location(origin_name, city, api_key)
    if err:
        return f"起点搞不定：{err}"
    dest, err = _amap_location(dest_name, city, api_key)
    if err:
        return f"终点搞不定：{err}"

    # 不传 strategy / extensions：实测这两个参数会触发 MISSING_REQUIRED_PARAMS，
    # 而最简参数集返回的 route 里本来就带着 taxi_cost。
    data, err = _amap_get(
        "/direction/driving", {"origin": origin, "destination": dest}, api_key
    )
    if err:
        return f"路线规划失败：{err}"

    route = data.get("route") or {}
    paths = route.get("paths") or []
    if not paths:
        return f"高德没给出「{origin_name}」到「{dest_name}」的驾车路线，可能太近或无法驾车抵达。"

    path = paths[0]
    distance_km = float(path.get("distance", 0)) / 1000
    minutes = float(path.get("duration", 0)) / 60

    lines = [
        f"驾车：{origin_name} → {dest_name}",
        f"距离 {distance_km:.1f} 公里，耗时约 {minutes:.0f} 分钟",
    ]
    taxi_cost = str(route.get("taxi_cost", "")).strip()
    if taxi_cost and taxi_cost not in ("0", ""):
        lines.append(f"打车约 {taxi_cost} 元（高德按通用计价模型估算，未套用当地运价文件，可能有出入）")
    else:
        lines.append("高德没有给出打车估价（距离过近时为 0），这一项按「待确认」处理")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. hotel_options —— 只挂给 researcher
# ---------------------------------------------------------------------------

#: 按商圈+档位搜酒店时返回几条。命中数常有三四十家，全灌进群聊没有意义——
#: researcher 要的是「这一带有哪些连锁、什么档位」，看前几条足够。
POI_MAX_RESULTS = 8

#: 住宿服务的 POI 类型码。不限定的话，搜「春熙路 快捷酒店」会把餐馆也带回来。
POI_TYPE_LODGING = "100000"


def hotel_options(arguments: str) -> str:
    """按商圈和档位列出酒店，给出店名、所属区、档位分类与评分。

    用它回答「这一带有什么酒店、哪家是哪一档」。**它不返回房价**：酒店实价按日期和
    房型动态生成、且在登录墙后面，任何接口都查不到。价格靠用户拍板，或者按搜索到的
    商圈价位区间给——别指望这个工具。

    Args:
        arguments: JSON 字符串，必填 city，另需 keywords（商圈/地名 + 档位词）。
                   例如 {"city": "成都", "keywords": "春熙路 快捷酒店"}
                   档位词可用：快捷酒店 / 经济型 / 商务酒店 / 舒适型 / 高档型 / 民宿
    """
    import json

    # 先校参数、后查 key，理由同 taxi_fare（自测要能在没有 key 的机器上离线测守卫）
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return f'参数不是合法 JSON：{exc}。应形如 {{"city":"成都","keywords":"春熙路 快捷酒店"}}'
    if not isinstance(args, dict):
        return '参数应是一个 JSON 对象，形如 {"city":"成都","keywords":"春熙路 快捷酒店"}'

    city = str(args.get("city", "")).strip()
    keywords = str(args.get("keywords", "")).strip()
    if not city or not keywords:
        return '缺少 city 或 keywords。应形如 {"city":"成都","keywords":"春熙路 快捷酒店"}'

    api_key = load_key(ENV_AMAP)
    if not api_key:
        return "错误：未配置高德 key（环境变量 AMAP_API_KEY 或 keys.py）"

    data, err = _amap_get(
        "/place/text",
        {
            "keywords": keywords,
            "region": city,
            "city_limit": "true",
            "types": POI_TYPE_LODGING,
            "page_size": str(POI_MAX_RESULTS),
            "show_fields": "business",
        },
        api_key,
        base=AMAP_POI_BASE,
    )
    if err:
        return f"搜酒店失败：{err}"

    pois = data.get("pois") or []
    if not pois:
        return f"高德在「{city} {keywords}」没搜到住宿，换个商圈名或去掉档位词再试。"

    # 不报 count：v5 的 count 就是本次返回条数（被 page_size 截断），不是命中总数。
    # 报成「命中 N 家」会让模型以为这一带只有 N 家。
    lines = [f"{city} {keywords}：列出 {len(pois)} 家（高德按相关度排序，可能还有更多）"]
    for i, poi in enumerate(pois, 1):
        biz = poi.get("business") or {}
        tier = biz.get("keytag") or poi.get("type", "").split(";")[-1] or "档位未标"
        rating = biz.get("rating") or "无评分"
        address = poi.get("address") or ""
        # address 有时是 "锦江区xxx" 这种带区名的形式，为空的就退回行政区
        where = address if isinstance(address, str) and address.strip() else poi.get("adname", "")
        lines.append(
            f"{i}. {poi.get('name', '(无名)')}｜{poi.get('adname', '')}｜{tier}｜{rating} 分\n"
            f"   {where}"
        )
    lines.append("以上只有档位和位置，**没有房价**——酒店实价查不到，价格要用户拍板或按商圈区间估。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. calculator —— 只挂给 critic
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
# 6. 注册表
# ---------------------------------------------------------------------------

ALL_TOOLS: dict[str, Callable[[str], str]] = {
    "search": search,
    "fetch_page": fetch_page,
    "taxi_fare": taxi_fare,
    "hotel_options": hotel_options,
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
