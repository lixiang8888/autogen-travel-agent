# -*- coding: utf-8 -*-
"""
ledger.py —— 群聊之外的状态：素材台账 + 无进展判据
================================================================================

**这个文件不认识 AutoGen，也不认识 persona。** 它只吃字符串和消息对象，吐数据。
分层理由同 persona.py：改判据不用碰组队，换拓扑不用碰判据。

**为什么需要它。** 群聊的上下文是**只追加的消息序列**，没有撤销。于是：

  - researcher 把港台 OTA 的商圈均价当房价采信后，即使重查，旧素材仍留在
    thread 里，planner 和 critic 分不清该信哪一条（architecture.md 第 7 节第 13 条）
  - 唯一的兜底是 MaxMessageTermination(15) 和 MAX_HANDOFFS，撞到都算失败，
    但**分不清「在收敛」和「在原地绕圈」**

台账解决第一个：把「素材」从消息流里捞出来，变成一个**可作废、可重渲染**的对象。
无进展判据解决第二个：给出几条文末没有的确定性信号。

**台账靠解析 researcher 的 Markdown 填充，不加新工具。** tools.py 里的工具全是
无状态纯函数（「每次返回同一批函数对象——它们是纯函数，无状态」），加一个有状态
工具会破坏那个设计；而 researcher 的 format_contract 本来就强制了刚性行格式
`【类别】内容 价格（来源：URL）`，文档明写这是「给下游 agent 读的接口」。

**两条硬约束：**

1. `Board.absorb` 必须**幂等**。真 AutoGen 的 `TaskResult.messages` 是**本次 run 的
   增量**，而离线自测的 `_FakeTeam` 给的是**累计列表**——两种都要能正确处理，
   幂等是唯一稳妥的做法。
2. 解析不出来的行走 `Board.unparsed`，**不许静默丢弃**，并且照样出现在 `render()`
   里。漏掉一行素材等于让 critic 的幻觉比对失去依据。这条对齐 researcher.md 里
   「把『缺』从一个被掩盖的状态，变成一个被暴露的状态」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 四类素材，顺序与 researcher 的 format_contract 一致。
CATEGORIES: tuple[str, ...] = ("交通", "住宿", "景点", "市内")


# ---------------------------------------------------------------------------
# 1. 解析：把契约行捞成 Material
# ---------------------------------------------------------------------------

#: 匹配契约里那条刚性行格式：`【类别】内容 价格（来源：URL）`。
#:
#: body 用**贪婪**匹配。若正文里出现别的括号，贪婪会让它切在**最后一个**
#: 「（来源：」上，比非贪婪稳。source 排掉 `）`，因为来源栏不会包含它。
_MATERIAL_RE = re.compile(r"^【(交通|住宿|景点|市内)】(.+)（来源：([^）]+)）\s*$")


@dataclass(frozen=True)
class Material:
    """一条素材。**故意不把「内容」和「价格」拆开。**

    契约行 `北京⇄成都 高铁二等座 约750/人 单程` 里两者的边界是模糊的，硬切会
    出错；而台账只负责版本与作废、不做算术，所以整段留作 body 更稳。
    """

    category: str
    body: str
    source: str

    def line(self) -> str:
        """还原成契约里的那一行。"""
        return f"【{self.category}】{self.body}（来源：{self.source}）"


def parse_materials(text: str) -> tuple[list[Material], list[str]]:
    """把一段 researcher 输出解析成（素材, 没解析出来的行）。

    第二项是**要原样保留**的：模型偶尔会跑出契约外的行，那些行里可能有数字。
    丢一行，critic 的幻觉比对就少一条依据，而且没人会发现。
    """
    materials: list[Material] = []
    unparsed: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        matched = _MATERIAL_RE.match(line)
        if matched is None:
            unparsed.append(line)
            continue
        category, body, source = matched.groups()
        materials.append(Material(category=category, body=body.strip(), source=source.strip()))
    return materials, unparsed


# ---------------------------------------------------------------------------
# 2. 台账
# ---------------------------------------------------------------------------

@dataclass
class Board:
    """素材台账：四类条目 + 未解析行 + 已作废条目。

    `version` 每有新条目或作废动作就 +1，用来判断「台账变了没有」——注入作废声明
    的前提是它真的变了，否则每条续跑消息都会挂一坨没用的附件。
    """

    version: int = 0
    items: dict[str, list[Material]] = field(
        default_factory=lambda: {c: [] for c in CATEGORIES}
    )
    unparsed: list[str] = field(default_factory=list)
    stale: list[Material] = field(default_factory=list)

    # -- 写入 ---------------------------------------------------------------

    def absorb(self, text: str) -> bool:
        """把一段 researcher 输出并进台账。返回**台账是否真的变了**。

        幂等：按 `(类别, 正文, 来源)` 去重，同一段喂两次结果相同。所以可以放心地
        对每个 run 的 `TaskResult.messages` 整段调用，不必操心它是增量还是累计。

        只对**当前**条目去重，不对 `stale` 去重——researcher 重查后若把同一条又
        报了一遍，说明它是真的又查了一次，那就该重新采信。若它在原地复读，
        会被 detect_stall 抓成「重复搜索」交给人，而不是在这里被静默吃掉。
        """
        materials, unparsed = parse_materials(text)
        changed = False

        for item in materials:
            bucket = self.items[item.category]
            if item not in bucket:
                bucket.append(item)
                changed = True

        for line in unparsed:
            if line not in self.unparsed:
                self.unparsed.append(line)
                changed = True

        if changed:
            self.version += 1
        return changed

    def supersede(self, category: str) -> list[Material]:
        """作废某一类的全部当前条目。返回被作废的那些（供注入声明引用）。

        类别非法时返回空列表——**不抛异常**：这个方法的输入来自模型输出，
        不可信，抛异常会把整条流程打断，而「没作废任何东西」是可以安全降级的。
        """
        if category not in self.items:
            return []
        removed = self.items[category]
        if not removed:
            return []
        self.items[category] = []
        self.stale.extend(removed)
        self.version += 1
        return removed

    # -- 读出 ---------------------------------------------------------------

    def render(self) -> str:
        """渲染成最新台账全文，供注入回群聊。"""
        lines = [f"【素材台账 · 第 {self.version} 版】"]
        for category in CATEGORIES:
            for item in self.items[category]:
                lines.append(item.line())
        if self.unparsed:
            lines.append("以下行没能解析成素材，原样附上：")
            lines.extend(self.unparsed)
        if self.stale:
            lines.append(f"（另有 {len(self.stale)} 条历史素材已作废，不再列出）")
        return "\n".join(lines)

    def corrective_note(self, category: str, keyword: str = "") -> str:
        """作废声明：告诉全群「这一类旧素材不算数了，重新查」。

        刻意**逐条列出**被作废的条目，而不是只说「住宿类作废」——planner 是照抄
        数字的那一个，critic 是拿素材做幻觉比对的那一个，把原话摊出来，它们才
        认得出自己手里那条是废的。
        """
        just_removed = [m for m in self.stale if m.category == category]
        lines = [
            f"【素材台账更新 · 第 {self.version} 版】",
            f"下列 {category} 类素材已作废，不得再作为报价或行程依据：",
        ]
        lines.extend(f"  {item.line()}" for item in just_removed)
        if keyword:
            lines.append(f"重查要求：{keyword}")
        lines.append(f"其余三类素材不变。以本轮 researcher 新查到的{category}素材为准。")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. 无进展判据
# ---------------------------------------------------------------------------

#: 连续两轮的 URL 重合到这个程度，判定为「在重复搜索」。
SEARCH_OVERLAP_THRESHOLD = 0.8

_URL_RE = re.compile(r"https?://[^\s）)】\]]+")
#: critic 的**问题清单形态**：契约写的是「问题 N 条：」。
#:
#: 判据必须先认这个形态，才去看内容。理由见下面 _problem_stall 的注释。
_PROBLEM_HEAD_RE = re.compile(r"^问题\s*\d+\s*条\s*[:：]")
_PROBLEM_ITEM_RE = re.compile(r"^\d+\s*[.、]\s*(.+)$")
_TABLE_ROW_RE = re.compile(r"^\|.*\|$")


@dataclass(frozen=True)
class Stall:
    """一条「在原地绕圈」的证据。`detail` 是给人看的，会直接印到终端。"""

    kind: str
    detail: str

    def render(self) -> str:
        return f"⚠️ 无进展（{self.kind}）：\n{self.detail}\n\n需要你给一个带动作的指示，或者换掉其中一环。"


def _texts_by_source(messages) -> dict[str, list[str]]:
    """按 source 归拢**纯文本**消息。工具调用事件（content 是列表）跳过。"""
    grouped: dict[str, list[str]] = {}
    for message in messages or ():
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        grouped.setdefault(getattr(message, "source", ""), []).append(content.strip())
    return grouped


def _urls(text: str) -> set[str]:
    return {u.rstrip(".,;，。；") for u in _URL_RE.findall(text)}


def _problem_items(text: str) -> list[str] | None:
    """取问题清单里的条目；不是问题清单形态就返回 None。

    **形态判据是这里的关键。** 离线自测 [9] 用的假 critic 消息正文是
    `"问题 1"`、`"问题 2"`……（`_handoff_batch(f"问题 {i}")`），跟真清单长得有点像
    但不是清单。判据一松，[9] 的段数断言立刻崩——那几条断言锁的是
    「--reply 用完绝不回退到 ask」这条规矩。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or _PROBLEM_HEAD_RE.match(lines[0]) is None:
        return None
    items: list[str] = []
    for line in lines[1:]:
        matched = _PROBLEM_ITEM_RE.match(line)
        if matched:
            items.append(re.sub(r"\s+", "", matched.group(1)))
    return items


def _table_rows(text: str) -> list[str] | None:
    rows = [ln.strip() for ln in text.splitlines() if _TABLE_ROW_RE.match(ln.strip())]
    return rows if len(rows) >= 2 else None


def _search_stall(texts: list[str]) -> Stall | None:
    if len(texts) < 2:
        return None
    prev, curr = _urls(texts[-2]), _urls(texts[-1])
    if len(prev) < 2 or len(curr) < 2:
        return None
    overlap = len(prev & curr) / len(prev | curr)
    if overlap < SEARCH_OVERLAP_THRESHOLD:
        return None
    repeated = "、".join(sorted(prev & curr)[:3])
    return Stall(
        kind="重复搜索",
        detail=f"researcher 连续两轮搜到的网址重合 {overlap:.0%}：{repeated}",
    )


def _problem_stall(texts: list[str]) -> Stall | None:
    lists = [items for items in (_problem_items(t) for t in texts) if items]
    if len(lists) < 2:
        return None
    prev, curr = lists[-2], lists[-1]
    if not prev or not curr:
        return None
    kept = [item for item in curr if item in set(prev)]
    if len(kept) != len(curr):
        return None
    return Stall(
        kind="重复问题",
        detail=f"critic 连续两轮报的是同 {len(curr)} 条问题，第一条仍是：{curr[0][:60]}",
    )


def _table_stall(texts: list[str]) -> Stall | None:
    tables = [rows for rows in (_table_rows(t) for t in texts) if rows]
    if len(tables) < 2:
        return None
    if tables[-2] != tables[-1]:
        return None
    return Stall(
        kind="空转表格",
        detail=f"planner 连续两轮输出逐字相同的行程表（{len(tables[-1])} 行）",
    )


def detect_stall(messages, *, researcher: str, planner: str, critic: str) -> Stall | None:
    """在**同一个 run 累积的消息**里找「连续两轮没进展」的证据。找不到返回 None。

    三条判据都是确定性的字符串比对，不调 LLM——质量监控如果自己也要一次模型
    调用，那它本身就成了新的不确定源。

    角色名由调用方传入（main.py 从 persona 的 spec 上取），这个文件不认识它们。

    **判据按可信度排序，不是按角色顺序。** critic 连续两轮报同一批问题 = 这个回路
    确实没在收敛；planner 连续两轮输出同一张表 = 它确实没在干活。而「researcher 两轮
    搜到同一批网址」是三者里最弱的一条——它的契约允许「后续轮次只补差的那部分」，
    模型也常把整份清单重发一遍，重合未必等于卡住。弱的排后面，免得盖住强的。
    """
    grouped = _texts_by_source(messages)
    for probe, source in (
        (_problem_stall, critic),
        (_table_stall, planner),
        (_search_stall, researcher),
    ):
        found = probe(grouped.get(source, []))
        if found is not None:
            return found
    return None
