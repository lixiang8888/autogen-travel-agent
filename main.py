# -*- coding: utf-8 -*-
"""
main.py —— 组队 + 跑：拓扑、终止条件、入口
============================================

改拓扑、调轮数上限、换 agent 说话顺序，只动这个文件。这里不认识 persona.py 里
那些字符串的内容——这是刻意的分层。

**拓扑**：RoundRobinGroupChat，固定顺序 researcher → planner → critic → 循环。
起步不用 SelectorGroupChat：先要可预测，再要聪明。等整个流程跑通了、知道「正常的
对话长什么样」了，再换拓扑做对照实验（见 docs/architecture.md 第 8 节）。

**终止条件**：四重，缺一不可。
    ExactTextTermination("APPROVED", "critic")  critic 认可（精确匹配，非子串）
    MarkedLineTermination(需重查：, "critic")   critic 指名某类素材不可信，要重查
    HandoffTermination(target="user")           需要人拍板，交回控制权
    MaxMessageTermination(15)                   兜底保险丝——LLM 不一定老实

**素材台账**（ledger.py）是群聊之外的状态。researcher 查错时，光能重查没用——旧
素材还留在只追加的消息流里，planner 和 critic 分不清该信哪一条。台账把素材捞出来，
变成一个**可作废、可重渲染**的对象。重查有两个入口，共用同一套「作废 → 重渲染 →
注入」：critic 自动指名（写「需重查：<类别> <关键词>」），或用户拍板时手动指名。
另外台账还供无进展判据（detect_stall）用，撞上「原地绕圈」时把证据摆给用户看。

**user 不在 participants 里。** UserProxyAgent 的默认 input_func 读控制台，一旦进队，
RoundRobin 每转到它就会阻塞整个 team，官方文档说这会让 team 变成「无法保存或恢复」
的状态。改用 HandoffTermination 后，user 只是 HandoffMessage 里的一个字符串标签。
详见 docs/agents/user.md §2.1。

**续跑必须用 HandoffMessage**，不能直接传字符串——否则报
`ValueError: The existing handoff target user is not one of the participants`。

跑法：
    python main.py "一句话需求"    # 完整流程（不传需求用内置示例）
    python main.py --selftest     # 离线自测，不需要 key、不联网
    python main.py --stage 1..4   # 调试用：逐层排查是哪一层坏的，见 README
    python launcher.py            # 网页界面，见 README「网页版」

**这个文件不认识 launcher.py。** 反向依赖是不存在的（自测第 [10] 节守着），
所以把 launcher.py 删掉，`--selftest` 照样绿。
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from typing import Callable

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff, TaskResult, TerminatedException, TerminationCondition
from autogen_agentchat.conditions import ExternalTermination, HandoffTermination, MaxMessageTermination
from autogen_agentchat.messages import HandoffMessage, StopMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from ledger import Board, detect_stall
from llm import build_model
from persona import CRITIC, PLANNER, RESEARCHER, AgentSpec, build_system_message
from tools import build_tools

DEFAULT_TASK = "帮我规划国庆去成都玩三天，两个人，预算 4000，北京出发。"
MAX_MESSAGES = 15
STAGE3_MAX_TURNS = 4          # 阶段 3/4 用硬停，先不设终止条件
#: 最多问你几次拍板。
#:
#: 原设 3，实测在旅行规划这种多自由度任务上不够：critic 每轮挑出的都是**真的**
#: 新问题（实测 4→3→4→5 条，逐条验证过），3 个回答解决不完全部取舍，于是永远
#: 差一轮，看着像"不收敛"，其实是轮次预算不够。放宽到 6 再观察。
#: 若 6 次仍不收敛，那才是 critic 的尺度问题——届时应要求它把问题分级
#: （必须解决 / 可选优化），只对前者拦人，而不是继续加轮次。
MAX_HANDOFFS = 6

#: 一轮里模型最多可以「调工具 → 看结果 → 再调」几次。
#:
#: **AutoGen 的默认值是 1，那是个会静默毁掉整个流程的坑。** 看
#: AssistantAgent._process_model_result 的循环：只要模型第一轮返回的是工具调用
#: （content 不是字符串），执行完工具后 `loop_iteration == max_tool_iterations - 1`
#: 立刻成立并 break，然后**自动生成一个 ToolCallSummaryMessage 收尾**。
#: 加上 reflect_on_tool_use 默认也是 False（没有事后反思），结果是：
#:
#:   - 模型永远看不到自己的工具返回了什么（搜索到的、算出来的）
#:   - 于是永远写不出 persona 要求的格式化输出
#:   - researcher 的「素材清单」变成搜索结果原样拼接
#:   - critic 调完 calculator 就没下文了，永远不会说 APPROVED
#:
#: 只有 planner 看上去正常——因为它没有工具，content 直接就是字符串。
#: 见 docs/architecture.md 第 7 节第 2 条。
MAX_TOOL_ITERATIONS = 5

_SEP = "=" * 72

#: 拍板时给用户的那句提示。
#:
#: 独立成常量是因为图形启动器要拿它当输入框的标签——总不能让它去 strip
#: `input()` 提示串尾巴上那个 "> "。
ASK_HINT = "你的决定（一句话，带一个数字或一个动作）"
_ASK_PROMPT = f"\n{ASK_HINT}> "


# ---------------------------------------------------------------------------
# 1. 造 agent 和 team
# ---------------------------------------------------------------------------

class ExactTextTermination(TerminationCondition):
    """只在「某个 agent 发出的**一条消息恰好等于**该文本」时终止。

    为什么不用 `TextMentionTermination("APPROVED")`：那个按**子串**匹配任意消息，
    而实测 critic 会把思考写进消息正文（推理泄露）。只要那段思考里出现 APPROVED
    这个子串——**哪怕是否定句**（「我还不能输出 APPROVED」）——整个对话就会提前
    终止，把一份没审完的行程当通过交付出去。

    实测见过 critic 的消息正文是这种形态：

        All checks pass: 3730 ≤ 4000 … No time conflicts, no geographic detour.
        Output APPROVED.

    那次侥幸蒙对了（泄露的思考恰好也是通过），但方向反过来就是静默的错交付。

    这条契约 docs/agents/critic.md §8.2 本来就写着（「该行必须只有 APPROVED 这八个
    字符」），这里让代码去强制执行它，而不是指望模型自觉。见 docs/architecture.md 第 7 节第 3 条。
    """

    def __init__(self, text: str, source: str) -> None:
        self._text = text
        self._source = source
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")
        for message in messages:
            if getattr(message, "source", None) != self._source:
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip() == self._text:
                self._terminated = True
                return StopMessage(
                    content=f"'{self._text}' from {self._source} (exact match)",
                    source="ExactTextTermination",
                )
        return None

    async def reset(self) -> None:
        self._terminated = False


#: critic 用来要求重查素材的标记。台账机制的另一半在 ledger.py。
RESEARCH_MARKER = "需重查："

#: 只有「整行以标记开头、且类别合法」才算数。**这是防推理泄露的装置。**
#:
#: 跟 ExactTextTermination 是同一个教训（架构第 7 节第 3 条）：critic 会把思考写进
#: 正文，子串匹配会被自己的思考误触发。所以这里要求两件事同时成立——
#:   1. 以「需重查：」**开头**（泄露出的「不需要重查：…」不以它开头）
#:   2. 后面跟着四类之一（光写「需重查：春熙路价位」缺类别，不算数）
#: 契约（critic.md §8.2）把类别写成了硬要求，正是为了让这个判据能立住。
_RESEARCH_RE = re.compile(r"^需重查：\s*(交通|住宿|景点|市内)\s*(\S.*)?$")

#: 用户手动指名重查时用的宽松版：允许写成「重查：住宿 …」。
#:
#: 上面那条严格判据是**专门用来防 critic 的推理泄露**的，人不在这个威胁模型里
#: ——真人不会把思考过程写进正文再让它意外成行。要求人一字不差地打「需重查：」
#: 只会让这个入口没人用。两条入口共用同一个解析函数，只是喂不同的正则。
_USER_RESEARCH_RE = re.compile(r"^(?:需)?重查：\s*(交通|住宿|景点|市内)\s*(\S.*)?$")


class MarkedLineTermination(TerminationCondition):
    """某个 agent 发出的消息里，有**一行**匹配给定正则时终止。

    与 `ExactTextTermination` 的区别：那个要求整条消息恰好等于某文本，这个只要求
    某一行匹配。用在「critic 在问题清单后面另起一行写重查要求」这个场景上——它
    那一轮的消息正文还有问题清单，不可能整条相等。
    """

    def __init__(self, pattern: re.Pattern[str], source: str) -> None:
        self._pattern = pattern
        self._source = source
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")
        for message in messages:
            if getattr(message, "source", None) != self._source:
                continue
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                continue
            for line in content.splitlines():
                if self._pattern.match(line.strip()):
                    self._terminated = True
                    return StopMessage(
                        content=f"{RESEARCH_MARKER!r} in a line from {self._source}",
                        source="MarkedLineTermination",
                    )
        return None

    async def reset(self) -> None:
        self._terminated = False


def build_termination():
    """四重终止条件。`|` 不要写成 `&`——`&` 要两个同时满足，会一直跑到上限。"""
    return (
        ExactTextTermination("APPROVED", source=CRITIC.name)   # critic 认可，须精确匹配
        | MarkedLineTermination(_RESEARCH_RE, source=CRITIC.name)  # critic 要求重查素材
        | HandoffTermination(target="user")                   # 需要人拍板，交回控制权
        | MaxMessageTermination(MAX_MESSAGES)                 # 兜底保险丝
    )


def build_agent(spec: AgentSpec, model) -> AssistantAgent:
    """把一个 AgentSpec 变成一个 AutoGen 的 AssistantAgent。"""
    kwargs: dict = {
        "name": spec.name,
        "system_message": build_system_message(spec),
        "model_client": model,
        "tools": build_tools(spec.tool_names),
        # 下面两个默认值都必须显式改掉，理由见 MAX_TOOL_ITERATIONS 的注释
        "max_tool_iterations": MAX_TOOL_ITERATIONS,
        "reflect_on_tool_use": True,
    }
    if spec.handoffs:
        # Handoff 是通过模型生成工具调用来触发的，所以要求模型支持 function calling
        # ——llm.py 里 model_info 的 function_calling: True 就是这个前提。
        kwargs["handoffs"] = [
            Handoff(target=target, message="需要用户拍板") for target in spec.handoffs
        ]
    return AssistantAgent(**kwargs)


def build_team(model, specs, *, termination=None, max_turns: int | None = None):
    """按 specs 的顺序组一个 RoundRobinGroupChat。"""
    return RoundRobinGroupChat(
        [build_agent(spec, model) for spec in specs],
        termination_condition=termination,
        max_turns=max_turns,
    )


# ---------------------------------------------------------------------------
# 2. 跑与打印（+ 回调接缝：CLI 打印与图形界面共用同一套格式化）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormattedMessage:
    """一条消息格式化之后的样子。

    kind 只有两种：
        "text"  模型或用户写的正文
        "tool"  工具调用与返回值（图形界面里会渲染成等宽、缩进、退到次要色）
    """

    source: str
    text: str
    kind: str


def format_message(message) -> FormattedMessage | None:
    """把一条消息格式化；没有可显示内容时返回 None。

    **纯函数，不打印。** 抽出来是为了让终端和图形界面共用同一套格式化逻辑：
    `_print_message` 拿它去 print，图形界面拿它去渲染。否则图形界面只能去
    正则解析 `[source]` 那一行——那是「解析自己的终端输出」的坏味道。

    content 可能是 str（TextMessage），也可能是**对象列表**——工具调用类事件
    （ToolCallRequestEvent / ToolCallExecutionEvent）就是后者。早期版本直接
    `.strip()`，一遇到工具调用就 AttributeError，阶段 3/4/5 全崩。
    """
    source = getattr(message, "source", "?")
    content = getattr(message, "content", "")

    if isinstance(content, list):
        kind = "tool"
        lines: list[str] = []
        for item in content:
            name = getattr(item, "name", None) or type(item).__name__
            args = getattr(item, "arguments", None)
            result = getattr(item, "content", None)
            if args is not None:                    # 模型发出的调用
                lines.append(f"  → {name}({args})")
            elif result is not None:                # 工具返回的结果
                body = str(result).strip()
                if len(body) > 400:
                    body = f"{body[:400]}…（共 {len(body)} 字符）"
                lines.append(f"  ← {name} 返回：{body}")
            else:
                lines.append(f"  · {name}")
        text = "\n".join(lines)
    else:
        kind = "text"
        text = (content or "").strip()

    if not text:
        return None
    return FormattedMessage(source=source, text=text, kind=kind)


def _print_message(message) -> None:
    """打印一条消息。

    **签名不许动**——selftest 第 [7] 节以单参数调它，守着「工具调用事件不能崩」
    这条回归。要换出口请用 `RunHooks.on_message`，不要给它加参数。
    """
    formatted = format_message(message)
    if formatted is None:
        return
    print(f"\n{_SEP}\n[{formatted.source}]\n{_SEP}")
    print(formatted.text)


def _print_notice(text: str, kind: str = "plain") -> None:
    """终端版通知出口。**所有终端版式都锁在这一个函数里。**

    kind 的六种取值一一对应重构前的六处 print，逐字节等价。别想着合并——
    `note` 和 `plain` 的区别就是前导换行，合并了输出就变了：

        banner    行首换行 + `=` 框    「该你拍板了（第 N 次）」
        final     行首换行 + `=` 框    「结束原因：…」
        note      行首换行             回答回显、预设用完、没有拿到输入
        question  原样                  critic 的问题清单（人必须读的正文）
        plain     原样                 「（空输入，停止。）」
        warning   原样                 撞到轮数上限的警告
    """
    if kind in ("banner", "final"):
        print(f"\n{_SEP}\n{text}\n{_SEP}")
    elif kind == "note":
        print(f"\n{text}")
    else:                                   # question / plain / warning
        print(text)


def _console_ask(prompt: str) -> str | None:
    """终端版拍板输入。返回 None 表示拿不到输入（EOF 或 Ctrl-C）。"""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        # stdin 是空的（非交互跑）或用户按了 Ctrl-C/Ctrl-D。
        # 这里不该糊一个 traceback——没输入就干净地停在当前进度。
        return None


@dataclass
class RunHooks:
    """一次 run 的出口与开关。默认值 = 终端行为；图形启动器传自己的实现。

    `ask` 是**同步**的，别改成 async。包成 `asyncio.to_thread(input, ...)` 之后：
      1. Python 只在主线程处理信号，3.11+ 的 Runner 会把第一次 Ctrl-C 变成
         `main_task.cancel()`，下面的 `except KeyboardInterrupt` 就接不到了，
         行为从「干净停在当前进度」变成「被取消」；
      2. `asyncio.run` 收尾时 `shutdown_default_executor()` 会 join 那个仍阻塞在
         stdin 上的线程（3.12+ 默认无限等），**进程退出时会挂死**。
    同步版把 `input()` 留在原来那个线程、那个位置，catch 原封不动。
    图形界面那边同样是把 worker 线程阻塞在队列上等人回话，形态与本函数一致。
    """

    on_message: Callable[[object], None] = _print_message
    on_notice: Callable[[str, str], None] = _print_notice
    ask: Callable[[str], str | None] = _console_ask
    #: 外部停止开关。None = 不提供停止（终端路径的默认情形，对象图与重构前完全相同）。
    stop: ExternalTermination | None = None


async def run_and_print(team, task, *, on_message=None) -> TaskResult:
    """流式跑，实时报告每条消息，返回 TaskResult。

    用 run_stream 而不是 run，是因为一次完整对话要十几轮 LLM 调用，
    等全部跑完再打印会让人以为卡死了。

    Args:
        on_message: 收到**原始 message 对象**的回调——不是格式化好的字符串，
            调用方自己决定怎么渲染。不给就是打印到终端。
    """
    report = on_message or _print_message
    result: TaskResult | None = None
    async for message in team.run_stream(task=task):
        if isinstance(message, TaskResult):
            result = message
        else:
            report(message)
    assert result is not None, "run_stream 没有返回 TaskResult"
    return result


def stopped_for_user(result: TaskResult) -> bool:
    """判断这次停下是不是在等用户拍板。"""
    if not result.messages:
        return False
    last = result.messages[-1]
    return isinstance(last, HandoffMessage) and getattr(last, "target", "") == "user"


def last_content(result: TaskResult) -> str:
    """取最后一条**文本**消息。跳过 content 是列表的工具调用事件。"""
    for message in reversed(result.messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def last_substantive_content(result: TaskResult) -> str:
    """取触发 handoff **之前**那段正文——通常是 critic 的问题清单。

    为什么要单独一个函数：`Handoff` 的工具是个**零参数**函数，源码就是

        def _handoff_tool() -> str:
            return self.message

    ——它只原样返回 `Handoff(message=...)` 里那句固定的话。实测 critic 调
    `transfer_to_user({})` 时参数确实是空的，所以它想问什么**只能在正文里写**。

    实测它有时会跳过正文直接调工具（有一轮 6/6 次都没写），于是用户在终端只看到
    「需要用户拍板」四个字，完全不知道要决定什么——人机回路等于白设。
    这里往前翻，把那段正文捞出来给用户看。
    """
    for message in reversed(result.messages):
        if isinstance(message, HandoffMessage):
            continue                      # 跳过 handoff 本身，要的是它前面那段
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


@dataclass(frozen=True)
class ResearchRequest:
    """一句「重查」要求的两个字段。类别决定作废哪一类，关键词转给 researcher。"""

    category: str
    keyword: str


def parse_research_line(text: str, pattern: re.Pattern[str] = _RESEARCH_RE) -> ResearchRequest | None:
    """在文本里找「需重查：<类别> <关键词>」那一行。找不到返回 None。

    自动入口（critic 写）传默认的严格版，手动入口（用户写）传 `_USER_RESEARCH_RE`。
    两条入口共用这一个解析器、只是判据宽严不同——解析完之后的「作废 → 重渲染 →
    注入」完全同一条路径，不会出现「机器说的算数、人说的不算数」。
    """
    for line in (text or "").splitlines():
        matched = pattern.match(line.strip())
        if matched:
            return ResearchRequest(
                category=matched.group(1),
                keyword=(matched.group(2) or "").strip(),
            )
    return None


def research_request(result: TaskResult) -> ResearchRequest | None:
    """这一轮 critic 有没有要求重查素材？

    **只看最后一条有正文的消息，且必须出自 critic。** 往前翻会把上一轮的旧要求
    重新捞出来，变成一个永不退出的重查循环。
    """
    if not result.messages:
        return None
    for message in reversed(result.messages):
        if isinstance(message, HandoffMessage):
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if getattr(message, "source", "") != CRITIC.name:
            return None                   # 最后说话的不是 critic，它的话不作数
        return parse_research_line(content)
    return None


def absorb_into(board: Board | None, messages) -> None:
    """把 researcher 的正文并进台账。没有台账、或这批里没有 researcher 就跳过。

    **入参是「这次真正新进来的消息」，不是整个 TaskResult。** 调用方先过一遍
    `remember` 拿到新消息再喂进来——真 AutoGen 给的是本次 run 的增量，而自测的
    `_FakeTeam` 给的是**累计列表**，后者会把同一段素材反复喂。台账若重复吸收，
    上一步刚作废的条目会被自己复活，作废机制静默失效。
    """
    if board is None:
        return
    for message in messages or ():
        if getattr(message, "source", "") != RESEARCHER.name:
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            board.absorb(content)


def remember(history: list, messages) -> list:
    """把消息按原对象去重后追加进历史，**返回这次真正新加进来的那些**。

    两件事共用这一次遍历：

    - 无进展判据要的是**跨段**的连续两条同源消息，而 `TaskResult.messages` 是每段
      的增量——只看一段永远凑不齐两条，所以要跨段累积。
    - 但自测的 `_FakeTeam` 给的是累计列表，直接追加会把同一条算两遍，把「自己和
      自己比」误判成重复；台账那边更糟，会把刚作废的条目复活。按对象身份去重同时
      满足这两种形态（history 持有引用，id 不会被回收复用）。
    """
    seen = {id(m) for m in history}
    fresh: list = []
    for message in messages or ():
        if id(message) not in seen:
            seen.add(id(message))
            history.append(message)
            fresh.append(message)
    return fresh


def find_stall(history: list):
    """在跨段历史里找「连续两轮没进展」的证据。角色名从 persona 取，ledger 不认识它们。"""
    return detect_stall(
        history,
        researcher=RESEARCHER.name,
        planner=PLANNER.name,
        critic=CRITIC.name,
    )


def supersede_notice(board: Board, category: str) -> str:
    """作废一类的素材并返回给用户看的那句话。"""
    removed = board.supersede(category)
    return f"台账：{category} 类 {len(removed)} 条素材作废（现为第 {board.version} 版）"


def resolve_answer(queue, scripted, ask, handoffs, notice) -> str | None:
    """拿一个拍板答案。返回 None 表示该停下来了。

    **分支顺序是有意的，别调**：预设队列 > 预设用完就停 > 才轮到问人。

    「预设用完就停」锁死的是「绝不回退到 input()」这条规矩——后台/管道场景下
    stdin 可能是「开着但不给数据」，input() 不抛 EOFError 而是永久阻塞，最后被
    timeout 杀掉（实测撞过一次，白烧 30 分钟才看出来）。见 docs/architecture.md 第 7 节第 8 条。
    """
    if queue:
        answer = queue.pop(0).strip()
        notice(f"[--reply 第 {handoffs} 个回答] {answer}", "note")
    elif scripted:
        notice("（预设回答已用完，停在当前进度。）", "note")
        return None
    else:
        raw = ask(_ASK_PROMPT)
        if raw is None:
            notice("（没有拿到输入，停在当前进度。）", "note")
            return None
        answer = raw.strip()

    if not answer:
        notice("（空输入，停止。）", "plain")
        return None
    return answer


# ---------------------------------------------------------------------------
# 3. 五个阶段（调试用：逐层排查是哪一层坏的，日常不传 --stage）
# ---------------------------------------------------------------------------

async def stage1_connectivity() -> None:
    """阶段 1 · 环境与模型连通性。

    验证点：不报错，能拿到回复。特别确认没有 model_info 相关的报错
    ——这是 DeepSeek 接 AutoGen 的第一个坎。
    """
    model = build_model()
    agent = AssistantAgent(name="ping", model_client=model)
    result = await agent.run(task="用一句话回答：你是什么模型？")
    print(f"\n{_SEP}\n阶段 1 通过：模型有回复\n{_SEP}")
    print(result.messages[-1].content)
    await model.close()


async def stage2_researcher() -> None:
    """阶段 2 · 单 agent 带 search 跑通。

    验证点：输出里有**真实可点的网址**，不是编的。点开一个确认内容对得上
    ——这一步验证的是「工具真的调通了」，不是「模型假装调通了」。
    """
    model = build_model()
    agent = build_agent(RESEARCHER, model)
    result = await agent.run(
        task="帮我查一下成都大熊猫繁育研究基地的门票价格，以及成都市区地铁+打车的日均花费。"
    )
    await model.close()

    content = result.messages[-1].content or ""
    print(f"\n{_SEP}\nresearcher 的完整输出\n{_SEP}\n{content}\n")

    # 消息轨迹：能直接看出 search 有没有被真的调用
    print(f"{_SEP}\n消息轨迹\n{_SEP}")
    for msg in result.messages:
        name = type(msg).__name__
        src = getattr(msg, "source", "?")
        extra = ""
        calls = getattr(msg, "content", None)
        if isinstance(calls, list):                     # 模型发出的 tool call
            extra = "  工具调用: " + ", ".join(
                str(getattr(c, "name", c)) for c in calls)
        print(f"  {name:26} from {src}{extra}")

    print(f"\n{_SEP}\n阶段 2 验证点\n{_SEP}")
    if "搜索失败" in content or "未配置" in content:
        print("✗ search 被调用了，但搜索本身失败——原因见上面的输出。")
        print("  若提示认证失败，Tavily 的 key 是错的：正确的以 tvly- 开头。")
    else:
        # 别拿 'http' 当判据——报错信息里也含 api.tavily.com，会假通过。
        urls = [u for u in re.findall(r"https?://\S+", content)
                if "api.tavily.com" not in u]
        if urls:
            print(f"✓ 输出里有 {len(urls)} 条外部网址。点开至少一个，确认内容对得上。")
        else:
            print("✗ 输出里没有外部网址——工具可能没调通，回查 tools.py 与 Tavily key。")


async def stage3_two_agents() -> None:
    """阶段 3 · 两个 agent 进群聊（先不设终止条件，max_turns 硬停）。

    验证点：planner 的输出里引用了 researcher 查到的**具体内容**
    （具体的酒店区域、具体票价），而不是自己编了一套。
    如果 planner 在编，说明素材没进到它的上下文里。
    """
    model = build_model()
    team = build_team(model, (RESEARCHER, PLANNER), max_turns=STAGE3_MAX_TURNS)
    await run_and_print(team, DEFAULT_TASK)
    await model.close()


async def stage4_with_critic() -> None:
    """阶段 4 · 加 critic。

    验证点：critic 至少挑出**一个真问题**（时间冲突 / 绕路 / 预算超支 / 幻觉）。
    如果 critic 一直在说「行程很合理」——那是失败，不是成功。
    """
    model = build_model()
    team = build_team(model, (RESEARCHER, PLANNER, CRITIC), max_turns=STAGE3_MAX_TURNS)
    await run_and_print(team, DEFAULT_TASK)
    await model.close()


async def run_with_handoffs(team, task, replies=None, *, hooks=None, board=None) -> TaskResult:
    """跑一轮完整流程 + 人机回路。**只认 team 和 hooks，不认识 model。**

    验证点：连跑三次，每次都在合理轮数内自然结束（而不是撞到 15 轮上限）。
    撞上限说明 critic 不肯说 APPROVED，回去改它的 system_message。

    和 stage5_full 分成两层是为了**可测性**：stage5_full 第一行就 build_model()，
    一要 key 二要联网，于是这段最贵、最易错的回路在离线自测里覆盖率一直是零。
    切开之后一个假 team 就能把它整段跑穿（见 selftest 第 [9] 节）。

    Args:
        replies: 预设的拍板回答，第 N 次 handoff 用第 N 个。给完就停。
            一个都不给则每次调 hooks.ask。用于非交互跑（脚本化、CI、
            或者「用不同的回答测收敛性」）。
            **注意 `replies=[]` 不等于「没给」**——下面 scripted 判的是 `is not None`，
            传空列表会在第一次 handoff 就「预设用完」停下。不想预设就传 None。
        hooks: 出口回调。不给就是打印到终端、读键盘。
        board: 素材台账。不给就现建一个——终端单跑用不着从外面传，自测想事后
            检查台账内容时才需要传进来。
    """
    hooks = hooks or RunHooks()
    board = board if board is not None else Board()

    queue = list(replies or [])
    scripted = replies is not None      # 给了 --reply：用完就干净停下
    result = await run_and_print(team, task, on_message=hooks.on_message)

    #: 跨段累积的消息，只给无进展判据用。`TaskResult.messages` 是每段的增量，
    #: 只看一段凑不齐「连续两条同源消息」。
    history: list = []
    #: 总预算。三类续跑（拍板 / 无进展 / 自动重查）共用一个上限——自动重查也吃
    #: 这个预算是有意的：没有它，critic 反复要求重查就能拖到 MaxMessageTermination
    #: 才停，而且中途没人看得见。
    rounds = 0
    #: 只有「真的问了人」才 +1，用作「第 N 次拍板」的文案和 --reply 的第 N 个回答。
    #: 与总预算分开，否则自动重查会把编号顶乱。
    asked = 0
    #: 已经自动重查过哪几类。同一类第二次被要求重查就直接升级给人，理由见下面。
    researched: set[str] = set()

    while rounds < MAX_HANDOFFS:
        absorb_into(board, remember(history, result.messages))

        request = research_request(result)
        if request is None and not stopped_for_user(result):
            break                         # 自然结束（APPROVED / 撞上限 / 外部停止）

        rounds += 1

        # 同一类素材**第二次**被要求重查，说明第一次重查没解决问题。再自动跑一遍
        # 同样的动作不太可能有用，只会白烧轮次——升级给人，这正是「无进展停下交人」
        # 的同一原则。压力测试里，不设这一条时会一路自动重查到撞 MAX_HANDOFFS。
        escalate = request is not None and request.category in researched

        if request is not None and not escalate:
            # 自动重查：critic 指名某类素材不可信。作废它，把最新台账注入回去。
            # RR 忽略 HandoffMessage 的 target，critic 之后索引归 0，所以下一个
            # 发言的本来就是 researcher——这里写 target 只是让消息语义正确。
            researched.add(request.category)
            hooks.on_notice(supersede_notice(board, request.category), "warning")
            result = await run_and_print(
                team,
                HandoffMessage(
                    source="user",
                    target=RESEARCHER.name,
                    content=board.corrective_note(request.category, request.keyword),
                ),
                on_message=hooks.on_message,
            )
            continue

        asked += 1
        hooks.on_notice(f"该你拍板了（第 {asked} 次）", "banner")
        # 取正文而不是 handoff 那句固定话（Handoff 工具是零参数的，见该函数注释）
        question = last_substantive_content(result) or "（critic 没写出问题清单，只调了 handoff 工具）"
        # 提醒是**加在**问题清单前面，是加不是替。判据可能误判（researcher 把整份清单
        # 重发一遍就会撞上「重复搜索」），替掉的话用户就看不到自己要回答的那个问题了。
        if escalate:
            question = (
                f"⚠️ {request.category} 类素材自动重查过一次仍未通过，再重试也是同样的动作，"
                f"需要你介入。\n\n{question}"
            )
        else:
            stall = find_stall(history)
            if stall is not None:
                question = f"{stall.render()}\n\n{question}"
        hooks.on_notice(question, "question")

        answer = resolve_answer(queue, scripted, hooks.ask, asked, hooks.on_notice)
        if answer is None:
            break

        # 用户手动指名重查：走同一套「作废 → 重渲染 → 注入」。extra 只在用户真写了
        # 标记时才非空，所以普通拍板的续跑消息逐字不变（自测 [9] 锁着这一点）。
        extra = ""
        if (manual := parse_research_line(answer, _USER_RESEARCH_RE)) is not None:
            hooks.on_notice(supersede_notice(board, manual.category), "warning")
            extra = "\n\n" + board.corrective_note(manual.category, manual.keyword)

        # 续跑必须用 HandoffMessage，交回给触发 handoff 的那个 agent
        target = getattr(result.messages[-1], "source", CRITIC.name)
        result = await run_and_print(
            team, HandoffMessage(source="user", target=target, content=answer + extra),
            on_message=hooks.on_message,
        )

    hooks.on_notice(f"结束原因：{result.stop_reason}", "final")
    if result.stop_reason and "Maximum number of messages" in str(result.stop_reason):
        hooks.on_notice(
            "⚠️  撞到轮数上限了。这不算自然结束——回去查 critic 为什么不肯说 APPROVED。",
            "warning",
        )
    return result


async def stage5_full(task: str, replies: list[str] | None = None, *, hooks=None, board=None) -> None:
    """阶段 5 · 完整流程（含人机回路），日常用法。

    这一层只管**模型与队伍的生命周期**，对话流程在 run_with_handoffs 里。
    **前两个参数必须是位置参数**——launcher.py 用的是 `stage5_full(self.task, None,
    hooks=...)`，自测 [10] 按这个字面量核对，往中间插参数会静默打断它。

    Args:
        replies: 透传给 run_with_handoffs 的预设拍板回答，语义见那个函数。
        hooks: 出口回调。不给就是打印到终端、读键盘。
        board: 素材台账，透传。不给就由 run_with_handoffs 现建。
    """
    model = build_model()
    try:
        termination = build_termination()
        if hooks is not None and hooks.stop is not None:
            # 外部停止开关用 `|` **外挂**，不动 build_termination()——selftest 第 [6]
            # 节依赖它的行为。终端路径 hooks.stop is None，对象图与重构前完全相同。
            #
            # 为什么用 ExternalTermination 而不是 task.cancel()：它让「停止」走的是本来
            # 就存在的那条路——run_stream 正常返回 TaskResult（stop_reason 里写着
            # "External termination requested"），run_and_print 的 assert 不用动，
            # 也不用伪造 TaskResult。粒度是一个 agent 回合。
            termination = termination | hooks.stop
        team = build_team(model, (RESEARCHER, PLANNER, CRITIC), termination=termination)
        await run_with_handoffs(team, task, replies, hooks=hooks, board=board)
    finally:
        # 循环中途抛异常时也要关掉 client（重构前这条路径会漏关）。
        await model.close()


# ---------------------------------------------------------------------------
# 4. 离线自测（不需要 key，不联网）
# ---------------------------------------------------------------------------

def selftest() -> int:
    """验证接线是否正确。跑这个不需要 API key，也不会发起任何网络请求。"""
    from tools import (
        ALL_TOOLS,
        _amap_location,
        build_tools,
        calculator,
        fetch_page,
        hotel_options,
        taxi_fare,
    )

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  {'✓' if ok else '✗'} {label}{'  ' + detail if detail and not ok else ''}")
        if not ok:
            failures.append(label)

    print("\n[1] 工具：calculator 算得对")
    for expr, expect in [
        ("750*2*2 + 350*3", "4050"),
        ("55*2 + 50*2 + 80*2", "370"),
        ("300*3", "900"),
        ("(1+2)*3", "9"),
        ("10/4", "2.5"),
    ]:
        got = calculator(expr)
        check(f"{expr} = {expect}", got.endswith(f"= {expect}"), f"实际：{got}")

    print("\n[2] 工具：calculator 挡得住脏输入")
    for bad in ["__import__('os').system('ls')", "a + 1", "(1).__class__", "1/0", "2**99"]:
        got = calculator(bad)
        check(f"拒绝 {bad[:28]!r}", got.startswith("计算失败"), f"实际：{got}")

    print("\n[3] 工具：挂载关系与 README「三个 AI 分别干什么」表一致")
    # 联网的四把（search / fetch_page / taxi_fare / hotel_options）必须都只挂在 researcher
    # 身上——它们合起来才是「唯一联网者」这一条权力，拆开挂给别人照样废掉拓扑（tools.py 文件头）。
    check(
        "researcher 挂四把联网工具",
        RESEARCHER.tool_names == ("search", "fetch_page", "taxi_fare", "hotel_options"),
    )
    check("planner 一把都不挂", PLANNER.tool_names == ())
    check("critic 只挂 calculator", CRITIC.tool_names == ("calculator",))
    check("critic 有 handoff 通到 user", CRITIC.handoffs == ("user",))
    check(
        "ALL_TOOLS 正好五个",
        set(ALL_TOOLS) == {"search", "fetch_page", "taxi_fare", "hotel_options", "calculator"},
    )
    check("build_tools 能取到 search", len(build_tools(["search"])) == 1)
    # 下面几条是入参守卫，都要在联网之前生效，所以这一段不联网、也不需要高德 key。
    # fetch_page：模型经常把搜索关键词当网址传进来。
    check("fetch_page 挡住非网址入参", fetch_page("成都大熊猫基地").startswith("错误："))
    # taxi_fare / hotel_options：模型经常把 JSON 写坏，或者漏掉一半参数。
    check("taxi_fare 挡住坏 JSON", taxi_fare("{不是 json").startswith("参数不是合法 JSON"))
    check("taxi_fare 挡住缺 to", taxi_fare('{"from":"诸暨站"}').startswith("缺少 from 或 to"))
    check("hotel_options 挡住非对象", hotel_options("[1,2]").startswith("参数应是一个 JSON"))
    check(
        "hotel_options 挡住缺 keywords",
        hotel_options('{"city":"成都"}').startswith("缺少 city 或 keywords"),
    )
    # 已经是坐标的入参不该再走一次地理编码（省一次配额，也避免地名消歧出错）
    # （传空 key 是安全的：坐标那条路在用到 key 之前就返回了）
    check("taxi_fare 认得坐标入参", _amap_location("120.18,29.72", "", "")[0] == "120.18,29.72")
    try:
        build_tools(["nope"])
        check("未登记的工具名会报错", False, "没报错")
    except KeyError:
        check("未登记的工具名会报错", True)

    print("\n[4] 人格：拼接公式与禁用词")
    for spec in (RESEARCHER, PLANNER, CRITIC):
        msg = build_system_message(spec)
        check(f"{spec.name} 含角色 prompt", spec.role_prompt.strip() in msg)
        check(f"{spec.name} 含格式契约", spec.format_contract.strip() in msg)
        # critic.md §8.3 反模式 1：prompt 里不许出现否定式举例的禁用词
        leaked = [w for w in ("尚未", "之前还需") if w in msg]
        check(f"{spec.name} 无禁用词残留", not leaked, f"残留：{leaked}")

    print("\n[5] AutoGen 接线：能真的造出 agent 和 team")
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # 假 key：只造对象，不发请求。这一步能抓出 import 路径写错、参数名不对这类问题。
    model = OpenAIChatCompletionClient(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key="sk-selftest-not-a-real-key",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": False,
        },
    )
    try:
        agents = [build_agent(spec, model) for spec in (RESEARCHER, PLANNER, CRITIC)]
        check("三个 agent 造得出来", len(agents) == 3)
        # 这两个默认值不改的话，带工具的 agent 永远写不出结论（见 MAX_TOOL_ITERATIONS）
        check(
            "max_tool_iterations 已从默认 1 提高",
            all(a._max_tool_iterations == MAX_TOOL_ITERATIONS for a in agents),
        )
        check(
            "reflect_on_tool_use 已从默认 False 打开",
            all(a._reflect_on_tool_use is True for a in agents),
        )
        team = build_team(model, (RESEARCHER, PLANNER, CRITIC), termination=build_termination())
        check("team 造得出来", team is not None)
        check("user 不在 participants 里", "user" not in [a.name for a in team._participants])
    except Exception as exc:                       # noqa: BLE001
        check("AutoGen 接线", False, f"{type(exc).__name__}: {exc}")

    print("\n[6] 终止条件：用假对话验证各条分支")
    # 注意：0.7.x 的 TerminationCondition.reset() 和 __call__() **都是 async**。
    # 直接同步调用会拿到一个 coroutine 对象——它恒为真值，会让检查假通过。
    async def _probe() -> dict[str, object]:
        term = build_termination()
        out: dict[str, object] = {}
        cases = [
            ("问题清单", TextMessage(source="critic", content="问题 2 条：\n1. 预算超支 570。")),
            ("裸APPROVED", TextMessage(source="critic", content="APPROVED")),
            ("带空白", TextMessage(source="critic", content="  APPROVED\n")),
            ("handoff", HandoffMessage(source="critic", target="user", content="需要拍板")),
            ("附和话", TextMessage(source="critic", content="行程安排合理，考虑周到")),
            # 下面三条是回归项：子串匹配会误触发，精确匹配不会
            ("推理泄露", TextMessage(source="critic", content="All checks pass. Output APPROVED.")),
            ("否定句", TextMessage(source="critic", content="我还不能输出 APPROVED，预算仍超支 672。")),
            ("非critic", TextMessage(source="planner", content="APPROVED")),
            # 「需重查」标记的两个方向：合法形态要触发，被推理泄露或缺类别的不要。
            ("重查标记", TextMessage(
                source="critic", content="问题 1 条：\n1. 幻觉。住宿行的币种不对。\n需重查：住宿 春熙路 人民币价位区间")),
            ("缺类别的重查", TextMessage(source="critic", content="需重查：春熙路 人民币价位区间")),
            ("否定式重查", TextMessage(source="critic", content="不需要重查：住宿的素材是可信的")),
            ("非critic重查", TextMessage(source="planner", content="需重查：住宿 换一家")),
        ]
        for label, msg in cases:
            await term.reset()
            out[label] = await term([msg])
        return out

    try:
        probe = asyncio.run(_probe())
        for label in ("问题清单", "附和话", "推理泄露", "否定句", "非critic",
                      "缺类别的重查", "否定式重查", "非critic重查"):
            check(f"{label} 不触发终止", probe[label] is None, f"实际：{probe[label]}")
        for label in ("裸APPROVED", "带空白", "重查标记"):
            check(f"{label} 触发终止", probe[label] is not None, f"实际：{probe[label]}")
        check("handoff 到 user 触发终止", probe["handoff"] is not None, f"实际：{probe['handoff']}")
    except Exception as exc:                       # noqa: BLE001
        check("终止条件探测", False, f"{type(exc).__name__}: {exc}")

    print("\n[7] 消息打印：工具调用事件（content 是列表）不能崩")
    try:
        import contextlib
        import io

        from autogen_agentchat.messages import ToolCallRequestEvent
        from autogen_core import FunctionCall

        event = ToolCallRequestEvent(
            source="researcher",
            content=[FunctionCall(id="c1", name="search", arguments='{"query":"成都"}')],
        )
        with contextlib.redirect_stdout(io.StringIO()):     # 自测本身别刷屏
            _print_message(event)                            # 以前这里 AttributeError
        check("_print_message 能吃列表 content", True)

        class _FakeResult:                                   # last_content 只用 .messages
            messages = [event]

        with contextlib.redirect_stdout(io.StringIO()):
            got = last_content(_FakeResult())
        check("last_content 会跳过工具事件", got == "", f"实际：{got!r}")
    except AttributeError as exc:
        check("消息打印处理列表 content", False, f"AttributeError: {exc}")
    except Exception as exc:                                 # noqa: BLE001
        check("消息打印处理列表 content", False, f"{type(exc).__name__}: {exc}")

    print("\n[8] 文档与 persona 同步：§8.1/§8.2 必须逐字等于 persona 里的字符串")
    # 这条不变量已经被人为破坏过一次（改代码忘了改文档），所以用自测锁住：
    # docs/agents/*.md 的 §8.1 / §8.2 两个代码块，按约定就是 persona.py 里
    # role_prompt / format_contract 的原文，给读者看的解释一律写在代码块外面。
    from pathlib import Path as _Path

    for spec in (RESEARCHER, PLANNER, CRITIC):
        doc_path = _Path(__file__).resolve().parent / "docs" / "agents" / f"{spec.name}.md"
        if not doc_path.is_file():
            check(f"{spec.name}.md 存在", False, "文件缺失")
            continue
        try:
            doc = doc_path.read_text(encoding="utf-8")
            s81 = doc.split("### 8.1 角色 prompt")[1].split("```")[1].strip()
            s82 = doc.split("### 8.2 格式契约")[1].split("```")[1].strip()
        except IndexError:
            check(f"{spec.name}.md 有 §8.1/§8.2 代码块", False)
            continue
        check(f"{spec.name} §8.1 与 role_prompt 一致", s81 == spec.role_prompt.strip())
        check(f"{spec.name} §8.2 与 format_contract 一致", s82 == spec.format_contract.strip())

    print("\n[9] 人机回路：回调接缝与 --reply 队列（不需要 key、不联网）")
    # 这一段是补历史欠账：stage5_full 第一行就 build_model()，离线跑不动，所以
    # 「handoff 回路」这个最贵、最易错的部件此前在自测里覆盖率是零。抽出
    # run_with_handoffs 之后用假 team 就能把它整段跑穿。
    import contextlib
    import io
    from types import SimpleNamespace

    from autogen_agentchat.messages import ToolCallRequestEvent
    from autogen_core import FunctionCall

    tool_event = ToolCallRequestEvent(
        source="researcher",
        content=[FunctionCall(id="c1", name="search", arguments='{"query":"成都"}')],
    )

    class _FakeTeam:
        """按脚本吐消息的假 team。不联网、不建模型、不烧 token。"""

        def __init__(self, scripts):
            self._scripts = list(scripts)
            self._all: list = []
            self.tasks: list = []          # 记下每次 run_stream 收到的 task

        async def run_stream(self, *, task=None):
            self.tasks.append(task)
            batch = self._scripts.pop(0) if self._scripts else []
            self._all.extend(batch)
            for message in batch:          # async 生成器里不能用 yield from
                yield message
            # 累积而不是只给当前批：真 AutoGen 的 TaskResult.messages 是整轮的，
            # 而 run_with_handoffs 靠 result.messages[-1] 判断是不是又轮到人了。
            yield TaskResult(messages=list(self._all))

    def _handoff_batch(question: str) -> list:
        """一批消息：critic 的问题清单 + 它请求拍板。"""
        return [
            TextMessage(source="critic", content=question),
            HandoffMessage(source="critic", target="user", content="需要用户拍板"),
        ]

    # ---- A. 格式层：抽出来的纯函数要保住原来那两条分支 ----
    fm = format_message(tool_event)
    check("format_message 认出工具调用事件", fm is not None and fm.kind == "tool", f"实际：{fm}")
    check("format_message 保留了 → 调用行", fm is not None and "→ search(" in fm.text)
    fake_return = SimpleNamespace(
        source="researcher", content=[SimpleNamespace(name="search", content="x" * 500)]
    )
    fm_long = format_message(fake_return)
    check(
        "工具返回超 400 字符会截断并报总长",
        fm_long is not None and "…（共 500 字符）" in fm_long.text,
        f"实际：{fm_long.text[:60] if fm_long else None}",
    )
    check("空白消息返回 None", format_message(TextMessage(source="x", content="   ")) is None)

    # ---- B. 硬约束：_print_message 必须还能单参数调（selftest [7] 依赖它）----
    with contextlib.redirect_stdout(io.StringIO()):
        _print_message(TextMessage(source="planner", content="正文"))
    check("_print_message 仍能单参数调", True)

    # ---- C. 六种通知的终端版式逐字节锁定 ----
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_notice("该你拍板了（第 1 次）", "banner")
        _print_notice("问题清单", "question")
        _print_notice("[--reply 第 1 个回答] 砍住宿", "note")
        _print_notice("（空输入，停止。）", "plain")
        _print_notice("结束原因：x", "final")
        _print_notice("⚠️ 警告", "warning")
    expect = (
        f"\n{_SEP}\n该你拍板了（第 1 次）\n{_SEP}\n"
        "问题清单\n"
        "\n[--reply 第 1 个回答] 砍住宿\n"
        "（空输入，停止。）\n"
        f"\n{_SEP}\n结束原因：x\n{_SEP}\n"
        "⚠️ 警告\n"
    )
    check("六种通知的终端版式逐字节不变", buf.getvalue() == expect, f"实际：{buf.getvalue()!r}")

    def _quiet(**over):
        """一套静默 hooks，只覆盖需要观察的那几个出口。"""
        base = {
            "on_message": lambda message: None,
            "on_notice": lambda text, kind: None,
            "ask": lambda prompt: None,
        }
        base.update(over)
        return RunHooks(**base)

    async def _probe_hooks() -> dict:
        out: dict = {}

        # ① 续跑必须用 HandoffMessage，且交回给触发 handoff 的那个 agent
        seen: list = []
        team = _FakeTeam([_handoff_batch("住宿 400 超预算，怎么办？"),
                          [TextMessage(source="critic", content="APPROVED")]])
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            res = await run_and_print(team, "任务", on_message=seen.append)
        out["原始对象"] = bool(seen) and getattr(seen[0], "content", None) == "住宿 400 超预算，怎么办？"
        out["不打印"] = stdout.getvalue() == ""
        out["返回 TaskResult"] = isinstance(res, TaskResult)

        team = _FakeTeam([_handoff_batch("住宿 400 超预算，怎么办？"),
                          [TextMessage(source="critic", content="APPROVED")]])
        await run_with_handoffs(
            team, "任务", None, hooks=_quiet(ask=lambda prompt: "住宿砍到 250 一晚")
        )
        out["续跑条数"] = len(team.tasks)
        second = team.tasks[1] if len(team.tasks) > 1 else None
        out["续跑是 HandoffMessage"] = isinstance(second, HandoffMessage)
        out["续跑 source"] = getattr(second, "source", None)
        out["续跑 target"] = getattr(second, "target", None)
        out["续跑 content"] = getattr(second, "content", None)

        # ② --reply 用完就停，且**绝不回退到 ask**（docs/architecture.md 第 7 节第 8 条）
        asked: list = []
        team = _FakeTeam([_handoff_batch(f"问题 {i}") for i in range(4)])
        await run_with_handoffs(
            team, "任务", ["住宿砍到 250", "改高铁"],
            hooks=_quiet(ask=lambda prompt: asked.append(prompt) or "不该被调到"),
        )
        out["预设用完时 ask 次数"] = len(asked)
        out["预设用完跑了几次"] = len(team.tasks)

        # ③ ask 拿不到输入 / ④ ask 返回空串：两条都必须干净停下
        for label, reply in (("没有输入", None), ("空输入", "")):
            notices: list = []
            team = _FakeTeam([_handoff_batch("问题")])
            await run_with_handoffs(
                team, "任务", None,
                hooks=_quiet(ask=lambda prompt, r=reply: r,
                             on_notice=lambda text, kind: notices.append((text, kind))),
            )
            out[f"{label} 通知"] = notices
            out[f"{label} 条数"] = len(team.tasks)

        # ⑤ 最多问 MAX_HANDOFFS 次就收手
        team = _FakeTeam([_handoff_batch(f"问题 {i}") for i in range(8)])
        await run_with_handoffs(team, "任务", None, hooks=_quiet(ask=lambda prompt: "继续"))
        out["封顶条数"] = len(team.tasks)

        # ⑥ 自动重查：critic 写「需重查：」，作废该类素材并把声明注入回群聊
        _material = "【住宿】春熙路一带 经济型/快捷连锁 约300-400/晚（来源：https://b.example）"
        _marker = "问题 1 条：\n1. 幻觉。住宿行的币种不对，不是可订价。\n需重查：住宿 春熙路 人民币价位区间"
        asked_auto: list = []
        board = Board()
        team = _FakeTeam([[TextMessage(source="researcher", content=_material),
                           TextMessage(source="critic", content=_marker)],
                          [TextMessage(source="critic", content="APPROVED")]])
        await run_with_handoffs(
            team, "任务", None, board=board,
            hooks=_quiet(ask=lambda prompt: asked_auto.append(prompt) or "不该被调到"),
        )
        out["自动重查 跑了几次"] = len(team.tasks)
        out["自动重查 没问人"] = not asked_auto
        out["自动重查 作废了住宿"] = board.items["住宿"] == [] and len(board.stale) == 1
        third = team.tasks[1] if len(team.tasks) > 1 else None
        out["自动重查 用 HandoffMessage"] = isinstance(third, HandoffMessage)
        out["自动重查 target 是 researcher"] = getattr(third, "target", None) == RESEARCHER.name
        _note = getattr(third, "content", "") or ""
        out["自动重查 声明含被作废的原文"] = "汉庭" in _note or "春熙路一带" in _note
        out["自动重查 声明含重查要求"] = "重查要求" in _note and "人民币价位区间" in _note

        # ⑦ 手动重查：用户拍板时写同样的话，走同一套「作废 → 重渲染 → 注入」
        board = Board()
        team = _FakeTeam([[TextMessage(source="researcher", content=_material),
                           TextMessage(source="critic", content="问题 1 条：\n1. 幻觉。"),
                           HandoffMessage(source="critic", target="user", content="需要拍板")],
                          [TextMessage(source="critic", content="APPROVED")]])
        await run_with_handoffs(
            team, "任务", None, board=board,
            hooks=_quiet(ask=lambda prompt: "重查：住宿 春熙路人民币价位"),
        )
        second = team.tasks[1] if len(team.tasks) > 1 else None
        out["手动重查 作废了住宿"] = board.items["住宿"] == []
        out["手动重查 声明进了续跑消息"] = "重查要求" in (getattr(second, "content", "") or "")
        out["手动重查 仍交回 critic"] = getattr(second, "target", None) == "critic"

        # ⑧ 无进展：critic 连续两轮报同一批问题，第二轮的「问题」位换成证据
        questions: list = []
        _same = "问题 1 条：\n1. 幻觉。住宿行的币种不对，应换人民币来源。"
        team = _FakeTeam([_handoff_batch(_same), _handoff_batch(_same)])
        await run_with_handoffs(
            team, "任务", None,
            hooks=_quiet(ask=lambda prompt: "重查：住宿 换人民币来源",
                         on_notice=lambda text, kind: questions.append((text, kind)))
        )
        out["无进展提示"] = [t for t, k in questions if k == "question"]

        # ⑨ 同一类素材第二次被要求重查 → 升级给人，不再自动重查。
        # 不设这一条时，critic 反复指名同一类会一路自动重查到撞 MAX_HANDOFFS。
        esc_notices: list = []
        esc_asked: list = []
        board = Board()
        team = _FakeTeam([[TextMessage(source="researcher", content=_material),
                           TextMessage(source="critic", content=_marker)],
                          [TextMessage(source="researcher", content=_material),
                           TextMessage(source="critic", content=_marker)],
                          [TextMessage(source="critic", content="APPROVED")]])
        await run_with_handoffs(
            team, "任务", None, board=board,
            hooks=_quiet(ask=lambda prompt: esc_asked.append(prompt) or "住宿按 250/晚",
                         on_notice=lambda text, kind: esc_notices.append((text, kind))),
        )
        out["同类再重查 自动作废次数"] = sum(1 for _, k in esc_notices if k == "warning")
        out["同类再重查 问了人"] = len(esc_asked) == 1
        out["同类再重查 提示"] = [t for t, k in esc_notices if k == "question"]
        out["同类再重查 段数"] = len(team.tasks)
        return out

    try:
        hooks_probe = asyncio.run(_probe_hooks())
        check("on_message 收到原始 message 对象", hooks_probe["原始对象"])
        check("on_message 给了就不再打印到 stdout", hooks_probe["不打印"],
              f"实际：{hooks_probe['不打印']!r}")
        check("run_and_print 仍返回 TaskResult", hooks_probe["返回 TaskResult"])
        check("续跑了一次", hooks_probe["续跑条数"] == 2, f"实际：{hooks_probe['续跑条数']}")
        check("续跑用的是 HandoffMessage", hooks_probe["续跑是 HandoffMessage"])
        check("续跑 source 是 user", hooks_probe["续跑 source"] == "user",
              f"实际：{hooks_probe['续跑 source']}")
        check("续跑交回给 critic（触发 handoff 的那个）", hooks_probe["续跑 target"] == "critic",
              f"实际：{hooks_probe['续跑 target']}")
        check("续跑 content 是人的原话", hooks_probe["续跑 content"] == "住宿砍到 250 一晚",
              f"实际：{hooks_probe['续跑 content']}")
        check("--reply 预设用完绝不回退到 ask", hooks_probe["预设用完时 ask 次数"] == 0,
              f"实际调了 {hooks_probe['预设用完时 ask 次数']} 次")
        check("--reply 第 3 次不给就停（1 首次 + 2 续跑）", hooks_probe["预设用完跑了几次"] == 3,
              f"实际：{hooks_probe['预设用完跑了几次']}")
        check("ask 拿不到输入 → 干净停下",
              ("（没有拿到输入，停在当前进度。）", "note") in hooks_probe["没有输入 通知"])
        check("拿不到输入后不再续跑", hooks_probe["没有输入 条数"] == 1,
              f"实际：{hooks_probe['没有输入 条数']}")
        check("ask 返回空串 → 空输入停止",
              ("（空输入，停止。）", "plain") in hooks_probe["空输入 通知"])
        check("空输入后不再续跑", hooks_probe["空输入 条数"] == 1,
              f"实际：{hooks_probe['空输入 条数']}")
        check(f"最多问 {MAX_HANDOFFS} 次就收手（1 首次 + {MAX_HANDOFFS} 续跑）",
              hooks_probe["封顶条数"] == MAX_HANDOFFS + 1,
              f"实际：{hooks_probe['封顶条数']}")
        check("自动重查：作废后自动续跑一次", hooks_probe["自动重查 跑了几次"] == 2,
              f"实际：{hooks_probe['自动重查 跑了几次']}")
        check("自动重查不占用人的注意力", hooks_probe["自动重查 没问人"])
        check("自动重查作废了该类素材", hooks_probe["自动重查 作废了住宿"])
        check("自动重查用 HandoffMessage 续跑", hooks_probe["自动重查 用 HandoffMessage"])
        check("自动重查指名 researcher",
              hooks_probe["自动重查 target 是 researcher"],
              f"实际：{hooks_probe['自动重查 target 是 researcher']}")
        check("自动重查的声明含被作废的原文", hooks_probe["自动重查 声明含被作废的原文"])
        check("自动重查的声明含重查要求", hooks_probe["自动重查 声明含重查要求"])
        check("手动重查作废了该类素材", hooks_probe["手动重查 作废了住宿"])
        check("手动重查的声明进了续跑消息", hooks_probe["手动重查 声明进了续跑消息"])
        check("手动重查仍交回 critic（不动 [9] 的 target 契约）",
              hooks_probe["手动重查 仍交回 critic"])
        # 证据是**加在**问题清单前面，不是替掉它——判据可能误判（researcher 重发整份
        # 清单就会撞上「重复搜索」），替掉的话用户就看不到要回答的那个问题了。
        _stalled = [t for t in hooks_probe["无进展提示"] if "无进展" in t]
        check("撞上绕圈时把证据摆给用户", bool(_stalled),
              f"实际：{hooks_probe['无进展提示']}")
        check("证据不替掉问题清单本身",
              bool(_stalled) and all("幻觉" in t for t in _stalled),
              f"实际：{_stalled}")
        check("同一类第二次要求重查只自动作废一次",
              hooks_probe["同类再重查 自动作废次数"] == 1,
              f"实际：{hooks_probe['同类再重查 自动作废次数']}")
        check("同一类第二次要求重查时升级给人",
              hooks_probe["同类再重查 问了人"]
              and any("自动重查过一次" in t for t in hooks_probe["同类再重查 提示"]),
              f"实际：{hooks_probe['同类再重查 提示']}")
    except Exception as exc:                       # noqa: BLE001
        check("人机回路探测", False, f"{type(exc).__name__}: {exc}")

    print("\n[10] 启动器：语法自检与分层红线")
    # 启动器是**可选**的：没装它、没浏览器、没起服务，这一节也应该能跑。
    # 所以这里只 compile（不 exec），既不启服务器也不占端口。
    launcher_src = ""
    launcher_path = _Path(__file__).resolve().parent / "launcher.py"
    if not launcher_path.is_file():
        check("launcher.py 存在", False, "文件缺失")
    else:
        launcher_src = launcher_path.read_text(encoding="utf-8")
        try:
            compile(launcher_src, "launcher.py", "exec")
            check("launcher.py 语法正确", True)
        except SyntaxError as exc:
            check("launcher.py 语法正确", False, f"{exc}")
        check("launcher.py 通过 RunHooks 接 main", "RunHooks" in launcher_src)
        check(
            "launcher.py 用 format_message 而不是去解析终端文本",
            "format_message" in launcher_src,
        )
        # 这条守着一个很容易再犯的坑：replies=[] 不等于「没给预设」，
        # 传空列表会让第一次 handoff 就「预设用完」停下（见 run_with_handoffs 的注释）。
        check(
            "launcher.py 给 stage5_full 传 replies=None 而不是 []",
            "stage5_full(self.task, None," in launcher_src,
            "调用点写成了别的形式，回去确认第二个实参是 None",
        )
        check("launcher.py 有 /health 身份端点", '"/health"' in launcher_src)

        # 「重复双击不该起第二个服务」靠 _existing_instance 认人。真起一个服务来验，
        # 顺带验反面：陌生端口不能被误认成自己人（认错就会把浏览器指到不相干的页面）。
        # 绑 0 号端口让系统分配空闲端口，不会和正在跑的服务打架。
        try:
            import importlib.util as _ilu
            import threading as _threading

            _spec = _ilu.spec_from_file_location("_launcher_probe", launcher_path)
            _lmod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_lmod)

            # 59999 上基本不可能有东西；真有的话这条会红，那也是有用的信息。
            check("空端口不会被误认成自己人", _lmod._existing_instance(59999) is False)
            _probe_server = _lmod._Server(("127.0.0.1", 0), _lmod._Handler)
            _threading.Thread(target=_probe_server.serve_forever, daemon=True).start()
            try:
                check(
                    "起过的服务能被认出来",
                    _lmod._existing_instance(_probe_server.server_address[1]) is True,
                )
            finally:
                _probe_server.shutdown()
                _probe_server.server_close()
        except Exception as exc:                       # noqa: BLE001
            check("启动器身份探测", False, f"{type(exc).__name__}: {exc}")

    # 建桌面快捷方式那个脚本，同样只做语法检查。
    _shortcut_path = _Path(__file__).resolve().parent / "make_shortcut.py"
    if not _shortcut_path.is_file():
        check("make_shortcut.py 存在", False, "文件缺失")
    else:
        try:
            compile(_shortcut_path.read_text(encoding="utf-8"), "make_shortcut.py", "exec")
            check("make_shortcut.py 语法正确", True)
        except SyntaxError as exc:
            check("make_shortcut.py 语法正确", False, f"{exc}")

    # 分层红线：launcher 依赖 main，main 绝不反向依赖 launcher。
    # 这样这个文件删掉、或者在一台没有浏览器的机器上，--selftest 照样绿。
    # 针要拼出来：直接写整串的话，这一行自己就会命中（第一次跑就踩了这个自摆乌龙）。
    _needle = "import " + "launcher"
    check("main.py 不反向依赖启动器", _needle not in _Path(__file__).read_text(encoding="utf-8"))

    print("\n[11] 素材台账：解析、作废、幂等")
    from ledger import CATEGORIES as _CATEGORIES
    from ledger import Board as _Board
    from ledger import parse_materials as _parse_materials

    # 逐字取自 researcher 契约（§8.2）里的示例——契约能解析是最低要求，
    # 真输出能不能解析得靠 `--stage 2` 目视，自测守不了那个。
    _contract_lines = [
        "【交通】北京⇄成都 高铁二等座 约750/人 单程（来源：https://a.example）",
        "【住宿】春熙路一带 经济型/快捷连锁 约300-400/晚（来源：https://b.example）",
        "【住宿】春熙路 汉庭IFS国金中心店 经济型 4.7分（来源：hotel_options）",
        "【景点】大熊猫繁育研究基地 成人票55（来源：https://c.example）",
        "【景点】大熊猫繁育研究基地 6岁以下免票（来源：https://d.example）",
        "【市内】巡游出租车 起步价10元/3公里，续程2.4元/公里（来源：https://e.example）",
        "【市内】诸暨站→五泄风景区 27.1公里 打车约71元（来源：taxi_fare 估算）",
    ]
    _parsed, _leftover = _parse_materials("\n".join(_contract_lines))
    check("契约里的 7 行示例全部解析得出", len(_parsed) == 7, f"实际：{len(_parsed)}")
    check("契约行没有落进未解析桶", not _leftover, f"实际：{_leftover}")
    check("类别都在四类里", all(m.category in _CATEGORIES for m in _parsed))
    check("来源含工具名的行也解析得出",
          any(m.source == "hotel_options" for m in _parsed))

    _board = _Board()
    _feed = "\n".join(_contract_lines) + "\n这是跑题的一句。"
    check("首次吸收算作有变化", _board.absorb(_feed) is True)
    _snapshot = {c: len(v) for c, v in _board.items.items()}
    check("同一段再喂一次是幂等的", _board.absorb(_feed) is False)
    check("幂等后条目数不变",
          {c: len(v) for c, v in _board.items.items()} == _snapshot)
    # 未解析的行必须留住——丢一行素材，critic 的幻觉比对就少一条依据。
    check("解析不了的行进了 unparsed", _board.unparsed == ["这是跑题的一句。"],
          f"实际：{_board.unparsed}")
    check("未解析的行照样出现在 render 里", "这是跑题的一句。" in _board.render())

    _removed = _board.supersede("住宿")
    check("作废返回被作废的条目", len(_removed) == 2, f"实际：{len(_removed)}")
    check("作废后该类清空", _board.items["住宿"] == [])
    check("作废后 render 不再列出旧条目",
          "汉庭IFS国金中心店" not in _board.render())
    _note = _board.corrective_note("住宿", "春熙路 人民币价位区间")
    check("作废声明逐条列出被作废的素材", "汉庭IFS国金中心店" in _note)
    check("作废声明带上重查关键词", "春熙路 人民币价位区间" in _note)
    check("作废其余类别不受影响", len(_board.items["交通"]) == 1)
    check("非法类别不抛异常、也不作废", _board.supersede("餐饮") == [])

    print("\n[12] 无进展判据：命中真信号，且不误伤假数据")
    from ledger import detect_stall as _detect

    def _msg(source: str, text: str):
        return TextMessage(source=source, content=text)

    def _stall_of(messages):
        return _detect(messages, researcher=RESEARCHER.name,
                       planner=PLANNER.name, critic=CRITIC.name)

    _urls = "查到 https://x.example/a https://x.example/b https://x.example/c"
    _hit = _stall_of([_msg("researcher", _urls), _msg("researcher", _urls)])
    check("连续两轮同一批网址 → 重复搜索",
          _hit is not None and _hit.kind == "重复搜索", f"实际：{_hit}")
    check("网址只重合一半 → 不算无进展",
          _stall_of([_msg("researcher", "https://x.example/a https://x.example/b"),
                     _msg("researcher", "https://y.example/c https://y.example/d")]) is None)
    _table = "| Day | 时段 | 项目 | 花费 | 备注 |\n|---|---|---|---|---|\n| 1 | 上午 | 熊猫基地 | 110 | 2人 |"
    _hit = _stall_of([_msg("planner", _table), _msg("planner", _table)])
    check("连续两轮逐字相同的表 → 空转表格",
          _hit is not None and _hit.kind == "空转表格", f"实际：{_hit}")
    _problems = "问题 1 条：\n1. 幻觉。住宿行的币种不对，应换人民币来源。"
    _hit = _stall_of([_msg("critic", _problems), _msg("critic", _problems)])
    check("连续两轮同一批问题 → 重复问题", _hit is not None and _hit.kind == "重复问题")

    # **这条是 [9] 不被误伤的回归保证。** [9] 的假 critic 正文是 "问题 1"…"问题 8"，
    # 跟真问题清单长得像但不是清单；判据一松，[9] 的段数断言立刻崩。
    check("「问题 N」这种假正文不误判成问题清单",
          _stall_of([_msg("critic", "问题 1"), _msg("critic", "问题 2")]) is None)
    check("只有一条消息时谈不上「连续两轮」", _stall_of([_msg("critic", _problems)]) is None)
    check("空历史不报无进展", _stall_of([]) is None)
    # 排序：三条同时命中时，报可信度最高的那条。「researcher 重发整份清单」是会误判的
    # 最弱信号，不该盖住「critic 在复读」这条实锤。
    _both = _stall_of([
        _msg("researcher", _urls), _msg("critic", _problems),
        _msg("researcher", _urls), _msg("critic", _problems),
    ])
    check("三条同时命中时优先报「重复问题」",
          _both is not None and _both.kind == "重复问题", f"实际：{_both}")

    print("\n" + _SEP)
    if failures:
        print(f"离线自测失败 {len(failures)} 项：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(
        "离线自测全部通过。接线没问题，可以填 key 跑 `python main.py \"需求\"`\n"
        "（或者 `python launcher.py` 用网页界面）了。"
    )
    return 0


# ---------------------------------------------------------------------------
# 5. 入口
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AutoGen 多智能体旅行规划器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            '  python main.py "我国庆去诸暨玩，三天，两个人，预算 2000。"\n'
            "  python main.py --selftest\n"
            "  python launcher.py       # 网页界面，不用终端\n"
            "\n"
            "不传需求就用内置示例任务。跑起来后 critic 会停下来问你拍板——"
            "回答时带一个具体的数字或动作，「住宿砍到 250 一晚」比「再优化一下」有用。"
        ),
    )
    # 需求用**位置参数**：日常用法的全部就是 `python main.py "一句话需求"`。
    parser.add_argument("task", nargs="?", default=None, help="旅行需求，一句话（不传用内置示例）")
    parser.add_argument("--selftest", action="store_true", help="离线自测，不需要 key、不联网")
    parser.add_argument(
        "--task", dest="task_opt", default=None, metavar="需求",
        help="同上，位置参数的别名（早期文档里用的写法）",
    )
    parser.add_argument(
        "--reply",
        action="append",
        default=None,
        metavar="回答",
        help="拍板回答，可重复给多次（第 N 次问你用第 N 个）。一个都不给则读键盘",
    )
    # 调试选项：日常不用。出问题时按 1→2→3→4 往回退，能定位是哪一层坏的。
    parser.add_argument(
        "--stage", choices=["1", "2", "3", "4", "5", "all"], help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.task and args.task_opt:
        print("\n需求用位置参数给一次就行，不要同时写 --task。\n", file=sys.stderr)
        return 2
    task = args.task or args.task_opt or DEFAULT_TASK

    # 不传 --stage 就是完整流程（含人机回路）。--stage 只用于逐层排查。
    stage = args.stage or "5"
    if stage in ("5", "all"):
        runner = lambda: asyncio.run(stage5_full(task, args.reply))   # noqa: E731
    else:
        stage_fn = {"1": stage1_connectivity, "2": stage2_researcher,
                    "3": stage3_two_agents, "4": stage4_with_critic}[stage]
        runner = lambda: asyncio.run(stage_fn())               # noqa: E731

    try:
        runner()
    except RuntimeError as exc:
        # 最常见的一种：没配 key。给出干净提示，不要糊一堆 traceback。
        print(f"\n{_SEP}\n无法启动：\n{exc}\n{_SEP}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
