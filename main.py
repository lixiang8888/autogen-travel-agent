# -*- coding: utf-8 -*-
"""
main.py —— 组队 + 跑：拓扑、终止条件、入口
============================================

改拓扑、调轮数上限、换 agent 说话顺序，只动这个文件。这里不认识 persona.py 里
那些字符串的内容——这是刻意的分层。

**拓扑**：RoundRobinGroupChat，固定顺序 researcher → planner → critic → 循环。
起步不用 SelectorGroupChat：先要可预测，再要聪明。等整个流程跑通了、知道「正常的
对话长什么样」了，再换拓扑做对照实验（BUILD.md 已知坑 7）。

**终止条件**：三重，缺一不可。
    TextMentionTermination("APPROVED")   critic 认可
    HandoffTermination(target="user")    需要人拍板，交回控制权
    MaxMessageTermination(15)            兜底保险丝——LLM 不一定老实

**user 不在 participants 里。** 这是本项目与 BUILD.md 阶段 5 的一处有意分歧：
UserProxyAgent 的默认 input_func 读控制台，一旦进队，RoundRobin 每转到它就会阻塞
整个 team，官方文档说这会让 team 变成「无法保存或恢复」的状态。改用
HandoffTermination 后，user 只是 HandoffMessage 里的一个字符串标签。
详见 docs/agents/user.md §2.1。

**续跑必须用 HandoffMessage**，不能直接传字符串——否则报
`ValueError: The existing handoff target user is not one of the participants`。

跑法：
    python main.py --selftest     # 离线自测，不需要 key、不联网
    python main.py --stage 1      # 阶段 1：DeepSeek 连通性
    python main.py --stage 2      # 阶段 2：单 agent 带 search 跑通
    python main.py --stage 3      # 阶段 3：两个 agent 进群聊
    python main.py --stage 4      # 阶段 4：加 critic
    python main.py --stage 5      # 阶段 5：终止条件 + user 回路
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff, TaskResult
from autogen_agentchat.conditions import (
    HandoffTermination,
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_agentchat.messages import HandoffMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat

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
#: 见 BUILD.md「已实测修正」第 6 条。
MAX_TOOL_ITERATIONS = 5

_SEP = "=" * 72


# ---------------------------------------------------------------------------
# 1. 造 agent 和 team
# ---------------------------------------------------------------------------

def build_termination():
    """三重终止条件。`|` 不要写成 `&`——`&` 要两个同时满足，会一直跑到上限。"""
    return (
        TextMentionTermination("APPROVED")
        | HandoffTermination(target="user")
        | MaxMessageTermination(MAX_MESSAGES)
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
# 2. 跑与打印
# ---------------------------------------------------------------------------

def _print_message(message) -> None:
    """打印一条消息。

    content 可能是 str（TextMessage），也可能是**对象列表**——工具调用类事件
    （ToolCallRequestEvent / ToolCallExecutionEvent）就是后者。早期版本直接
    `.strip()`，一遇到工具调用就 AttributeError，阶段 3/4/5 全崩。
    """
    source = getattr(message, "source", "?")
    content = getattr(message, "content", "")

    if isinstance(content, list):
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
        text = (content or "").strip()

    if not text:
        return
    print(f"\n{_SEP}\n[{source}]\n{_SEP}")
    print(text)


async def run_and_print(team, task) -> TaskResult:
    """流式跑，实时打印每条消息，返回 TaskResult。

    用 run_stream 而不是 run，是因为一次完整对话要十几轮 LLM 调用，
    等全部跑完再打印会让人以为卡死了。
    """
    result: TaskResult | None = None
    async for message in team.run_stream(task=task):
        if isinstance(message, TaskResult):
            result = message
        else:
            _print_message(message)
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


# ---------------------------------------------------------------------------
# 3. 五个阶段（对应 BUILD.md §增量搭建路线）
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


async def stage5_full(task: str, replies: list[str] | None = None) -> None:
    """阶段 5 · 终止条件与 user 回路。

    验证点：连跑三次，每次都在合理轮数内自然结束（而不是撞到 15 轮上限）。
    撞上限说明 critic 不肯说 APPROVED，回去改它的 system_message。

    Args:
        replies: 预设的拍板回答，第 N 次 handoff 用第 N 个。给完就停。
            一个都不给则每次读一行标准输入。用于非交互跑（脚本化、CI、
            或者「用不同的回答测收敛性」）。
    """
    model = build_model()
    team = build_team(model, (RESEARCHER, PLANNER, CRITIC), termination=build_termination())

    queue = list(replies or [])
    result = await run_and_print(team, task)
    handoffs = 0
    while stopped_for_user(result) and handoffs < MAX_HANDOFFS:
        handoffs += 1
        print(f"\n{_SEP}\n该你拍板了（第 {handoffs} 次）\n{_SEP}")
        print(last_content(result))

        if queue:
            answer = queue.pop(0).strip()
            print(f"\n[--reply 第 {handoffs} 个回答] {answer}")
        else:
            try:
                answer = input("\n你的决定（一句话，带一个数字或一个动作）> ").strip()
            except (EOFError, KeyboardInterrupt):
                # stdin 是空的（非交互跑）或用户按了 Ctrl-C/Ctrl-D。
                # 这里不该糊一个 traceback——没输入就干净地停在当前进度。
                print("\n（没有拿到输入，停在当前进度。）")
                break

        if not answer:
            print("（空输入，停止。）")
            break
        # 续跑必须用 HandoffMessage，交回给触发 handoff 的那个 agent
        target = getattr(result.messages[-1], "source", CRITIC.name)
        result = await run_and_print(
            team, HandoffMessage(source="user", target=target, content=answer)
        )

    await model.close()

    print(f"\n{_SEP}\n结束原因：{result.stop_reason}\n{_SEP}")
    if result.stop_reason and "Maximum number of messages" in str(result.stop_reason):
        print("⚠️  撞到轮数上限了。这不算自然结束——回去查 critic 为什么不肯说 APPROVED。")


# ---------------------------------------------------------------------------
# 4. 离线自测（不需要 key，不联网）
# ---------------------------------------------------------------------------

def selftest() -> int:
    """验证接线是否正确。跑这个不需要 API key，也不会发起任何网络请求。"""
    from tools import ALL_TOOLS, calculator, build_tools

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

    print("\n[3] 工具：挂载关系与 BUILD.md 的表格一致")
    check("researcher 只挂 search", RESEARCHER.tool_names == ("search",))
    check("planner 一把都不挂", PLANNER.tool_names == ())
    check("critic 只挂 calculator", CRITIC.tool_names == ("calculator",))
    check("critic 有 handoff 通到 user", CRITIC.handoffs == ("user",))
    check("ALL_TOOLS 正好两个", set(ALL_TOOLS) == {"search", "calculator"})
    check("build_tools 能取到 search", len(build_tools(["search"])) == 1)
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
            ("APPROVED", TextMessage(source="critic", content="APPROVED")),
            ("handoff", HandoffMessage(source="critic", target="user", content="需要拍板")),
            ("附和话", TextMessage(source="critic", content="行程安排合理，考虑周到")),
        ]
        for label, msg in cases:
            await term.reset()
            out[label] = await term([msg])
        return out

    try:
        probe = asyncio.run(_probe())
        check("问题清单不触发终止", probe["问题清单"] is None, f"实际：{probe['问题清单']}")
        check(
            "APPROVED 触发终止",
            probe["APPROVED"] is not None and "APPROVED" in str(probe["APPROVED"]),
            f"实际：{probe['APPROVED']}",
        )
        check("handoff 到 user 触发终止", probe["handoff"] is not None, f"实际：{probe['handoff']}")
        check("无信息量的附和话不触发终止", probe["附和话"] is None, f"实际：{probe['附和话']}")
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

    print("\n" + _SEP)
    if failures:
        print(f"离线自测失败 {len(failures)} 项：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("离线自测全部通过。接线没问题，可以填 key 跑 --stage 1 了。")
    return 0


# ---------------------------------------------------------------------------
# 5. 入口
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AutoGen 多智能体旅行规划器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="阶段编号对应 BUILD.md §增量搭建路线（5 阶段）。",
    )
    parser.add_argument("--selftest", action="store_true", help="离线自测，不需要 key")
    parser.add_argument("--stage", choices=["1", "2", "3", "4", "5", "all"], help="跑哪个阶段")
    parser.add_argument("--task", default=DEFAULT_TASK, help="初始需求，一句话")
    parser.add_argument(
        "--reply",
        action="append",
        default=None,
        metavar="回答",
        help="阶段 5：拍板回答，可重复给多次（第 N 次 handoff 用第 N 个）。一个都不给则读键盘",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    if not args.stage:
        parser.print_help()
        return 0

    if args.stage in ("5", "all"):
        runner = lambda: asyncio.run(stage5_full(args.task, args.reply))   # noqa: E731
    else:
        stage_fn = {"1": stage1_connectivity, "2": stage2_researcher,
                    "3": stage3_two_agents, "4": stage4_with_critic}[args.stage]
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
