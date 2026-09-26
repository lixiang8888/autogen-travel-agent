# 制作手册 · AutoGen 多智能体旅行规划

> 这份手册是**搭建蓝图**，写在代码之前。它规定每个文件干什么、四个 agent 怎么划界、
> 按什么顺序往上搭、以及哪些坑一定会踩。
>
> 体例对齐 [plan-solve-agent-practice](../plan-solve-agent-practice/) 的 MANUAL.md：
> 「想改 X 就动 Y」+「已知坑与设计取舍」。假设读者（未来的我）已经懂 LLM agent，
> 不复述基础概念。

## 目录

- [这是什么](#这是什么)
- [框架选型：为什么是 AutoGen 0.7](#框架选型为什么是-autogen-07)
- [目录结构与职责](#目录结构与职责)
- [四个 Agent](#四个-agent)
- [对话流程与终止](#对话流程与终止)
- [增量搭建路线（5 阶段）](#增量搭建路线5-阶段)
- [想改 X，就动 Y](#想改-x就动-y)
- [预期对话示意](#预期对话示意)
- [已知坑与设计取舍](#已知坑与设计取舍)
- [和 PS 框架的对照](#和-ps-框架的对照)
- [已实测修正](#已实测修正)

---

## 这是什么

一个纯 AutoGen 的多智能体旅行规划器。输入一句自然语言需求
（「帮我规划国庆去成都玩三天，两个人，预算 4000」），四个 agent 通过群聊
协作产出一份逐日行程。

**它不是什么**：不是 `plan-solve-agent-practice` 的升级版，两者是**并列的姊妹项目**，
演示两种不同的 agent 架构。互不依赖、互不 import。

**做它的目的**：手上有 PS（单 agent 内部三阶段，自问自答）的经验，
现在要补上「多个 LLM 实例开会」这一半。旅行规划是刻意选的靶子——
它的对错**你自己一眼能判断**，不需要任何领域知识：
日程有没有时间冲突、有没有东一个西一个来回跑、预算加总对不对、
景点是不是编的。

---

## 框架选型：为什么是 AutoGen 0.7

**已确认的事实**（2026-09 查证）：

- AutoGen 最后一个活跃版本是 `autogen-agentchat` **0.7.5**（2025-09）。
- 2025-10 起进入**维护模式**：只修 bug，不加新特性。
- 微软把 AutoGen 和 Semantic Kernel 合并成了 **Microsoft Agent Framework（MAF）**，
  2026-04-03 发布 1.0。AutoGen 的群聊编排概念被原样搬进了 MAF。

**为什么还是选它**：教程和文档最多，遇到问题搜得到答案；而且群聊编排这套概念
能直接迁移到 MAF。锁定 0.7 是为了让手册里的每一段代码都有确定的 API 可对。

**要付的代价**：它不会再有新特性了。如果哪天要长期维护，迁移成本存在但可控
（概念不变，主要是 API 形状变了）。

### 版本红线

**绝对不要用 `from autogen import GroupChat` 这种写法。** 那是 0.2 时代的 API，
和 0.4+ 完全不兼容，而网上大量中文教程还停在那里。本手册一律用 0.4+ 的异步 API：

```python
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import (
    TextMentionTermination, MaxMessageTermination, HandoffTermination)
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.base import Handoff
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

安装：

```bash
uv venv --python /usr/bin/python3.14     # 为什么不是 pip / 3.12，见文末修正 3
uv pip install "autogen-agentchat==0.7.*" "autogen-ext[openai]==0.7.*" "requests>=2.31"
```

注意 `autogen-core` / `autogen-agentchat` / `autogen-ext` 是三个包，
0.4 之后拆开的。**版本号要一致**，混装会出莫名其妙的导入错误。

---

## 目录结构与职责

对齐 `plan-solve-agent-practice` 的分层，以后在两个项目之间搬东西不用重新理解布局。

```
autogen-travel-agent/
├── BUILD.md           # 本文件（蓝图）
├── README.md          # 定位、快速开始、结构一览
├── docs/agents/       # 四份 agent 说明书：角色、边界、产出契约
│   ├── researcher.md  planner.md  critic.md  user.md
├── llm.py             # 模型客户端：DeepSeek 走 OpenAI 兼容协议
├── tools.py           # 工具：search（Tavily）+ calculator
├── persona.py         # 三个 LLM agent 的人格：角色 prompt + 格式契约
├── main.py            # 组队 + 跑：拓扑、终止条件、5 个阶段的入口
├── keys.py            # key，已在 .gitignore 里（照抄现有仓库的做法）
├── keys.example.py    # keys.py 的模板，可提交
├── pyproject.toml     # 依赖声明
└── .gitignore
```

| 文件 | 职责 | 什么时候要动它 |
|---|---|---|
| `README.md` | 上手入口：定位、快速开始、结构一览 | 改安装方式、加开关时 |
| `llm.py` | 造 `OpenAIChatCompletionClient` 实例 | 换模型、换 key |
| `tools.py` | 工具函数 + 注册表 | 加工具、换搜索后端 |
| `persona.py` | **三个 LLM agent 的全部人格** | 改职责、改口吻、加减 agent |
| `main.py` | 把 agent 组成队、定终止条件 | 换拓扑、调轮数上限 |
| `keys.py` / `keys.example.py` | key 与其模板 | 只在换 key 时 |
| `docs/agents/*.md` | 每份的 §8 是 `persona.py` 里那两段 prompt 的出处 | 改人格前先改它 |

**分层的意义**：`persona.py` 不认识 AutoGen 的 team，`main.py` 不认识
system_message 的内容。改 agent 性格不用碰组队逻辑，换拓扑不用碰人格。

---

## 四个 Agent

### 划界判据（最重要的一条）

新手的通病是按**角色名**拆 agent——「产品经理 / 程序员 / 测试」，
结果拆出五个 agent，它们工具一样、看到的信息一样，只是在互相客气
「好的，我来做」，纯烧 token。

**硬判据：工具集不同，或信息视野不同。两者都相同的，就该是同一个 agent。**

按这条，旅行规划只需要四个：

| Agent | 工具 | 独有的信息/权力 | 结构化产出 |
|---|---|---|---|
| `researcher` | **search**（唯一有联网权的） | 原始素材 | 素材清单（含来源） |
| `planner` | 无 | 编排能力 | 逐日行程表 |
| `critic` | **calculator** | 对抗立场 | 问题清单 / `APPROVED` |
| `user` | — | 真实需求、拍板权 | 初始约束、中途决策 |

### researcher —— 资料员

唯一的联网者。`search` 只挂给它一个人，这是刻意的：如果 planner 也能搜，
它就会自己查而不去问 researcher，群聊立刻退化成三个各自为战的单 agent。

system_message 要点：

```
你是资料员。用 search 工具查交通、住宿、景点、门票价格。
一次任务需要多轮搜索（往返交通 / 住宿区域 / 景点清单 / 票价），
不要一次搜完就交差，把每个方面都查到再汇总。

输出一份素材清单，每条注明来源。不要评论，不要复述别人说过的话，
不要请求许可，直接干活。
```

**为什么强调「不要复述」「不要请求许可」**：AutoGen 默认的多轮对话礼仪
会让 agent 花大量 token 说「好的，我明白了，那么接下来我会……」。写死禁令
是最有效的止血手段，比调参管用。

### planner —— 编排者

不挂任何工具。它只做一件事：把 researcher 查到的素材编成逐日行程。

```
你是行程编排者。基于资料员给出的素材，排出逐日行程。

硬要求：
- 按地理位置聚类，同一天的活动尽量在相邻区域，避免来回跑
- 标注每项的耗时和花费，花费要与素材一致
- 素材里没有的信息不要编，宁可写「待确认」

直接输出表格，不要复述素材原文。
```

**不挂工具是有意的**：它的信息源就该只有 researcher 的产出。让它能联网，
它就会绕过 researcher 自己查，那 researcher 就白设了。

### critic —— 审查员

**唯一挂计算器的 agent**，也是唯一持对抗立场的。

```
你是审查员。对行程表逐条检查：
1. 时间冲突——同一时段安排了两件事
2. 地理绕路——同一天跨区来回跑
3. 预算超支——各项花费加总是否超过用户预算（用计算器算，不要心算）
4. 幻觉——行程里的景点和价格，是否在资料员的素材清单里出现过

有问题就列出问题清单，说清是哪一条、该怎么改。
只有当所有问题都已解决，才输出 APPROVED。
```

**为什么计算器必须挂在它身上**：算钱是 LLM 最容易翻车的地方，
而这个 agent 的立身之本就是抓错。让它心算等于自废武功。

第 4 条检查是它的杀手锏：**它靠比对 researcher 的原文来抓幻觉**。
这要求素材必须是对话里的可见消息，不能藏在某个 agent 的私有内存里——
这是 AutoGen 里很容易踩的架构坑，见[已知坑](#已知坑与设计取舍)。

### user —— 你的入口

代表你本人。职责是提供初始需求和拍板。

**它不进 `participants`，也不是一个 `UserProxyAgent` 实例**——只是
`HandoffMessage` 里的一个字符串标签。critic 遇到需要人取舍的问题时触发 handoff，
team 停下把控制权交回给你；你回复后 team 继续跑。

它出现的时机只有两个：开局那一句 `task=`，以及中途被 critic 叫到时
（「住宿预算砍一半」/「这个景点不去了」）。

完整机制与踩过的坑见 [docs/agents/user.md](docs/agents/user.md) §2.1。

---

## 对话流程与终止

### 拓扑

**起步用 `RoundRobinGroupChat`，别一上来就 `SelectorGroupChat`。**

固定顺序 `researcher → planner → critic → 循环`，可预测、便宜、好 debug。
researcher 需要搜五六次没问题——AutoGen 的 agent 在**一轮之内**可以连续调多次
工具，直到它自己认为答完为止，不需要为每次搜索单开一轮。

跑通之后再换 `SelectorGroupChat`（LLM 选下一个说话人）做对照实验。
这本身就是个值得记录的实验：同一任务两种拓扑，结果差多少、贵多少。

### 终止条件（不写就是无限跑）

```python
termination = (
    TextMentionTermination("APPROVED")      # critic 认可，正常收工
    | HandoffTermination(target="user")     # 需要人拍板，交回控制权
    | MaxMessageTermination(15)             # 兜底保险丝
)
```

**三重保险**。`TextMentionTermination` 依赖 critic 老实输出 APPROVED，而 LLM
不一定老实；`HandoffTermination` 是主观取舍的出口（见[修正 2](#修正-2--user-不靠-userproxyagent-进队靠-handofftermination)）；
`MaxMessageTermination` 是最后那根保险丝。

配套地，critic 的 system_message 里必须写死
「**只有**当所有问题都已解决，才输出 APPROVED」——
否则它会在一堆问题没解决时就客气地说「基本没问题，APPROVED」。

**`APPROVED` 必须裸着独占一行**：`TextMentionTermination` 按**子串**匹配，
写成「尚未 APPROVED」会误触发终止。完整说明见
[docs/agents/critic.md](docs/agents/critic.md) §8.3。

**用 `|` 不要用 `&`**：`&` 要求两个条件同时满足，那要跑到 15 轮才停。

### 启动

```python
result = await team.run(task="帮我规划国庆去成都玩三天，两个人，预算 4000")
print(result.messages[-1].content)
```

---

## 增量搭建路线（5 阶段）

**每个阶段都是可运行、可验证的。** 多 agent 项目最容易犯的错是一上来全搭完，
然后面对一坨不工作的东西不知道怎么定位。按这个顺序走，出错时你永远知道
是新加的那一层的问题。

### 阶段 1 · 环境与模型连通性

只写 `llm.py`，加一个十行脚本，让 DeepSeek 回一句话。

**验证点**：不报错，能拿到回复。**特别确认没有 `model_info` 相关的报错**——
这是 DeepSeek 接 AutoGen 的第一个坎（见[已知坑](#已知坑与设计取舍)第 1 条）。

### 阶段 2 · 单 agent 带 search 跑通

写 `tools.py` + `persona.py` 里的 researcher，单独跑它一个。

**验证点**：输出里有**真实可点的网址**，不是编的。
点开一个确认内容对得上——这一步是在验证「工具真的调通了」，
不是「模型假装调通了」。

### 阶段 3 · 两个 agent 进群聊

`researcher` + `planner` 组成 `RoundRobinGroupChat`，**先不设终止条件**，
用 `max_turns=4` 硬停。

**验证点**：planner 的输出里引用了 researcher 查到的**具体内容**
（具体的酒店区域、具体票价），而不是自己编了一套。
如果 planner 在编，说明素材没进到它的上下文里。

### 阶段 4 · 加 critic

拉第三个 agent 进群。**验证点**：critic 至少挑出**一个真问题**
（时间冲突 / 绕路 / 预算超支 / 幻觉）。

如果 critic 一直在说「行程很合理」——那是失败，不是成功。
它的价值全在对抗性上，只会附和的 critic 等于多烧一份钱。
真挑不出问题时，回头查它是不是没看到 researcher 的原始素材。

### 阶段 5 · 终止条件与 user 回路

加上 `TextMentionTermination | HandoffTermination | MaxMessageTermination`，
并给 critic 声明 `handoffs=[Handoff(target="user")]`。

**不要把 `UserProxyAgent` 拉进队**（原稿如此写，已实测推翻，见[修正 2](#修正-2--user-不靠-userproxyagent-进队靠-handofftermination)）。
它的默认 `input_func` 读控制台，进队后 RoundRobin 每转到它就阻塞整个 team。
改用 handoff 后 user 不进 `participants`，续跑时用 `HandoffMessage` 交回给 critic。

**验证点**：连跑三次，每次都在合理轮数内自然结束（而不是撞到 15 轮上限）。
撞上限说明 critic 不肯说 APPROVED，回去改它的 system_message。

```bash
.venv/bin/python main.py --stage 5
```

---

## 想改 X，就动 Y

| 想改什么 | 动哪个文件 | 怎么改 |
|---|---|---|
| 换模型名 | 环境变量 / `keys.py` / `.env` | 设 `DEEPSEEK_MODEL`，不用改代码 |
| 换模型接入方式 | `llm.py` | 换 `OpenAIChatCompletionClient` 的参数 |
| 换某个 agent 的性格/职责 | `docs/agents/*.md` §8 + `persona.py` | 两处一起改：说明书的两个代码块就是 `role_prompt` / `format_contract` |
| 加一个新 agent | `persona.py` + `main.py` | 定一个 `AgentSpec` + 加进 `GROUP_MEMBERS` |
| 加一个新工具 | `tools.py` + `persona.py` | 写函数 + 登记 `ALL_TOOLS` + 写进某个 `AgentSpec.tool_names` |
| 换搜索后端 | `tools.py` | 重写 `search()` 的函数体，签名别动 |
| 换群聊拓扑 | `main.py` | `RoundRobinGroupChat` → `SelectorGroupChat` |
| 调轮数上限 | `main.py` | `MaxMessageTermination(n)` |
| 改终止词 | `main.py` + `persona.py` | 两处要一起改，否则永不终止 |
| 换 agent 说话顺序 | `main.py` | 改 `GROUP_MEMBERS` 的顺序 |
| 改 user 接话方式 | `persona.py` + `main.py` | `handoffs=` 与 `HandoffTermination` 两处配套 |

**最后两行是连体改动**，改一处忘一处会静默地退化成「跑到上限才停」。

---

## 预期对话示意

> ⚠️ **这一节是手写的示意，不是实测输出。** 手册写在代码之前，还没有真实对话
> 可录。它的用途是：搭之前就知道目标长什么样，跑出来之后拿它对。
>
> **代码跑通后，用真实对话替换这一节。**

```
[user]      帮我规划国庆去成都玩三天，两个人，预算 4000

[researcher] 【交通】北京⇄成都 高铁二等座 约 750/人 单程（来源：12306 查询页）
             【住宿】推荐春熙路/太古里一带，快捷酒店约 300-400/晚（来源：xx）
                    或宽窄巷子一带，民宿约 250-350/晚（来源：xx）
             【景点】大熊猫繁育研究基地 门票 55（来源：xx）
                    武侯祠 门票 50（来源：xx）  锦里 免费
                    都江堰 门票 80（来源：xx）
             【市内】地铁+打车 日均约 50/人（来源：xx）

[planner]    Day 1  抵达 + 春熙路太古里（住春熙路）
             Day 2  大熊猫基地（北郊，上午）→ 下午回市区武侯祠+锦里
             Day 3  都江堰一日（西郊）→ 晚返程
             预算：交通 3000 + 住宿 900 + 门票 370 + 市内 300 = 4570
                   ⚠️ 超出预算 570

[critic]     问题 2 条：
             1. 预算超支 570。交通 3000 占 66%，建议改单程高铁+单程机票，
                或住宿降到宽窄巷子民宿（省约 300）。
             2. Day 2 上午熊猫基地在北郊、下午武侯祠在西南，
                中间穿城，建议与 Day 3 的都江堰（同在西/北向）对调。
             另：行程内所有景点和票价均能在资料员素材中查到，无幻觉。

[user]       预算砍到 4000 以内，住宿改民宿

[planner]    （重排后的行程表）… 合计 3980

[critic]     复核：无时间冲突，地理聚类合理，合计 3980 未超预算，
             景点价格与素材一致。APPROVED
```

**注意 critic 那两条问题的形状**：它说的不是「行程不错」，
而是**具体的、可验证的、带数字的**问题。跑出来的 critic 如果话风不是这样，
就是没进入状态。

---

## 已知坑与设计取舍

### 1. DeepSeek 接 AutoGen：必须显式声明模型能力

**这是第一个会撞上的坑。** AutoGen 不认识 DeepSeek 的模型名，不给它能力声明，
它会拒绝注册工具，报错信息还不直白。

```python
model = OpenAIChatCompletionClient(
    model="deepseek-flash",                    # ← 不是 deepseek-chat，见修正 1
    base_url="https://api.deepseek.com",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model_info={
        "vision": False,
        "function_calling": True,      # ← 不写这个，工具挂不上
        "json_output": True,
        "family": "unknown",
        "structured_output": False,
    },
)
```

**`function_calling: True` 比原以为的更硬。** 实测 `AssistantAgent.__init__` 里有
这么一行：

```python
raise ValueError("The model does not support function calling, which is needed for handoffs.")
```

省掉它，`critic` 这个 agent 直接造不出来——因为它要挂 handoff 通道。

**取舍**：`family` 只能填 `"unknown"`，意味着 AutoGen 会走保守路径，
部分针对特定模型族的优化拿不到。可接受。

### 2. 0.2 与 0.4+ 的 API 不能混用

网上（尤其中文）大量教程还在 0.2 时代。看到 `ConversableAgent`、
`GroupChat`、`GroupChatManager`、`pyautogen` 这些名字，**整篇作废**，
别试图改成新 API，直接换一篇。两代的抽象完全不是一回事。

### 3. 群聊退化成「互相客气」

最高频的失败形态：agent 之间隔空行礼——「好的，我来做」「辛苦了，接下来我」
「感谢，那我补充一点」。十几轮过去没有任何实质产出。

**成因**：system_message 只写了职责，没写**输出格式约束**。
LLM 在群聊里天然倾向于社交化回复。

**对策（按有效性排序）**：
1. 每个 agent 的 system_message 里写死「直接输出成果，不要复述别人、不要请求许可」
2. 要求**结构化产出**（清单/表格/问题列表），不是对话
3. 降低对话轮数上限

### 4. critic 只会附和

比第 3 条更隐蔽：critic 确实在说话，但说的都是「行程安排合理，考虑周到」。
表面上流程在转，实际上这个 agent 毫无价值。

**成因**：多办是它看不到 researcher 的原始素材，没法做幻觉比对，
只能对 planner 的输出泛泛而谈。

**对策**：确认素材是在**对话消息**里，不是在某个 agent 的私有内存里。
AutoGen 的群聊是「所有消息广播给所有参与者」，所以素材只要被 researcher
**说出来**了，critic 就看得见。如果你用了某种自定义的消息过滤，就是这里出的问题。

### 5. 上下文爆炸

群聊里每条消息都会累加进后续每个 agent 的上下文。researcher 的素材清单
本身就很长（五六个网页摘要），几轮下来 token 会涨得很快。

**取舍**：现阶段**不做优化，先接受**。等确实贵到心疼了再上
`SelectorGroupChat` 的 `candidate_func`（限制候选）、或给 agent 加消息裁剪。
过早优化会让「跑不对」和「省 token」两个问题缠在一起，没法 debug。

### 6. 幻觉检查（critic 的第 4 条）依赖素材完整性

critic 只能比对「素材里有没有」，没法判断「素材本身对不对」。
如果 researcher 搜回来一个错的票价，critic 会认为它是真的。

**取舍**：这是**能力边界，不是 bug**。所以 researcher 的每条素材都要求
注明来源链接——最终把关的是你本人，来源链接是给你点的。

### 7. RoundRobin 的僵硬

固定顺序意味着：critic 挑出问题后，下一个说话的是 researcher 而不是 planner。
流程上有点绕。

**取舍**：故意的。**先要可预测，再要聪明。** 阶段 1-4 都用 RoundRobin，
等整个流程跑通了、知道「正常的对话长什么样」了，再换 `SelectorGroupChat`
做对照。没有基准就没法判断换拓扑是变好了还是变差了。

### 8. 成本不可忽略

一次完整对话大约十几轮 LLM 调用，比 `plan-solve-agent-practice` 里
跑一次 PS 贵一个量级（那边是规划一次 + 逐步执行）。这是多智能体的固有代价。

**取舍**：这是它的定位决定的——它买的是「对抗性」，
即 critic 能推翻 planner。流水线型的任务不该用它，见下节。

---

## 和 PS 框架的对照

两个项目是姊妹关系，演示两条路线：

| | plan-solve-agent-practice | autogen-travel-agent |
|---|---|---|
| 架构 | 一个 agent 内部三阶段 | 多个 LLM 实例开会 |
| 类比 | 一个人自问自答 | 几个人开会 |
| 上下文 | 共享（`ExecutionState`） | 各自独立，靠消息广播 |
| 成本 | 低 | 高一个量级 |
| 可控性 | 高，计划是显式的 | 低，对话走向不预定 |
| 强项 | **流水线**：步骤确定的任务 | **对抗**：需要多视角互相挑战 |
| 弱项 | 没有对抗，错了没人拦 | 发散、贵、难 debug |

**选型判据：流水线用 PS，多视角对抗用 AutoGen。**

旅行规划这个任务恰好两半都有——**查资料是流水线，挑毛病是对抗**。
所以理论上最省的架构是混合的：用 PS 收资料，再拉一个 critic agent 来审。

**但本项目刻意不做混合**，保持纯 AutoGen。理由：第一次学群聊编排，
把两套东西拼在一起，出问题时你分不清是 AutoGen 用错了还是接缝处错了。
混合方案留作下一个项目。

---

## 已实测修正

本节记录本手册中**已被实际运行或查证推翻**的说法。上面的原文已就地改过，
这里留一份集中的变更说明，方便回看「当初以为是什么、实际是什么」。
新发现偏差就往这里加。

### 修正 1 · 模型名是 `deepseek-flash`，不是 `deepseek-chat`

- **原稿**：`model="deepseek-chat"`
- **实测**：`api.deepseek.com` 只认 `deepseek-flash` 与 `deepseek-v4-pro`
  （后者是推理模型，慢且贵）。`deepseek-chat` 会 400。
- **已落地**：[llm.py](llm.py) 的 `DEFAULT_MODEL = "deepseek-flash"`，
  可用 `DEEPSEEK_MODEL` 覆盖（环境变量 / `keys.py` / `.env` 三处任一）。

### 修正 2 · user 不靠 `UserProxyAgent` 进队，靠 `HandoffTermination`

- **原稿**：阶段 5「把 `UserProxyAgent` 拉进队」
- **实测**：`UserProxyAgent` 的默认 `input_func` 读控制台。一旦它在
  `participants` 里，RoundRobin 每转到它就**阻塞整个 team**，`run_stream`
  遇到 `UserInputRequestedEvent` 不会终止，只是挂着等输入。官方文档明说这会让
  team 处于**「不稳定状态，无法保存或恢复」**，只建议用于短的即时交互。
- **改用**：`critic` 声明 `handoffs=[Handoff(target="user")]`，终止条件加
  `HandoffTermination(target="user")`。user 不进 `participants`，只是
  `HandoffMessage` 的 `source` 字符串。
  **续跑必须用 `HandoffMessage`**——直接传字符串会报
  `ValueError: The existing handoff target user is not one of the participants`。
- **已落地**：[main.py](main.py) 的 `build_termination()` 与 `stage5_full()`；
  完整说明见 [docs/agents/user.md](docs/agents/user.md) §2.1。

### 修正 3 · 装依赖用 uv，Python 用系统的 3.14

- **原稿**：`pip install "autogen-agentchat==0.7.*" ...`
- **实测**：这台机器**没有 pip 模块**（`python3 -m pip` 报 No module named pip），
  本地也没装任何 Python 3.12。而 `uv venv --python 3.12` 会去 GitHub 拉独立构建
  ——GitHub 在此环境被墙，它会**静默卡死**（进程活着、stdout 为空、`.venv` 不出现）。
  PyPI 直连正常，只有 GitHub 不通。
- **改用**：

  ```bash
  uv venv --python /usr/bin/python3.14     # 显式给系统解释器，避开一切下载
  uv pip install "autogen-agentchat==0.7.*" "autogen-ext[openai]==0.7.*" "requests>=2.31"
  ```

- **顺带确认**：`autogen-agentchat==0.7.5` 在 Python 3.14 上装得上也跑得通，
  不需要为它降级 Python。`api.deepseek.com` 与 `api.tavily.com` 直连都通，
  不用配代理。

### 修正 4 · `TerminationCondition` 是 async 的

- **原稿**：无（原稿没提）
- **实测**：0.7.5 里 `reset()` 和 `__call__()` **都是协程**。同步调用会拿到一个
  coroutine 对象——**它恒为真值**，会让「终止条件是否命中」的检查静默假通过。
- **已落地**：[main.py](main.py) 的 `selftest()` 里用 `await`。team 内部自己
  await，只有手写测试时要注意。

### 修正 5 · 目录比原稿多了几个文件

原稿的目录树只有 6 个文件。实际还多了：`docs/agents/`（四份说明书）、
`keys.example.py`（`keys.py` 的模板，可提交）、`pyproject.toml`、`.gitignore`。

### 修正 6 · `max_tool_iterations` 默认 1，带工具的 agent 永远写不出结论

**这是本项目最危险的一个坑——它让系统「看起来在跑」。**

- **原稿**：无（原稿完全没提这两个参数）
- **实测**：`AssistantAgent` 的 `max_tool_iterations` 默认是 **1**，
  `reflect_on_tool_use` 默认解析为 **False**。两者叠加，**任何带工具的 agent 都写不出
  自己的输出**。

  `_process_model_result` 的循环是这样：模型第一轮若返回工具调用（`content` 不是
  字符串），执行完工具后 `loop_iteration == max_tool_iterations - 1` 立刻成立并
  `break`，随后**自动生成一个 `ToolCallSummaryMessage` 收尾**；默认又没有事后反思，
  模型根本没机会看到工具返回了什么。

- **表现**（每一条都极具迷惑性）：
  - `critic` 调完 `calculator` 就没下文，永远不会说 `APPROVED` → 阶段 4/5 必然撞
    15 轮上限
  - `researcher` 的「素材清单」其实是**搜索结果原样拼接**，格式契约里那条
    `【类别】内容 价格（来源：URL）` 从未被执行过
  - **只有 `planner` 看着正常**——它没有工具，`content` 直接就是字符串
- **后果最阴的地方**：planner 照抄那堆原始搜索结果，于是阶段 3 的「溯源率」检查
  反而拿到 **100%**。是阶段 4 那句「critic 没输出结论」戳穿了整条链——这正好印证了
  分阶段验证的价值：**只有带工具的 agent 会暴露它**。
- **改用**：`build_agent()` 里显式设 `max_tool_iterations=5`、`reflect_on_tool_use=True`，
  并加进离线自测锁死这两个值，防止日后被改回默认。
- **修后实测**：`researcher` 从 1 轮搜索变 3 轮，输出变成逐条带来源的分段清单
  （147 处 `（来源：）`）；`critic` 开始输出 `问题 N 条：` 并正确 handoff；推理内容
  独立成 `ThoughtEvent`，不再混进正文。

### 修正 7 · `TextMentionTermination("APPROVED")` 会被推理泄露误触发

- **原稿**：`TextMentionTermination("APPROVED")`
- **实测**：它按**子串**匹配任意消息。而实测 critic 会把思考写进消息正文（推理泄露），
  于是只要那段思考里出现 `APPROVED`——**哪怕是否定句**——对话就提前终止。

  实测见到的形态：

  ```
  All checks pass: 3730 ≤ 4000 … No time conflicts, no geographic detour.
  Output APPROVED.
  ```

  那次侥幸蒙对（泄露的思考恰好也是「通过」）。方向反过来——「我还不能输出 APPROVED」
  ——就是**静默的错交付**：一份没审完的行程被当成通过的交付给你。

  这跟 [critic.md](docs/agents/critic.md) §8.3 反模式 1 是**同一类问题但来源不同**：
  那条管的是「别把禁用词写进 prompt」，而这里的脏数据来自**模型自己**，所以那条对策
  盖不住——必须靠代码兜。
- **改用**：[main.py](main.py) 的 `ExactTextTermination`——只在「critic 发出的**一条
  消息恰好等于** `APPROVED`」时终止。这等于让代码去强制执行 critic.md §8.2 本来就
  写着的契约（「该行必须只有 APPROVED 这八个字符」），而不是指望模型自觉。
- **回归**：离线自测第 [6] 节现在覆盖三种误触发（推理泄露、否定句、非 critic 来源）
  与两种正触发（裸词、带空白）。

---

## 参考

- AutoGen 0.5.4 teams API 文档 — https://microsoft.github.io/autogen/0.5.4/reference/python/autogen_agentchat.teams.html
- AutoGen 0.4 多智能体生产指南 — https://scaled2c.com/blog/multiagent-systems-aiops/autogen-04-multiagent-framework-production-guide.html
- Microsoft Agent Framework 现状（迁移目标）— https://atlan.com/know/ai-agent/microsoft/agent-framework/
- AutoGen 维护模式说明 — https://www.forbes.com/sites/janakirammsv/2026/04/06/microsofts-agent-stack-confuses-developers-while-rivals-simplify/

兄弟项目：`../plan-solve-agent-practice/`（PS 框架）、`../react agent/`（ReAct 版）
