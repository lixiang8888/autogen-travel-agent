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
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
```

安装：

```bash
pip install "autogen-agentchat==0.7.*" "autogen-ext[openai]==0.7.*"
```

注意 `autogen-core` / `autogen-agentchat` / `autogen-ext` 是三个包，
0.4 之后拆开的。**版本号要一致**，混装会出莫名其妙的导入错误。

---

## 目录结构与职责

对齐 `plan-solve-agent-practice` 的分层，以后在两个项目之间搬东西不用重新理解布局。

```
autogen-travel-agent/
├── BUILD.md          # 本文件
├── llm.py            # 模型客户端：DeepSeek 走 OpenAI 兼容协议
├── tools.py          # 工具：search（Tavily）
├── persona.py        # 四个 agent 的定义：名字、system_message、挂哪些工具
├── main.py           # 组队 + 跑：拓扑、终止条件、入口
└── keys.py           # key，已在 .gitignore 里（照抄现有仓库的做法）
```

| 文件 | 职责 | 什么时候要动它 |
|---|---|---|
| `llm.py` | 造 `OpenAIChatCompletionClient` 实例 | 换模型、换 key |
| `tools.py` | 工具类 + 注册表 | 加工具、换搜索后端 |
| `persona.py` | **四个 agent 的全部人格** | 改职责、改口吻、加减 agent |
| `main.py` | 把 agent 组成队、定终止条件 | 换拓扑、调轮数上限 |
| `keys.py` | key | 只在换 key 时 |

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

`UserProxyAgent`，代表你本人。职责是提供初始需求和拍板。

它不进「干活」的循环，只在 critic 提出问题、需要人做取舍时接话
（「住宿预算砍一半」/「这个景点不去了」）。

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
    TextMentionTermination("APPROVED")   # critic 认可
    | MaxMessageTermination(15)          # 兜底
)
```

**双保险**，缺一不可。`TextMentionTermination` 依赖 critic 老实输出 APPROVED，
而 LLM 不一定老实；`MaxMessageTermination` 是那根保险丝。

配套地，critic 的 system_message 里必须写死
「**只有**当所有问题都已解决，才输出 APPROVED」——
否则它会在一堆问题没解决时就客气地说「基本没问题，APPROVED」。

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

加上 `TextMentionTermination | MaxMessageTermination`，
把 `UserProxyAgent` 拉进队。

**验证点**：连跑三次，每次都在合理轮数内自然结束（而不是撞到 15 轮上限）。
撞上限说明 critic 不肯说 APPROVED，回去改它的 system_message。

---

## 想改 X，就动 Y

| 想改什么 | 动哪个文件 | 怎么改 |
|---|---|---|
| 换模型 | `llm.py` | 换 `OpenAIChatCompletionClient` 的参数 |
| 换某个 agent 的性格/职责 | `persona.py` | 改对应的 `system_message` 字符串 |
| 加一个新 agent | `persona.py` + `main.py` | 定义 + 加进 `participants` |
| 加一个新工具 | `tools.py` + `persona.py` | 写函数 + 挂到某个 agent 的 `tools=` |
| 换搜索后端 | `tools.py` | 重写 `search()` 的函数体，签名别动 |
| 换群聊拓扑 | `main.py` | `RoundRobinGroupChat` → `SelectorGroupChat` |
| 调轮数上限 | `main.py` | `MaxMessageTermination(n)` |
| 改终止词 | `main.py` + `persona.py` | 两处要一起改，否则永不终止 |
| 换 agent 说话顺序 | `main.py` | 改 `participants` 列表的顺序 |

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

**这是第一个会撞上的坑。** AutoGen 不认识 `deepseek-chat` 这个模型名，
不给它能力声明，它会拒绝注册工具，报错信息还不直白。

```python
model = OpenAIChatCompletionClient(
    model="deepseek-chat",
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

## 参考

- AutoGen 0.5.4 teams API 文档 — https://microsoft.github.io/autogen/0.5.4/reference/python/autogen_agentchat.teams.html
- AutoGen 0.4 多智能体生产指南 — https://scaled2c.com/blog/multiagent-systems-aiops/autogen-04-multiagent-framework-production-guide.html
- Microsoft Agent Framework 现状（迁移目标）— https://atlan.com/know/ai-agent/microsoft/agent-framework/
- AutoGen 维护模式说明 — https://www.forbes.com/sites/janakirammsv/2026/04/06/microsofts-agent-stack-confuses-developers-while-rivals-simplify/

兄弟项目：`../plan-solve-agent-practice/`（PS 框架）、`../react agent/`（ReAct 版）
