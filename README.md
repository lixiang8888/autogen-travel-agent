# AutoGen 多智能体旅行规划

> 四个 agent 开会，产出一份逐日行程。演示 AutoGen 的群聊编排。

输入一句自然语言需求（「帮我规划国庆去成都玩三天，两个人，预算 4000」），
`researcher → planner → critic` 三个 LLM agent 在群聊里轮流协作；需要人拍板时把控制权
交回给你，最后交出一份逐日行程表。

**它不是什么**：不是 [plan-solve-agent-practice](../plan-solve-agent-practice/) 的升级版。
两者是**并列的姊妹项目**，演示两条路线——那边是「一个 agent 内部三阶段，自问自答」，
这边是「多个 LLM 实例开会」。互不依赖、互不 import。

**为什么挑旅行规划当靶子**：它的对错**你自己一眼能判断**，不需要任何领域知识——
日程有没有时间冲突、有没有东一个西一个来回跑、预算加总对不对、景点是不是编的。
正好用来回答「多 agent 协作到底有没有用」。

它的两个设计目标：

1. **按信息边界拆 agent，不按角色名拆。** 硬判据是「工具集不同，或信息视野不同」，
   两者都相同的就该是同一个 agent。按这条，旅行规划只需要**三个** LLM agent
   （第四个是你本人，不是 LLM）。
2. **改人格不用碰编排。** `persona.py` 不认识 AutoGen 的 team，`main.py` 不认识
   `system_message` 的内容。

想动手改东西看 **[BUILD.md](BUILD.md)**（搭建蓝图：目录职责、5 阶段搭建路线、
「想改 X 就动 Y」、已知坑与**已实测修正**）。想知道每个 agent 为什么长这样，看
**[docs/agents/](docs/agents/)**。

### 快速开始

依赖：**Python ≥ 3.12**（实测 3.14.4 可用）+ `autogen-agentchat` 0.7.5。

```bash
uv venv --python /usr/bin/python3.14     # 显式给系统解释器，理由见「已知的坑」
uv pip install "autogen-agentchat==0.7.*" "autogen-ext[openai]==0.7.*" "requests>=2.31"
```

配 key（优先级：环境变量 > `keys.py` > 项目根目录的 `.env`）：

```bash
cp keys.example.py keys.py    # 然后填进 DEEPSEEK_API_KEY 与 TAVILY_API_KEY
```

跑起来：

```bash
.venv/bin/python main.py --selftest    # 离线自测：不用 key、不联网
.venv/bin/python main.py --stage 1     # 阶段 1：DeepSeek 连通性
.venv/bin/python main.py --stage 5     # 阶段 5：完整流程（会停下来问你拍板）
```

`--stage 1..5` 对应 BUILD.md 的增量搭建路线，每一步的验证点写在输出里。
`--reply "..."` 可**重复给多次**（第 N 次拍板用第 N 个回答），用于非交互跑。

> 安全：`keys.py` 已在 [.gitignore](.gitignore) 中忽略、不会进 git。若曾把真实 key
> 填进文件并外传过，请到 DeepSeek / Tavily 控制台轮换重置。

### 谁有什么工具

**工具挂载是刻意不对称的**，这是整个拓扑成立的前提：

| Agent | 工具 | 独有的权力 |
| --- | --- | --- |
| `researcher` | `search` | 全队唯一能联网 |
| `planner` | 无 | —— |
| `critic` | `calculator` + handoff | 唯一能算账、唯一持对抗立场 |
| `user`（你） | 无 | 需求与拍板权 |

如果 `planner` 也能搜，它就会绕过 `researcher` 自己查，群聊立刻退化成三个各自为战的
单 agent。所以「唯一联网」不是权限设置，是**架构约束**——`docs/agents/planner.md` §6
专门写了「为什么一把都不挂」。

### 什么时候停下来

三条终止条件取**或**（是 `|` 不是 `&`——`&` 要两个同时满足，会一直跑到上限）：

| 条件 | 含义 |
| --- | --- |
| `ExactTextTermination("APPROVED", "critic")` | critic 认可，且**整条消息恰好是** `APPROVED` |
| `HandoffTermination(target="user")` | 需要你拍板，控制权交回 |
| `MaxMessageTermination(15)` | 兜底保险丝 |

第一条为什么不用 `TextMentionTermination`：那个按**子串**匹配任意消息，而 critic 会把
思考写进消息正文，只要那段思考里出现 `APPROVED`（**哪怕是否定句**「我还不能输出
APPROVED」）就会提前终止，把没审完的行程当通过交付。详见 BUILD.md 已实测修正 7。

`user` **不进 `participants`**——它是 `HandoffMessage` 里的一个字符串标签，不是一个
`UserProxyAgent` 实例。理由是后者默认读控制台，会阻塞整个 team 且官方文档说无法保存
恢复。详见 [docs/agents/user.md](docs/agents/user.md) §2.1。

### 结构一览

```
BUILD.md             搭建蓝图（先读这个）
docs/agents/         四个 agent 的说明书：角色、边界、产出契约
llm.py               DeepSeek 客户端（AutoGen 的 OpenAIChatCompletionClient）
tools.py             search(Tavily) + calculator + 注册表
persona.py           三个 LLM agent 的人格（角色 prompt + 格式契约）
main.py              组队、终止条件、5 个阶段入口、离线自测
keys.py              key（不进 git）
keys.example.py      keys.py 的模板
```

分层原则：`persona.py` 不认识 AutoGen 的 team，`main.py` 不认识 `system_message`
的内容。改 agent 性格不用碰组队逻辑，换拓扑不用碰人格。

`persona.py` 里每个 agent 的人格由两段拼成：

```
system_message = 角色 prompt + "\n\n" + 格式契约
```

第二段是给**下游 agent** 读的接口——critic 靠 researcher 的「（来源：URL）」做幻觉
比对，靠 planner 的固定列名逐条检查。单独留着，是为了日后想抽 `protocols.py` 时能
整块搬走。注意 `docs/agents/*.md` 的 §8.1/§8.2 就是这两段的**原文**，改一处必须改
另一处——离线自测第 [8] 节会核对，脱钩就跑不过。

### 已知的坑

完整的七条在 BUILD.md「已实测修正」。刚上手最容易撞的两个：

- **装依赖别用 `uv venv --python 3.12`**。它会去 GitHub 拉独立构建，而 GitHub 在
  某些网络环境下不通，表现为**静默卡死**（进程活着、stdout 为空、`.venv` 不出现）。
  显式给系统解释器路径即可绕开。
- **模型名是 `deepseek-flash`，不是 `deepseek-chat`**。后者已过期会 400。
  要换就设环境变量 `DEEPSEEK_MODEL`，不用改代码。

### 文档

| 文件 | 内容 |
| --- | --- |
| 本文件 | 定位、快速开始、工具挂载、终止条件、结构一览 |
| [BUILD.md](BUILD.md) | 搭建蓝图：5 阶段路线、想改 X 就动 Y、已知坑与已实测修正 |
| [docs/agents/](docs/agents/) | 四个 agent 的说明书（角色 / 能力 / 职责 / 工具 / 流程 / 输出格式） |
| [keys.example.py](keys.example.py) | key 模板与申请地址 |
