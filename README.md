# AutoGen 多智能体旅行规划。

以autogen架构为核心的多智能体履行规划，通过多个智能体之间的交流，从而产出一份逐日行程。

输入一句自然语言需求（「帮我规划国庆去成都玩三天，两个人，预算 4000」），
`researcher → planner → critic` 三个 LLM agent 在群聊里轮流协作；需要人拍板时把控制权
交回给你，最后交出一份逐日行程表。

它的两个设计目标：

1. **按信息边界拆 agent，不按角色名拆。** 硬判据是「工具集不同，或信息视野不同」，
   两者都相同的就该是同一个 agent。按这条，旅行规划只需要**三个** LLM agent
   （第四个是你本人，不是 LLM）。
2. **改人格不用碰编排。** `persona.py` 不认识 AutoGen 的 team，`main.py` 不认识
   `system_message` 的内容。

### 快速开始

依赖：**Python ≥ 3.12**（实测 3.14.4 可用）+ `autogen-agentchat` 0.7.5。

```bash
uv venv --python /usr/bin/python3.14     # 显式给系统解释器，理由见「实测踩过的坑」
uv pip install "autogen-agentchat==0.7.*" "autogen-ext[openai]==0.7.*" "requests>=2.31"
```

配 key（优先级：环境变量 > `keys.py` > 项目根目录的 `.env`）：

```bash
cp keys.example.py keys.py    # 然后填进 DEEPSEEK_API_KEY 与 TAVILY_API_KEY
```

跑起来：

```bash
python main.py "我国庆去诸暨玩，三天，两个人，预算 2000。"   # 直接跑完整流程
python main.py --selftest                                  # 离线自测：不用 key、不联网
```

不传需求就用内置的示例任务（成都三天两晚 4000 元）。

跑起来后：约两分钟终端不动是正常的（researcher 在联网查，一轮要搜七八次）；然后它
**停下来问你**，打出 critic 的问题清单——你敲一句话回答，带具体数字或动作
（「住宿砍到 250 一晚」比「再优化一下」有用），它继续跑；critic 说 `APPROVED` 就结束。

### 图形界面

不想看终端的话，同一个流程有个网页启动器。**双击桌面的「旅行规划」图标就行**——
它会起服务并自动打开浏览器，不用敲任何命令。（重复双击不会起第二个服务，只会把
浏览器指到已经在跑的那个。）

想从命令行起也可以：

```bash
python launcher.py            # 起服务，并自动打开浏览器
python launcher.py --demo     # 离线假数据：不用 key、不联网，几秒看完整个回路
```

要重新创建那个桌面图标（比如换了机器）：

```bash
.venv/bin/python make_shortcut.py
```

**零新增依赖**（只用标准库），也不用装 tkinter。WSL2 会把 localhost 转发到 Windows，
所以直接用 Windows 的浏览器开 `http://localhost:8765` 就行——不用记 WSL 的 IP。

页面上：上面写需求点「开始」，中间按 agent 分色滚出对话，critic 要你拍板时下面那个
输入框会亮起来、问题直接贴在框上面，你回一句话继续。工具调用（`→ search(...)`）会缩进
成等宽灰字，从正文里退下去；「该你拍板了」这类版式性提示只进右上角状态栏，不进对话流。

`launcher.py` 是**可选的**：删掉它，`python main.py --selftest` 照样绿——它单向依赖
`main.py`，反过来不成立（自测第 [10] 节守着这条红线）。

> 安全：`keys.py` 已在 [.gitignore](.gitignore) 中忽略、不会进 git。若曾把真实 key
> 填进文件并外传过，请到 DeepSeek / Tavily 控制台轮换重置。

### 谁有什么工具

**工具挂载是刻意不对称的**，这是整个拓扑成立的前提：

| Agent | 工具 | 独有的权力 |
| --- | --- | --- |
| `researcher` | `search` + `fetch_page` | 全队唯一能联网 |
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
APPROVED」）就会提前终止，把没审完的行程当通过交付。见「实测踩过的坑」第 3 条。

`user` **不进 `participants`**——它是 `HandoffMessage` 里的一个字符串标签，不是一个
`UserProxyAgent` 实例。理由是后者默认读控制台，会阻塞整个 team 且官方文档说无法保存
恢复。详见 [docs/agents/user.md](docs/agents/user.md) §2.1。

### 结构一览

```
README.md            本文件：定位、用法、改哪里、踩过的坑
docs/agents/         四个 agent 的说明书：角色、边界、产出契约
llm.py               DeepSeek 客户端（AutoGen 的 OpenAIChatCompletionClient）
tools.py             search + fetch_page(Tavily) + calculator + 注册表
persona.py           三个 LLM agent 的人格（角色 prompt + 格式契约）
main.py              组队、终止条件、入口、离线自测
launcher.py          网页启动器（可选：删掉它 --selftest 照样绿）
make_shortcut.py     在 Windows 桌面建「点击就启动」的快捷方式（只在 WSL 上有意义）
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

### 想改 X，就动 Y

| 想改什么 | 动哪个文件 | 怎么改 |
| --- | --- | --- |
| 换模型名 | 环境变量 / `keys.py` / `.env` | 设 `DEEPSEEK_MODEL`，不用改代码 |
| 换模型接入方式 | `llm.py` | 换 `OpenAIChatCompletionClient` 的参数 |
| 换某个 agent 的性格、职责 | `docs/agents/*.md` §8 + `persona.py` | 两处一起改：说明书的两个代码块就是 `role_prompt` / `format_contract` |
| 加一个新 agent | `persona.py` + `main.py` | 定一个 `AgentSpec` + 加进 `GROUP_MEMBERS` |
| 加一个新工具 | `tools.py` + `persona.py` | 写函数 + 登记 `ALL_TOOLS` + 写进某个 `AgentSpec.tool_names` |
| 换搜索后端 | `tools.py` | 重写 `search()` 的函数体，签名别动 |
| 换群聊拓扑 | `main.py` | `RoundRobinGroupChat` → `SelectorGroupChat` |
| 调轮数上限 | `main.py` | `MaxMessageTermination(n)` |
| 调最多问几次拍板 | `main.py` | `MAX_HANDOFFS` |
| 改终止词 | `main.py` + `persona.py` | 两处要一起改，否则永不终止 |
| 换 agent 说话顺序 | `main.py` | 改 `GROUP_MEMBERS` 的顺序 |
| 改 user 接话方式 | `persona.py` + `main.py` | `handoffs=` 与 `HandoffTermination` 两处配套 |

最后三行是**连体改动**，改一处忘一处会静默退化成「跑到上限才停」。

### 调试：逐阶段跑

`--stage` 是调试选项，日常不用（不传参数就是完整流程）。**出问题时就按这个顺序往回退**，
每一步的验证点写在输出里：

| 命令 | 验什么 |
| --- | --- |
| `--stage 1` | DeepSeek 连不连得上（应无 `model_info` 报错） |
| `--stage 2` | 只跑 researcher，输出里有没有**真实可点的网址**——点开一个确认内容对得上 |
| `--stage 3` | researcher + planner，planner 有没有引用查到的**具体票价**而非自己编 |
| `--stage 4` | 加 critic，它有没有挑出**真问题**（只会说「行程很合理」就是失败） |

非交互地跑（脚本化、测试用）：`--reply "回答"` 可重复给多次，第 N 次拍板用第 N 个。

### 实测踩过的坑

都已修，记录在此避免重蹈。每条都有对应代码/注释。

1. **模型名是 `deepseek-flash`，不是 `deepseek-chat`。** 后者已过期会 400。
   要换设 `DEEPSEEK_MODEL`，不用改代码。
2. **`max_tool_iterations` 默认 1，带工具的 agent 永远写不出结论。** AutoGen 执行完
   工具后立刻自动生成 `ToolCallSummaryMessage` 收尾，模型看不到工具返回了什么。表现
   极具迷惑性：critic 调完计算器就没下文（永不说 `APPROVED`），researcher 的「素材
   清单」其实是搜索结果原样拼接。**只有 planner 看着正常**（它没工具）。已显式设为 5。
3. **`TextMentionTermination` 会被推理泄露误触发。** critic 会把思考写进消息正文，
   只要思考里含 `APPROVED`（哪怕是否定句）就提前终止。已改成 `ExactTextTermination`
   精确匹配。
4. **`user` 不靠 `UserProxyAgent` 进队。** 它默认读控制台，进队后 RoundRobin 每转到
   它就阻塞整个 team，官方文档说这会让 team 无法保存恢复。改用 `HandoffTermination`。
5. **handoff 工具是零参数的。** 源码就是 `def _handoff_tool() -> str: return self.message`
   ——critic 想问什么**只能写在正文里**。实测它会跳过正文直接调工具，于是用户只看到
   「需要用户拍板」四个字。已在人格里写死「先写问题清单再调工具」，并在 `main.py`
   里往前翻出那段正文给用户看。
6. **`TerminationCondition` 是 async 的。** `reset()` 和 `__call__()` 都是协程；同步
   调用会拿到 coroutine 对象——**它恒为真值**，会让检查静默假通过。
7. **装依赖别用 `uv venv --python 3.12`。** 它会去 GitHub 拉独立构建，而 GitHub 在某
   些网络环境下不通，表现为**静默卡死**（进程活着、stdout 为空、`.venv` 不出现）。
   显式给系统解释器路径即可绕开。PyPI 直连通常正常。
8. **`--reply` 用完会挂死。** 早期实现用完回退到 `input()`，而管道场景下 stdin 可能是
   「开着但不给数据」——不抛 `EOFError` 而是永久阻塞，最后被 timeout 杀掉。已改为用完即停。
9. **搜索摘要会比正文早一步「结束」，而缺的那截恰好是价目表。** 起因是「门票、打车钱
   老是查不到」。实测同一条 `诸暨 出租车 起步价`，`basic` 深度确实命中了正确的
   《运价调整通知》，但摘要**正好断在「有关事项的通知」后面**——数字全在被截掉的部分。
   `诸暨 景点 门票价格` 更差：`basic` 返回的 5 条里一条真票价都没有（全是攻略聚合页和
   一日游产品），`advanced` 能带出 4 个具体票价。**改法**：深度提到 `advanced`，并加
   `fetch_page`（Tavily Extract）让 researcher 在「搜到了链接但没看见数字」时读全文。
   **剩余边界**：政府公告常把价目表做成图片（潮新闻那篇里就有一张 png），提取不到正文——
   这种只能换来源，任何爬虫都无解。
10. **契约自己在教模型编数字。** 格式契约的示例原本写着
   `【市内】地铁+打车 日均约50/人（来源：https://...）`——「日均 50」**没有对应的现实
   数据源**，模型只能自己折算。要命的是它带着「（来源：URL）」，`critic` 的幻觉比对
   （拿行程回查清单）**查不出这种编造**：清单里确实写着 50。于是「查不到打车钱」这个
   缺口被一个自算的日均值盖住，交付出去的是**假的确定性**。**改法**：契约改成照抄原始
   口径（费率、票种、免票条件），`planner` 的判据从「素材里有没有数字」收紧成「能不能
   直接算成这一项的总额」。缺口从此显形为「待确认」，而不是被抹平。

### 设计取舍

有意为之，不是 bug。改之前先想清楚代价。  

- **工具挂载不对称**：见「谁有什么工具」。这是拓扑的地基，不是权限设置。
- **critic 只能比对「素材里有没有」，不能判断「素材本身对不对」。** researcher 搜回一个
  错价，critic 会当成真的。这是**能力边界**，所以每条素材必须带来源链接——最终把关的
  是你点开链接那一刻。
- **RoundRobin 的僵硬**：固定顺序意味着 critic 挑出问题后，下一个说话的按顺序轮转。
  「先要可预测，再要聪明」——没有基准就没法判断换拓扑是变好了还是变差了。
- **上下文会膨胀**：群聊里每条消息都累加进后续每个 agent 的上下文，几轮下来 token 涨得
  很快。**现阶段接受，不做优化**——过早优化会让「跑不对」和「省 token」两个问题缠在
  一起，没法 debug。
- **成本比 PS 框架高一个量级**：一次完整对话十几轮 LLM 调用。多智能体的固有代价，它买
  的是「对抗性」（critic 能推翻 planner）。

### 文档

| 文件 | 内容 |
| --- | --- |
| 本文件 | 定位、用法、改哪里、踩过的坑、设计取舍 |
| [docs/agents/](docs/agents/) | 四个 agent 的说明书（角色 / 能力 / 职责 / 工具 / 流程 / 输出格式） |
| [launcher.py](launcher.py) | 网页启动器：HTTP 接口、线程模型、页面本身（模块 docstring 是主要说明） |
| [make_shortcut.py](make_shortcut.py) | 建桌面快捷方式；顺带解释了为什么这件事不能用 `.bat` 做 |
| [keys.example.py](keys.example.py) | key 模板与申请地址 |
