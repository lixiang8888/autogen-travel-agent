# AutoGen 多智能体旅行规划

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

配三个 key（优先级：环境变量 > `keys.py` > 项目根目录的 `.env`）：

```bash
cp keys.example.py keys.py    # 然后填 DEEPSEEK / TAVILY / AMAP 三个
```

各自去哪申请、有哪些坑（高德那个特别容易建错），都写在 [keys.example.py](keys.example.py) 里。

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
python launcher.py                   # 起服务，并自动打开浏览器
python launcher.py --demo            # 离线假数据：不用 key、不联网，几秒看完整个回路
.venv/bin/python make_shortcut.py    # 重建桌面图标（换了机器才需要）
```

**零新增依赖**（只用标准库），不用装 tkinter。WSL2 会把 localhost 转发到 Windows，
所以直接用 Windows 浏览器开 `http://localhost:8765` 就行——不用记 WSL 的 IP。

页面上：写需求点「开始」，中间按 agent 分色滚出对话，critic 要你拍板时下面的输入框会
亮起来、问题贴在框上面，回一句话继续。工具调用缩进成等宽灰字，从正文里退下去。

`launcher.py` 是**可选的**：删掉它 `--selftest` 照样绿——它单向依赖 `main.py`，反过来
不成立（自测第 [10] 节守着这条红线）。

> 安全：`keys.py` 已在 [.gitignore](.gitignore) 中忽略、不会进 git。若曾把真实 key
> 填进文件并外传过，请到对应控制台轮换重置。

### 谁有什么工具

**工具挂载是刻意不对称的**，这是整个拓扑成立的前提：

| Agent | 工具 | 独有的权力 |
| --- | --- | --- |
| `researcher` | `search` + `fetch_page` + `taxi_fare` + `hotel_options` | 全队唯一能联网 |
| `planner` | 无 | —— |
| `critic` | `calculator` + handoff | 唯一能算账、唯一持对抗立场 |
| `user`（你） | 无 | 需求与拍板权 |

researcher 那四把是一件事的四个动作：**`search` 搜、`fetch_page` 读全文、`taxi_fare`
算两点之间的打车钱**（搜索只能给运价表，给不了「从 A 到 B 多少钱」）、**`hotel_options`
列这一带有哪些酒店和档位**（它不返回房价——那查不到，见「踩过的坑」第 13 条）。合起来
才构成「唯一联网者」这一个身份，所以拆开挂给别人和整条搬走一样。

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
tools.py             search + fetch_page(Tavily) + taxi_fare / hotel_options(高德) + calculator
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

**1** 是模型名，**2–6** 是 AutoGen 接线，**7–8** 是本地环境，**9–13** 是取数
（搜索深度 / 契约 / 高德 / 住宿）。

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
9. **搜索摘要会比正文早一步「结束」，而缺的那截恰好是价目表。** 实测门票查询 `basic`
   深度返回的 5 条里**一条真票价都没有**（全是攻略聚合页和一日游产品），`advanced` 能带出
   4 个；出租车那条更典型——命中了正确的运价通知，但摘要**正好断在数字之前**。**改法**：
   深度提到 `advanced`，另加 `fetch_page` 让 researcher 在「搜到链接却没看见数字」时读全文。
   **剩余边界**：价目表常被做成图片，提取不到正文，这种只能换来源。
10. **契约里不能有「查不到数据源的示例数字」。** 原契约示例写着「地铁+打车 日均约50/人」
   ——这个数只能靠模型自己折算，而它带着来源标记，critic 的幻觉比对**查不出**（清单里
   确实写着 50）。缺口被一个自算的日均值盖住，交付出去的是**假的确定性**。**改法**：契约
   改成照抄原始口径（费率、票种、免票条件），planner 的判据从「素材里有没有数字」收紧成
   「能不能直接算成这一项的总额」。
11. **接高德路径规划踩的三个坑**：`taxi_cost` 在 `route` 层级、**不在 `paths[]` 里**；
   带上 `strategy` / `extensions` 反而报 `MISSING_REQUIRED_PARAMS`，最简参数集才对；漏传
   `key` 报的是 `INVALID_USER_KEY`——看着像 key 坏了，其实是没传（第一版只给路径规划加了，
   地理编码那两处漏了，已改成在 `_amap_get` 里统一注入）。**另有一条不是 bug 是事实**：
   它走通用计价模型、不套当地运价文件，同一条路线高德报 71 元 vs 按诸暨运价手算 86 元，
   **差 18%**——所以返回文案里写死了口径声明，不让它被当成确定值。
12. **新建的高德 key QPS 很紧。** `taxi_fare` 一次要发三次请求（两次地理编码 + 一次路径
   规划），连着跑几条就撞 `10021`，而报出来的是「终点搞不定」，看着像地名有问题。已对
   10004 / 10021 做退避重试（1.2 秒 × 3 次）。个人认证能放宽这个限制，顺带把月配额
   提到 15 万次。
13. **酒店实价在网页里根本不存在，能搜到的中文价页还大多是外币。** 起因是实跑里
   `critic` 报「住宿行仍为待确认，而备注里唯一有数字的 `HK$462/晚` 高于用户选定的
   200-300 档」。查下来是三件事叠在一起：

   - **实价取不到。** `fetch_page` 打开携程酒店页（`hotels.ctrip.com/hotels/436016.html`），
     返回的**整页是一张登录表单**——房价按「日期 × 房型」在下单那一刻生成，网页上
     没有静态页。这跟第 11 条是同一类：不是"没写"，是"不存在"。
   - **搜到的是港台 locale 的商圈均价。** 搜 `成都春熙路 酒店 价位`，前 5 条里
     `hk.trip.com` 报 **HK$493 / HK$583**、facebook 报 **RM60-100**，只有天巡的
     `¥61 / ¥335` 是人民币。那些外币数字是 OTA 自己统计的区域均价，和"全季春熙路
     某两晚的大床房"根本不是一个东西——`HK$462` 就是这么进到备注里的。
   - **用户拍板的数字填不进去。** `critic` 的原话是「用户已选定 200-300 元/晚，但住宿项
     金额仍写待确认」——**它在问：用户都定了，为什么还是待确认。** 因为契约只认「素材里
     的数字」，而用户的话不算素材。于是回路空转。

   **改法三条一起**：加 `hotel_options`（高德 POI）查「哪一带、哪一档、哪几家」，并在
   契约里写死「外币 OTA 报价不当房价采信」；住宿改成查档位区间、不查实价；**把「用户
   拍板」确立为花费列的第三类合法来源**——它和第 10 条堵掉的编造有本质区别：编造的数字
   没有来源，「250/晚」的来源是 user 的 handoff 消息，全群可见、可查证。第 10 条当时
   把这两类一起禁掉了，**禁过头了**。

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
