# user —— 需求方与拍板人

> 本文是 [架构说明](../architecture.md) 的展开：`user` 的角色、边界与产出契约。

**体例说明**：`user` 是 `UserProxyAgent`，代表你本人，**不是 LLM agent**——它没有 `system_message`、没有工具、没有可调的 prompt。所以本文保留与另外三份相同的 8 个节名，但每节按**人机接口**重新定义。§8 不给代码块，给的是你自己该怎么说话。

## 1. 角色

你本人。需求方 + 拍板人。系统里唯一有真实偏好和真实约束的一方——预算是不是真的 4000、三天会不会太赶、都江堰值不值得跑一趟，只有你知道。

它不是第四个「干活的 agent」。另外三个是执行者，它是**输入源和裁决者**。

## 2. 工作场景

业务场景：你输入一句自然语言需求，比如「帮我规划国庆去成都玩三天，两个人，预算 4000」。

**群聊位置：不进 `participants`。** 这是它和另外三个最大的区别。循环是 `researcher → planner → critic → 循环`，`user` 不在里面。

它出现的时机只有两个：

- **开局一次**：`team.run(task=...)` 里那句需求，就是你唯一的输入
- **中途被叫到**：`critic` 提出一个需要人做取舍的问题时（如「预算超支 570，是砍住宿还是改交通」）——这类问题没有客观答案，需要你拍板

**实际看得到**：群聊全部消息——所有素材、行程表、问题清单。

### 2.1 接话靠 HandoffTermination，不是靠 UserProxyAgent

⚠️ **不要把 `UserProxyAgent` 拉进队。** 这是个实测过的坑：

`UserProxyAgent` 的默认 `input_func` 读控制台。一旦它在 `participants` 里，`RoundRobin` 每转到它就会**阻塞整个 team**，`run_stream` 遇到 `UserInputRequestedEvent` 不会终止，只是挂着等你敲键盘。官方文档明说这会让 team 处于**「不稳定状态，无法保存或恢复」**，只建议用于「短的、即时的交互」——你不在电脑前就永久卡死，连断点续跑都做不了。

**正确做法：让 team 主动停下，把控制权交回给你。** 三部分：

1. 给 `critic` 声明一条 handoff 通道（写法见 [critic.md](critic.md) §6）
2. 终止条件加一条 `HandoffTermination(target="user")`
3. 需要你拍板时 `critic` 触发 handoff，team 立即停下

```python
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination, MaxMessageTermination

termination = (
    TextMentionTermination("APPROVED")    # critic 认可，正常收工
    | HandoffTermination(target="user")   # 需要人拍板，交回控制权
    | MaxMessageTermination(15)           # 兜底保险丝
)
```

**续跑有硬性要求**：handoff 停下之后不能直接 `team.run(task="预算砍到 4000")`——会报 `ValueError: The existing handoff target user is not one of the participants`。必须用 `HandoffMessage` 指定交回给谁（见 §7 第 4 步）。

**两个附带结论**：

- **`user` 不必是一个 AutoGen agent 实例。** handoff 方案里它只是 `HandoffMessage` 的 `source` 字符串，不需要 `UserProxyAgent`，也不需要 `input_func`。这是 `HandoffTermination` 相比「拉进队」最实在的好处。
- **这套机制依赖 function calling。** handoff 是模型生成一次工具调用来触发的，所以 `critic` 的模型必须支持工具调用——`llm.py` 给 DeepSeek 显式打开 `function_calling: True`，正好是这个机制的前提（也是为什么那条不能省）。

> 来源：[Human-in-the-Loop 教程](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html) · [Issue #5599 阻塞状态](https://github.com/microsoft/autogen/issues/5599) · [Discussion #5623 续跑报错](https://github.com/microsoft/autogen/discussions/5623)

## 3. 目标

让最终行程落在你**真实可接受**的范围内。不是「最优解」，是「你能照着走完的那一版」。

`critic` 保证的是内部一致性（不冲突、不绕路、不超预算、不编造）；`user` 保证的是外部有效性（这趟旅行你真的想去、真的去得成）。两者缺一不可。

## 4. 能力

「能力」在这一节指**你手里的牌**：

| 牌 | 说明 |
|---|---|
| 改约束 | 改预算、改天数、改人数、改出发地——一句话就能重塑整个解空间 |
| 砍项目 | 「这个景点不去了」——`planner` 会重排 |
| 换项目 | 「换成室内的地方」——比砍掉更能保住行程完整度 |
| 否决重排 | 「这样更绕了，回到上一版」 |
| 直接叫停 | 流程跑偏时中断，改完需求重来 |

**你唯一没有的牌**：直接改 `planner` 的表。你改的是**约束**，重排是它的事。想「我就想住春熙路」，那是在改约束；想「把 Day 2 的下午挪到上午」，那是替 `planner` 干活——说成约束（「Day 2 上午我想睡懒觉」）效果更好。

## 5. 职责

| 该做 | 不该做 |
|---|---|
| 开局给一次完整需求：目的地、天数、人数、预算、出发地 | 开局给一半，跑到中途再补约束 |
| 被叫到时给**一个明确决定** | 在流程中途插话（「顺便问下……」「我觉得还行」） |
| 决定里带数字（「砍到 4000」「少玩一天」） | 给开放式反馈（「再优化一下」）——那会让 `planner` 空转一轮 |
| `APPROVED` 后取最终行程表 | 在 `critic` 还没审完时催最终答案 |

**「不该做」第二条是最容易踩的**：`RoundRobin` 是固定顺序的循环，你中途插一句，下一轮说话人不会因此变成你——你的话只会被塞进所有人的上下文，白跑一轮还可能把话题带偏。

**该做第三条的理由**：模糊的决策会被 LLM 礼貌地接受然后原样保留。说「再优化一下」，`planner` 会重排一版几乎一样的表；说「住宿砍到 300/晚以内」，它才知道结构该怎么变。

## 6. 工具

**没有工具。**

采用 `HandoffTermination` 方案后，`user` **根本不需要是一个 AutoGen agent 实例**——它只是 `HandoffMessage` 里的一个字符串标签：

```python
from autogen_agentchat.messages import HandoffMessage

# 需要你拍板时由 critic 触发 handoff；你回复时构造这条消息续跑：
task = HandoffMessage(source="user", target="critic", content="预算砍到 4000 以内，住宿改民宿")
```

如果因为别的原因确实要建一个 `UserProxyAgent`（比如以后要接 Web UI，用自定义 `input_func`），它也不该挂 `tools=`：

```python
user = UserProxyAgent(name="user")   # 不传 tools
```

**为什么它不该有工具**：它一旦能联网或能算账，就变成了第四个干活的 agent，`researcher` 和 `critic` 的独有性同时被打破，拓扑失去意义（划界判据：工具集不同，或信息视野不同）。

## 7. 工作流程

1. **开局**：在 `team.run(task=...)` 里给一句完整需求。约束一次说全——目的地、天数、人数、预算、出发地、特殊要求（带老人？不吃辣？）
2. **循环跑起来后不介入**。让 `researcher → planner → critic` 自己转
3. **team 停下 = 该你说话**。判据是停下来的原因：`stop_reason` 为 `Handoff to user from critic detected.` 时，说明 `critic` 遇到了需要人取舍的问题，读它的问题清单，给一句话决策。决策里必须带一个具体的数或一个具体的动作
4. **续跑必须用 `HandoffMessage`**：

   ```python
   from autogen_agentchat.messages import HandoffMessage

   task = HandoffMessage(source="user", target="critic", content="预算砍到 4000 以内，住宿改民宿")
   result = await team.run(task=task)
   ```

   交回给**触发 handoff 的那个 agent**（这里是 `critic`），不是 `planner`——`critic` 会带着你的决定重新审一遍，再决定是通过还是让 `planner` 重排。
5. **收到 `APPROVED`**：读最后那版行程表。注意 `print(result.messages[-1].content)` 拿到的是 `critic` 的 `APPROVED`，要看行程表得往前翻到 `planner` 的最后一条消息
6. **不满意**：不是重跑一遍，而是补约束再跑一轮——「预算不变，但我不想早起，每天 10 点后出发」

## 8. 输出格式

`user` 不套「角色 prompt + 格式契约」的两段式（它没有 prompt）。它的「输出格式」就是**你怎么说话**。

**拍板句式：一个约束 + 一个动作，一句话说完。**

| 场景 | 好的说法 | 坏的说法 |
|---|---|---|
| 预算超了 | 「预算砍到 4000 以内，住宿改民宿」 | 「有点贵，再想想办法」 |
| 太赶了 | 「砍掉都江堰，三天只玩市区」 | 「行程是不是有点满？」 |
| 想换住的地方 | 「住宽窄巷子那边，安静点」 | 「换个住的地方吧」 |
| 不满意要重排 | 「Day 2 和 Day 3 对调，我不想第一天就跑远」 | 「重新排一版」 |
| 认可 | 「就这样，可以了」 | 「嗯，还不错」 |

**规律**：好说法里都有一个**可执行的动作**（砍掉 X / 改成 Y / 对调到 Z）。坏说法里只有感受。后者会换来一版几乎一样的表——LLM 会礼貌地接受模糊反馈，然后照原样再输出一遍。

**开局需求的推荐模板**：

```
帮我规划<节假日>去<目的地>玩<N>天，<M>个人，预算<X>，<出发地>出发。<特殊要求>
```

例：`帮我规划国庆去成都玩三天，两个人，预算 4000，北京出发。不想太赶，每天能睡到自然醒最好。`

---

**相关**：[researcher.md](researcher.md) · [planner.md](planner.md) · [critic.md](critic.md)（三个干活的 agent）
