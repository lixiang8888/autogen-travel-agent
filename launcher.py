# -*- coding: utf-8 -*-
"""
launcher.py —— 网页启动器：把终端里的多智能体旅行规划搬进浏览器
==================================================================

跑法：
    python launcher.py            # 起服务，并自动打开浏览器
    python launcher.py --demo     # 离线假数据：不需要 key、不联网、不烧 token

**日常用法是双击桌面快捷方式**，不用敲命令。快捷方式的目标是

    wsl.exe -d Ubuntu --cd <项目目录> -- .venv/bin/python launcher.py

也就是「起服务 + 开浏览器」两件事一次做完。

两件为「双击」做的事：
1. `_open_browser()` 走 WSL 互操作调 Windows 的 explorer.exe——WSL 里 webbrowser
   模块找的是 Linux 侧的浏览器，基本没装，指望不上。
2. `_existing_instance()` + `/health`：**重复双击不会起第二个服务**，只会把浏览器
   指到已经在跑的那个。少了这一步，第二次双击会顺延到 8766，于是你有两个各自
   为政的页面，而且新那个啥历史都没有。

**零新增依赖**，只用标准库。之所以不做独立窗口（Tk）而做网页：中文排版和换行
交给浏览器，比 Tk 8.6 省心一个量级；而且不用给 venv 装 tkinter。

分层红线
--------
`launcher.py import main`，**`main.py` 绝不 import launcher**。
所以 `python main.py --selftest` 在这个文件不存在、也没有浏览器的机器上照样绿。

线程模型（改这个文件之前先读完这四条）
--------------------------------------
    HTTP 线程池（ThreadingHTTPServer，每个连接一个线程）
       ├─ GET  /events  → 阻塞在自己的订阅队列上，逐条 SSE 写出
       ├─ POST /answer  → session.answers.put(text)
       └─ POST /stop    → stopped.set() + stop.set() + answers.put(_STOP)

    daemon worker 线程：asyncio.run(stage5_full(task, None, hooks=...))
       ├─ on_message / on_notice → bus.publish(...)
       └─ ask → 广播一条 ask 事件，然后阻塞在 answers.get() 上等人回话

1. **worker 线程里一个 HTTP 调用都不许发**——只往队列/广播里放。
2. 状态只走队列与广播。唯一允许跨线程共享的裸状态是「停止」：一个
   `threading.Event` 和一个 `ExternalTermination` 的 bool，都是单向「只写不读回」，
   GIL 下原子，没有竞态窗口。
3. **worker 线程必须 daemon，而且永不 join**——join 会把「关掉页面」变成
   「等 LLM 把这一轮跑完」。
4. 每次点「开始」都新建 model 和 team，停止后不复用。这条顺手绕开了
   `_is_running` 标志、终止条件的 `reset()` 语义、以及 `OrTerminationCondition`
   在已 terminated 时抛 `RuntimeError` 这一堆边角。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from autogen_agentchat.conditions import ExternalTermination

from main import ASK_HINT, DEFAULT_TASK, RunHooks, format_message, stage5_full

#: 哨兵：区分「用户点了停止」和「用户提交了空串」。
#: 后者要触发 main.py 里「（空输入，停止。）」那条分支，不能混为一谈。
_STOP = object()

#: /health 里报的身份。启动时用它认「这个端口上是不是已经有一个我了」——
#: 双击图标两次是很常见的动作，不该因此起了两个服务、开出两个各自为政的页面。
_APP_ID = "autogen-travel-agent-launcher"


def _existing_instance(port: int) -> bool:
    """这个端口上已经有一个我们自己起的启动器吗？

    认的是 /health 里的 _APP_ID，而不是「端口通不通」——不然会把别人占用的
    端口误判成自己人，然后把浏览器指到一个不相干的页面上。
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.6) as resp:
            return json.loads(resp.read()).get("app") == _APP_ID
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False


def _open_browser(url: str) -> None:
    """尽力在**用户的**浏览器里打开这个 URL。

    WSL 里 `webbrowser` 模块基本指望不上——它找的是 Linux 侧的图形浏览器，多半
    没装。真正管用的是走 WSL 互操作去调 Windows 的 `explorer.exe`：它拿 URL 当
    参数时会用 Windows 的**默认浏览器**打开。所以顺序是 Windows 优先、webbrowser 兜底。
    """
    for argv in (["explorer.exe", url], ["cmd.exe", "/c", "start", "", url]):
        exe = shutil.which(argv[0])
        if not exe:
            continue
        try:
            # explorer.exe 成功时也常常返回非 0，所以不 check；超时兜住卡死的情况。
            subprocess.run(
                [exe, *argv[1:]],
                check=False, timeout=10,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
        except (OSError, subprocess.SubprocessError):
            continue
    try:
        webbrowser.open(url)
    except Exception:                                    # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# 1. 广播：worker 发一条，所有开着页面的浏览器各收一份
# ---------------------------------------------------------------------------

class _Bus:
    """极简广播 + 历史回放。

    历史回放是为了**刷新页面不丢内容**：新订阅者一上来先把已有事件补给它。
    事件带一个**单调递增、永不重置**的 seq，前端用它去重，于是重连/重放不会
    把消息渲染两遍。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._history: list[dict] = []
        self._seq = 0

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.append(q)
            replay = list(self._history)
        for event in replay:
            q.put(event)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            self._seq += 1
            event = {**event, "seq": self._seq}
            self._history.append(event)
            subs = list(self._subs)
        for q in subs:
            q.put(event)

    def reset(self) -> None:
        """开始新的一轮：清掉历史，但 **seq 不重置**（否则前端去重会误杀新事件）。"""
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# 2. 一次 run 的状态与出口
# ---------------------------------------------------------------------------

class _Session:
    """一次 run 的全部状态。每次点「开始」新建一个，用完即弃（见模块头第 4 条）。"""

    def __init__(self, bus: _Bus, task: str, demo: bool = False) -> None:
        self.bus = bus
        self.task = task
        self.demo = demo
        self.answers: queue.Queue = queue.Queue()   # 浏览器 -> worker
        self.stop = ExternalTermination()           # 交给 main.py 挂在终止条件上
        self.stopped = threading.Event()            # worker 自己判断该不该收手
        self.finished = threading.Event()
        self._question = ""
        self._thread: threading.Thread | None = None

    # ---- 下面三个方法都在 worker 线程里被 main.py 调用 ----

    def on_message(self, message) -> None:
        formatted = format_message(message)
        if formatted is None:
            return
        self.bus.publish({
            "type": "message",
            "source": formatted.source,
            "text": formatted.text,
            "kind": formatted.kind,
        })

    def on_notice(self, text: str, kind: str) -> None:
        if kind == "question":
            # **不进对话流。** CLI 里它印在终端上，是因为终端没别的地方放；网页有
            # 拍板输入区，它就贴在输入框上方当提示——这也是为什么「该你拍板了」
            # 那条横幅同样只进状态栏。critic 的正文本来就在对话流里，再印一遍是噪声。
            self._question = text
            return
        self.bus.publish({"type": "notice", "text": text, "kind": kind})

    def ask(self, prompt: str) -> str | None:
        """把问题广播出去，然后**阻塞 worker 线程**等人回话。

        和 main.py 里的 `_console_ask` 阻塞在 `input()` 上是同一种形态，
        只是等待对象从 stdin 换成了队列。
        """
        if self.stopped.is_set():
            return None
        self.bus.publish({
            "type": "ask",
            "hint": ASK_HINT,
            "question": self._question,
        })
        self._question = ""
        reply = self.answers.get()
        return None if reply is _STOP else reply

    @property
    def hooks(self) -> RunHooks:
        return RunHooks(
            on_message=self.on_message,
            on_notice=self.on_notice,
            ask=self.ask,
            stop=self.stop,
        )

    # ---- 生命周期 ----

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            if self.demo:
                asyncio.run(_demo_run(self.hooks))
            else:
                # replies 必须是 None 而不是 []——main.py 里 scripted 判的是
                # `replies is not None`，传空列表会在第一次 handoff 就「预设用完」停下。
                asyncio.run(stage5_full(self.task, None, hooks=self.hooks))
        except RuntimeError as exc:
            # 最常见的一种：没配 key（或 key 错了）。main.py 抛出的 RuntimeError 里
            # 带着一整段 traceback——终端里用 `=` 框起来还能读，塞进网页状态栏就是灾难。
            # 所以这里只往页面上放首行，完整内容原样留给终端。
            self._fail("无法启动", exc)
        except Exception as exc:                        # noqa: BLE001
            self._fail("出错了", exc)
        finally:
            self.finished.set()
            self.bus.publish({"type": "done"})

    def _fail(self, prefix: str, exc: BaseException) -> None:
        full = str(exc).strip()
        print(f"\n{prefix}：\n{full}\n", file=sys.stderr, flush=True)
        first = full.splitlines()[0] if full else type(exc).__name__
        if len(first) > 300:
            first = first[:300] + "…"
        self.bus.publish({
            "type": "notice",
            "text": f"{prefix}：{first}（完整堆栈见终端）",
            "kind": "warning",
        })

    def reply(self, text: str) -> None:
        self.answers.put(text)

    def request_stop(self) -> None:
        """优雅停：AutoGen 在下一个 agent 回合边界收手。

        `stop.set()` 只是个 bool 赋值（`ExternalTermination.set` 的源码就一行），
        跨线程写是安全的。同时放掉可能正卡在 `answers.get()` 上的那个等待。
        """
        self.stopped.set()
        self.stop.set()
        self.answers.put(_STOP)


# ---------------------------------------------------------------------------
# 3. 离线演示：把线程桥和 LLM 解耦
# ---------------------------------------------------------------------------

async def _demo_run(hooks: RunHooks) -> None:
    """假数据跑一遍完整回路。

    存在的意义是**把「桥」和「网络」拆开**：桥坏没坏，跑这个两秒就知道，
    不用等两分钟的联网搜索。它走的是和真实 run 完全相同的 worker/队列/SSE 管子。
    """
    from types import SimpleNamespace

    from autogen_agentchat.messages import TextMessage

    async def pause(seconds: float) -> None:
        await asyncio.sleep(seconds)

    await pause(0.3)
    hooks.on_message(TextMessage(
        source="researcher",
        content="素材清单（演示数据，未联网）：\n"
                "1. 五泄风景区门票 80 元/人，游览约 4 小时。\n"
                "2. 西施故里门票 100 元/人，夜游浣江免费。\n"
                "3. 市区快捷酒店普遍 380–420 元/晚。",
    ))
    await pause(0.4)
    hooks.on_message(SimpleNamespace(
        source="researcher",
        content=[SimpleNamespace(name="search", arguments='{"query": "诸暨 门票 价格"}')],
    ))
    await pause(0.4)
    hooks.on_message(SimpleNamespace(
        source="researcher",
        content=[SimpleNamespace(name="search", content="（演示数据）返回 8 条结果…")],
    ))
    await pause(0.5)
    hooks.on_message(TextMessage(
        source="planner",
        content="逐日行程（演示数据）：\n"
                "Day1 西施故里 → 浣江夜游\n"
                "Day2 五泄风景区（全天）\n"
                "Day3 斗岩 → 返程",
    ))
    await pause(0.5)
    hooks.on_message(TextMessage(
        source="critic",
        content="预算核算：住宿 400×2 晚 = 800，门票 (80+100)×2 = 360，"
                "合计已 1160，加交通后超 2000 预算。\n"
                "问题 1：住宿标准要不要降到 250/晚？\n"
                "问题 2：五泄要不要改成半天？",
    ))
    await pause(0.4)
    hooks.on_notice("该你拍板了（第 1 次）", "banner")
    hooks.on_notice(
        "预算核算：住宿 400×2 晚 = 800，门票 (80+100)×2 = 360，合计已 1160，"
        "加交通后超 2000 预算。\n问题 1：住宿标准要不要降到 250/晚？\n"
        "问题 2：五泄要不要改成半天？",
        "question",
    )

    answer = hooks.ask(ASK_HINT)          # ← 这里会真的阻塞，等你在页面上回话
    if answer is None:
        hooks.on_notice("结束原因：External termination requested", "final")
        return

    await pause(0.4)
    hooks.on_message(TextMessage(source="user", content=answer))
    await pause(0.6)
    hooks.on_message(TextMessage(
        source="planner",
        content=f"已按「{answer}」重排（演示数据）：总预算 1980，落在 2000 以内。",
    ))
    await pause(0.5)
    hooks.on_message(TextMessage(source="critic", content="APPROVED"))
    hooks.on_notice("结束原因：'APPROVED' from critic (exact match)", "final")


# ---------------------------------------------------------------------------
# 4. HTTP
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "TravelAgentLauncher"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args) -> None:      # 别把每条请求刷到终端
        pass

    # ---- 工具 ----

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {}

    def _json(self, payload: dict) -> None:
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    # ---- 路由 ----

    def do_GET(self) -> None:                       # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            boot = json.dumps(
                {"demo": self.server.demo, "defaultTask": DEFAULT_TASK}, ensure_ascii=False
            )
            page = _PAGE.replace("__BOOT__", boot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif path == "/health":
            # 只报身份，不碰 session——所以跑着一轮的时候它照样秒回，
            # 这正是「重复双击能不能认出自己人」需要的性质。
            self._json({"app": _APP_ID, "demo": self.server.demo})
        elif path == "/events":
            self._stream()
        elif path == "/favicon.ico":
            self.send_response(204)                 # 浏览器总会来要，别让它 404 刷控制台
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self) -> None:                      # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/start":
            task = str(self._body().get("task") or DEFAULT_TASK).strip() or DEFAULT_TASK
            self.server.bus.reset()
            session = _Session(self.server.bus, task, demo=self.server.demo)
            self.server.session = session
            session.start()
            self._json({"ok": True, "task": task})
        elif path == "/answer":
            session = self.server.session
            if session is None:
                self._json({"ok": False, "why": "还没开始跑"})
                return
            session.reply(str(self._body().get("text") or ""))
            self._json({"ok": True})
        elif path == "/stop":
            session = self.server.session
            if session is None:
                self._json({"ok": False, "why": "还没开始跑"})
                return
            session.request_stop()
            self._json({"ok": True})
        else:
            self.send_error(404)

    def _stream(self) -> None:
        """SSE。

        连接**不主动关闭**：EventSource 在连接断掉时会自动重连，如果我们在
        `done` 之后就断开，浏览器会重连、拿到历史、又立刻断开——变成无限重连循环。
        所以让连接一直挂着，靠前端的 seq 去重来处理重放。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = self.server.bus.subscribe()
        try:
            while True:
                event = q.get()
                chunk = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                                     # 关页/刷新，正常现象
        finally:
            self.server.bus.unsubscribe(q)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, demo: bool = False) -> None:
        super().__init__(addr, handler)
        self.bus = _Bus()
        self.session: _Session | None = None
        self.demo = demo


# ---------------------------------------------------------------------------
# 5. 页面（内嵌，不搞静态文件目录）
# ---------------------------------------------------------------------------

_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>旅行规划</title>
<style>
  :root {
    --bg: #f6f7f9;      --panel: #ffffff;   --fg: #1f2328;
    --muted: #59636e;   --border: #d8dee4;  --accent: #1f6feb;
    --accent-hi: #1a5fd0; --warn: #9a6700;
    --researcher: #0969da; --planner: #1a7f37; --critic: #9a6700; --user: #8250df;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.65 -apple-system, "Segoe UI", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  }
  .wrap { max-width: 940px; margin: 0 auto; padding: 20px 16px 32px; }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 16px; }
  h1 { font-size: 17px; font-weight: 600; margin: 0; letter-spacing: .01em; white-space: nowrap; }
  /* 状态栏要能包住长报错，不然一条 300 字的警告会把标题挤没 */
  #status { margin-left: auto; font-size: 13px; color: var(--muted);
            text-align: right; max-width: 70%; white-space: pre-wrap; }
  #status.warn { color: var(--warn); }
  #status.live { color: var(--accent); }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  }
  .row { display: flex; gap: 8px; padding: 10px; }
  input[type=text] {
    flex: 1; min-width: 0; font: inherit; color: var(--fg); background: var(--panel);
    border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; outline: none;
  }
  input[type=text]:focus { border-color: var(--accent); }
  input[type=text]:disabled { background: var(--bg); color: var(--muted); }
  button {
    font: inherit; border: 1px solid var(--border); border-radius: 6px; padding: 8px 16px;
    background: var(--panel); color: var(--fg); cursor: pointer; white-space: nowrap;
  }
  button:hover:not(:disabled) { background: var(--bg); }
  button:disabled { opacity: .45; cursor: default; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.primary:hover:not(:disabled) { background: var(--accent-hi); }
  #chat {
    margin-top: 12px; height: calc(100vh - 320px); min-height: 240px;
    overflow-y: auto; padding: 6px 14px 14px;
  }
  .msg { padding: 10px 0 2px; border-top: 1px solid var(--border); white-space: pre-wrap; }
  .msg:first-child { border-top: 0; }
  .msg .src { display: block; font-size: 12px; letter-spacing: .04em; margin-bottom: 2px; }
  .msg.tool { color: var(--muted); font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
              font-size: 12.5px; padding-left: 16px; white-space: pre; overflow-x: auto; }
  .msg.tool .src { display: none; }
  .empty { color: var(--muted); padding: 22px 0; text-align: center; }
  #askbox { margin-top: 12px; }
  #askbox .row { padding-top: 0; }
  #asklabel { padding: 10px 10px 0; font-size: 13px; color: var(--muted); }
  #askq { padding: 0 10px; white-space: pre-wrap; display: none; }
  #askq.show { display: block; padding-bottom: 8px; }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
         background: var(--muted); margin-right: 6px; vertical-align: 1px; }
  .dot.live { background: var(--accent); }
  .dot.warn { background: var(--warn); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>旅行规划</h1>
    <span id="status"><span class="dot" id="dot"></span><span id="statustext">就绪</span></span>
  </header>

  <div class="card">
    <div class="row">
      <input type="text" id="task" placeholder="一句话需求，例如：我国庆去诸暨玩，三天，两个人，预算 2000。">
      <button class="primary" id="start">开始</button>
      <button id="stop" disabled>停止</button>
    </div>
  </div>

  <div class="card" id="chat">
    <div class="empty" id="empty">输入需求后点「开始」。</div>
  </div>

  <div class="card" id="askbox">
    <div id="asklabel">拍板</div>
    <div id="askq"></div>
    <div class="row">
      <input type="text" id="answer" placeholder="要等 critic 问你的时候才能回话" disabled>
      <button class="primary" id="reply" disabled>回复</button>
    </div>
  </div>
</div>

<script>
const BOOT = __BOOT__;
const AGENT_COLORS = {
  researcher: "var(--researcher)", planner: "var(--planner)",
  critic: "var(--critic)", user: "var(--user)"
};
const $ = (id) => document.getElementById(id);
let lastSeq = -1;      // 去重：重连/重放时同一事件不会渲染两遍
let asking = false;

$("task").value = BOOT.defaultTask;
if (BOOT.demo) { $("task").value = "（演示模式，需求内容会被忽略）"; }

function setStatus(text, tone) {
  $("statustext").textContent = text;
  $("status").className = tone || "";
  $("dot").className = "dot" + (tone ? " " + tone : "");
}

function setRunning(on) {
  $("start").disabled = on;
  $("stop").disabled = !on;
  $("task").disabled = on;
}

function setAsking(on) {
  asking = on;
  $("answer").disabled = !on;
  $("reply").disabled = !on;
  $("askq").className = on ? "show" : "";
  $("answer").placeholder = on
    ? "带一个具体数字或动作，例如：住宿砍到 250 一晚"
    : "要等 critic 问你的时候才能回话";
  if (on) { $("answer").focus(); } else { $("answer").value = ""; }
}

function append(text, cls, source) {
  const empty = $("empty");
  if (empty) { empty.remove(); }
  const el = document.createElement("div");
  el.className = "msg" + (cls ? " " + cls : "");
  if (source) {
    const s = document.createElement("span");
    s.className = "src";
    s.textContent = source;
    s.style.color = AGENT_COLORS[source] || "var(--muted)";
    el.appendChild(s);
  }
  el.appendChild(document.createTextNode(text));
  const chat = $("chat");
  const atBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 40;
  chat.appendChild(el);
  if (atBottom) { chat.scrollTop = chat.scrollHeight; }
}

function handle(ev) {
  if (ev.seq <= lastSeq) { return; }
  lastSeq = ev.seq;

  if (ev.type === "message") {
    if (ev.kind === "tool") { append(ev.text, "tool"); }
    else { append(ev.text, "", ev.source); }
  } else if (ev.type === "ask") {
    $("askq").textContent = ev.question || "（critic 没写出问题清单，只调了 handoff 工具）";
    $("asklabel").textContent = ev.hint || "拍板";
    setAsking(true);
    setStatus("等你拍板", "warn");
  } else if (ev.type === "notice") {
    if (ev.kind === "banner") { setStatus(ev.text, "warn"); }
    else if (ev.kind === "warning") { setStatus(ev.text, "warn"); }
    else if (ev.kind === "final") { setStatus(ev.text); }
    else { setStatus(ev.text); }
  } else if (ev.type === "done") {
    setAsking(false);
    setRunning(false);
  }
}

function connect() {
  const es = new EventSource("/events");
  es.onmessage = (e) => handle(JSON.parse(e.data));
}

async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  return r.json();
}

$("start").onclick = async () => {
  $("chat").innerHTML = "";
  lastSeq = -1;
  setAsking(false);
  setRunning(true);
  setStatus("运行中（researcher 联网查资料时可能两分钟没有新消息）", "live");
  await post("/start", { task: $("task").value });
};

$("stop").onclick = async () => {
  setStatus("停止中…（会在当前 agent 这一轮结束时收手）", "warn");
  await post("/stop");
};

$("reply").onclick = async () => {
  const text = $("answer").value;
  setAsking(false);
  setStatus("运行中", "live");
  // **不在这里乐观追加**：回答会作为 HandoffMessage 的 task 被 run_stream 回显成一条
  // user 消息，本地再插一次就成了两条一模一样的。
  await post("/answer", { text: text });
};

$("answer").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && asking) { $("reply").click(); }
});
$("task").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !$("start").disabled) { $("start").click(); }
});

connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 6. 入口
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AutoGen 多智能体旅行规划的网页启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  python launcher.py            # 起服务，然后开它打印的 URL\n"
            "  python launcher.py --demo     # 离线假数据：不要 key、不联网\n"
            "\n"
            "跑起来后在页面里写需求、点「开始」。critic 要你拍板时输入框会亮，\n"
            "回答带一个具体数字或动作——「住宿砍到 250 一晚」比「再优化一下」有用。\n"
        ),
    )
    parser.add_argument("--demo", action="store_true", help="离线假数据，不需要 key、不联网")
    parser.add_argument("--port", type=int, default=8765, help="端口，默认 8765（被占用就往后顺延）")
    parser.add_argument("--no-browser", action="store_true", help="不要试着自动开浏览器")
    args = parser.parse_args(argv)

    # 已经有一个在跑？把浏览器指过去就完事，**不要再起第二个**。
    # 双击图标两次、或者服务在后台开着又点了一次，都会走到这里；不起这一步的话
    # 第二个实例会顺延到下一个端口，于是你得到两个各自为政的页面。
    for port in range(args.port, args.port + 11):
        if _existing_instance(port):
            url = f"http://localhost:{port}/"
            print(f"启动器已经在跑了，直接把浏览器指过去：\n\n    {url}\n", flush=True)
            if not args.no_browser:
                _open_browser(url)
            return 0

    server = None
    for port in range(args.port, args.port + 11):
        try:
            server = _Server(("127.0.0.1", port), _Handler, demo=args.demo)
            break
        except OSError:
            continue
    if server is None:
        print(f"\n{args.port}–{args.port + 10} 都被占用了，用 --port 换一个。", file=sys.stderr)
        return 2

    port = server.server_address[1]
    url = f"http://localhost:{port}/"
    # flush 是必须的：stdout 不是 TTY 时（`| tee`、后台跑）Python 会块缓冲，
    # 不 flush 的话这段话会一直卡在缓冲区里，用户看不到 URL 就只能干等。
    print(f"\n{'=' * 72}\n旅行规划启动器已就绪：\n\n    {url}\n", flush=True)
    print("正在打开浏览器……没反应就手动复制上面这个地址。" if not args.no_browser
          else "（--no-browser：不自动开浏览器，手动复制上面这个地址。）", flush=True)
    if args.demo:
        print("（--demo：离线假数据，不需要 key、不联网。）", flush=True)
    print(f"{'=' * 72}\n关掉这个窗口、或者按 Ctrl-C，都会停掉服务。", flush=True)

    if not args.no_browser:
        # 0.4 秒是给 serve_forever 让路。socket 在 _Server(...) 里就已经 bind+listen 了，
        # 所以浏览器这会儿连上来只是进 backlog 等一会儿，不会被拒。
        threading.Timer(0.4, _open_browser, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
