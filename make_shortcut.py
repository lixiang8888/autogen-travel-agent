# -*- coding: utf-8 -*-
"""
make_shortcut.py —— 在 Windows 桌面创建一个「点击就启动」的快捷方式
=====================================================================

只有 WSL 上有意义。它建的是一个 Windows 的 `.lnk`，内容是：

    wsl.exe -d <发行版> --cd <本项目目录> -- .venv/bin/python launcher.py

双击的效果 = 起服务 + 自动开浏览器，不用敲命令、不用开终端找路径。

跑法：
    python make_shortcut.py

关于编码：这个脚本里唯一带中文的是快捷方式的**名字**和**描述**。它们通过
WSL 互操作作为 argv 传给 PowerShell（UTF-8 → UTF-16），再由 `WScript.Shell`
写进 `.lnk`（内部就是 UTF-16），所以全程没有「按哪个 codepage 解释这些字节」
的问题——这也是这里不用 `.bat` 的原因：`.bat` 是要被 cmd.exe 按 OEM codepage
读的，中文路径放进去能不能活全看运气。
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

#: 快捷方式在桌面上的名字。
SHORTCUT_NAME = "旅行规划.lnk"

#: 快捷方式的描述（鼠标悬停时显示）。
DESCRIPTION = "启动 AutoGen 旅行规划网页界面"


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in pathlib.Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def powershell() -> str | None:
    for name in ("powershell.exe", "pwsh.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build_script(project_dir: str, distro: str) -> str:
    """拼出那段创建快捷方式的 PowerShell。

    路径用**单引号**括起来（PowerShell 的单引号串不做变量插值），免得路径里万一
    出现 `$` 被当成变量。这里没有 shell 参与——argv 是直接传给 powershell.exe 的，
    所以不用操心 bash 的引号。
    """
    arguments = f"-d {distro} --cd {project_dir} -- .venv/bin/python launcher.py"
    return ";".join([
        # PowerShell 5.1 默认按控制台的 OEM 代码页（中文机器上是 GBK）写 stdout，
        # 而这边按 UTF-8 读，中文就会花掉。让它先改成 UTF-8。
        '[Console]::OutputEncoding = [Text.Encoding]::UTF8',
        '$d = [Environment]::GetFolderPath("Desktop")',
        f'$lnk = Join-Path $d "{SHORTCUT_NAME}"',
        '$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)',
        '$s.TargetPath = "$env:SystemRoot\\System32\\wsl.exe"',
        f"$s.Arguments = '{arguments}'",
        '$s.WorkingDirectory = "$env:SystemRoot\\System32"',
        f'$s.Description = "{DESCRIPTION}"',
        '$s.Save()',
        'Write-Output $lnk',
    ])


def main() -> int:
    if not is_wsl():
        print("这个脚本只在 WSL 里有意义——它建的是 Windows 的 .lnk。", file=sys.stderr)
        print("非 WSL 环境下，直接用 `python launcher.py` 就是完整用法。", file=sys.stderr)
        return 1

    ps = powershell()
    if ps is None:
        print("找不到 powershell.exe，没法创建快捷方式。", file=sys.stderr)
        return 1

    distro = os.environ.get("WSL_DISTRO_NAME") or "Ubuntu"
    project_dir = str(pathlib.Path(__file__).resolve().parent)
    script = build_script(project_dir, distro)

    try:
        done = subprocess.run(
            [ps, "-NoProfile", "-Command", script],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=90, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"调用 PowerShell 失败：{exc}", file=sys.stderr)
        return 1

    if done.returncode != 0:
        print(f"创建失败（PowerShell 退出码 {done.returncode}）：", file=sys.stderr)
        print((done.stderr or done.stdout or "").strip(), file=sys.stderr)
        return 1

    lnk = (done.stdout or "").strip().splitlines()[-1] if done.stdout.strip() else SHORTCUT_NAME
    print(f"\n{'=' * 72}\n桌面快捷方式已创建：\n\n    {lnk}\n")
    print("双击它就会起服务并自动打开浏览器。重复双击不会起第二个服务——")
    print("启动器会认出已经在跑的那个，只把浏览器指过去。")
    print(f"{'=' * 72}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
