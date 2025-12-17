#!/usr/bin/env python3
import subprocess
import signal
import sys
import time
import os

# ================= 基本配置 =================

PYTHON = "/home/ljy/anaconda3/envs/mp_hands/bin/python"

APPDESIGN = (
    "/home/ljy/project/qt_press_image/pressprogress1/"
    "appdesign_4/build/AppDesign"
)

PROCS = []

# ================= Qt / X11 环境 =================

def _build_qt_env():
    """统一 Qt / X11 运行环境"""
    env = os.environ.copy()

    env["DISPLAY"] = os.environ.get("DISPLAY", ":1")
    env["XAUTHORITY"] = os.environ.get(
        "XAUTHORITY",
        os.path.expanduser("~/.Xauthority")
    )
    env["QT_QPA_PLATFORM"] = "xcb"
    env["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
    env["PYTHONUNBUFFERED"] = "1"

    return env


# ================= 启动函数 =================

def start(cmd, name):
    """启动 Python 进程"""
    print(f"[START] {name}")
    env = _build_qt_env()

    p = subprocess.Popen(cmd, env=env)
    PROCS.append((name, p))
    print(f"[PID]   {name} -> {p.pid}")
    return p


def start_binary(path, name):
    """启动 C++ Qt 二进制"""
    if not os.path.exists(path):
        print(f"[FATAL] binary not found: {path}")
        sys.exit(1)

    print(f"[START] {name}")
    env = _build_qt_env()

    p = subprocess.Popen([path], env=env)
    PROCS.append((name, p))
    print(f"[PID]   {name} -> {p.pid}")
    return p


# ================= 窗口布局（关键新增） =================

def layout_windows():
    """
    稳定布局：
    左：视觉
    右上：融合
    右下：压力
    """
    print("[INFO] arranging windows...")

    # 等待所有窗口真正出现（比 sleep 稳）
    for _ in range(20):
        ret = os.system("wmctrl -l | grep -q 'Hand Depth Monitor'")
        if ret == 0:
            break
        time.sleep(0.1)

    # time.sleep(0.00001)

    # ===== 1️⃣ 视觉窗口（左，大）=====
    os.system(
        "wmctrl -r 'Hand Depth Monitor' "
        "-e 0,0,0,1355,960"
    )

    time.sleep(0.00001)

    # ===== 2️⃣ 融合监控（右上）=====
    os.system(
        "wmctrl -r 'CPR Fusion Monitor' "
        "-e 0,1355,0,721,160"
    )

    # time.sleep(0.00001)

    # ===== 3️⃣ 压力界面（右下）=====
    os.system(
        "wmctrl -r 'Pressure Feedback Interface' "
        "-e 0,1355,255,721,827"
    )

    print("[INFO] window layout done")



# ================= 停止逻辑 =================

def stop_all(sig=None, frame=None):
    print("\n[INFO] Shutting down all processes...")

    for name, p in PROCS:
        if p.poll() is None:
            print(f"[STOP] {name}")
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass

    time.sleep(1.0)

    for name, p in PROCS:
        if p.poll() is None:
            print(f"[KILL] {name}")
            try:
                p.kill()
            except Exception:
                pass

    print("[INFO] Exit")
    sys.exit(0)


signal.signal(signal.SIGINT, stop_all)
signal.signal(signal.SIGTERM, stop_all)

# ================= 主入口 =================

if __name__ == "__main__":

    # ===== 清理旧 socket（非常重要）=====
    for p in (
        "/tmp/ui_event.sock",
        "/tmp/ui_pressure.sock",
    ):
        try:
            os.unlink(p)
            print(f"[CLEAN] removed {p}")
        except FileNotFoundError:
            pass

    print("[INFO] DISPLAY     =", os.environ.get("DISPLAY"))
    print("[INFO] XAUTHORITY  =", os.environ.get("XAUTHORITY"))
    print("[INFO] Python      =", PYTHON)

    # =====================================================
    # 启动顺序（不要随意改）
    # =====================================================

    # 0️⃣ C++ Qt UI（压力图 + CNN）
    start_binary(APPDESIGN, "pressure_ui")
    time.sleep(0.5)

    # 1️⃣ Python Qt UI（融合展示）
    start([PYTHON, "qt_fusion_monitor.py"], "qt_ui")
    time.sleep(0.5)

    # 2️⃣ Fusion / Listener
    start([PYTHON, "listener_fusion_4.py"], "fusion")
    time.sleep(0.5)

    # 3️⃣ Vision
    start([PYTHON, "hand_dec_3.py"], "vision")

    # ===== 自动布局窗口 =====
    layout_windows()

    print("\n[INFO] System running. Press Ctrl+C to exit.\n")

    # 主线程阻塞
    while True:
        time.sleep(1)
