# ============================================
# pressget.py —— 实时压力变化检测（静止段 + 峰值检测）
# 功能：
#   ✅ 实时监听 pressure_log.csv 文件变化
#   ✅ 计算最近窗口的标准差 → 判断静止段 / 动态段
#   ✅ 基于导数反转 + prominence 进行实时峰值检测
#   ✅ 输出 🟢 静止段 / 🔴 动态段 / 🟣 峰值检测事件
# ============================================

import pandas as pd
import numpy as np
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------- 配置 ----------
FILE_PATH = "/home/ljy/project/hand_dec/datacap/1/pressure_log.csv"
DIR_PATH  = os.path.dirname(FILE_PATH)

WINDOW_STD = 30             # 静止判断窗口（最近30点）
STATIC_STD_THRESH = 0.02    # 静止段标准差阈值

WINDOW_PEAK = 80            # 峰值检测窗口（约1~1.5秒）
PROM_THRESH = 0.03          # 峰谷差阈值（压差至少 3%）
MIN_INTERVAL = 0.3          # 两个峰值最小时间间隔(s)

# ---------- 状态变量 ----------
last_peak_time = 0
last_peak_val  = 0.0
last_valley    = np.inf
armed = True                # “武装状态”防止重复触发

# ---------- 文件事件回调 ----------
class PressureWatcher(FileSystemEventHandler):
    def on_modified(self, event):
        global last_peak_time, last_peak_val, last_valley, armed

        if not event.src_path.endswith("pressure_log.csv"):
            return

        try:
            df = pd.read_csv(FILE_PATH)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return

        if "press_sum" not in df.columns:
            return

        # 转为浮点 + 去除 NaN
        df["press_sum"] = pd.to_numeric(df["press_sum"], errors="coerce")
        df = df.dropna(subset=["press_sum"])
        if len(df) == 0:
            return

        # ---------- 静止段判断 ----------
        window_std = np.array(df["press_sum"].tail(WINDOW_STD))
        std_val = np.std(window_std)
        if std_val < STATIC_STD_THRESH:
            print(f"🟢 静止段 (STD={std_val:.4f})")
        else:
            print(f"🔴 动态段 (STD={std_val:.4f})")

        # ---------- 峰值检测 ----------
        vals = np.array(df["press_sum"].tail(WINDOW_PEAK))
        if len(vals) < 5:
            return

        d1 = np.diff(vals)
        # 找到导数正变负的转折点
        turning_idx = np.where((d1[:-1] > 0) & (d1[1:] <= 0))[0] + 1
        if len(turning_idx) == 0:
            return

        # 取最后一个候选峰
        peak_i = turning_idx[-1]
        peak_val = vals[peak_i]

        # 估算 valley（取峰值前的最小值）
        valley_val = np.min(vals[:peak_i]) if peak_i > 0 else peak_val
        delta = peak_val - valley_val

        now = time.time()
        dt = now - last_peak_time

        # 满足峰值条件
        if delta > PROM_THRESH and dt > MIN_INTERVAL and armed:
            print(f"🟣 峰值检测: {peak_val:.3f} (Δ={delta:.3f}, dt={dt:.2f}s)")
            last_peak_time = now
            last_peak_val = peak_val
            armed = False
            last_valley = np.inf

        # 自动重新武装（下降回基线）
        current_val = vals[-1]
        if not armed:
            if current_val < peak_val - PROM_THRESH/2 or dt > 1.0:
                armed = True

# ---------- 主程序 ----------
if __name__ == "__main__":
    print(f"[INFO] 实时监控启动中: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print(f"⚠️ 未找到文件: {FILE_PATH}")
        exit(1)

    event_handler = PressureWatcher()
    observer = Observer()
    observer.schedule(event_handler, DIR_PATH, recursive=False)
    observer.start()

    print("[INFO] 文件监听已启动，等待压力数据写入...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[INFO] 手动退出监听。")

    observer.join()
