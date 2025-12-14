# ============================================
# pressget_dual_v7_detector_merge10ms.py
# —— 实时压力检测（融合版：10ms峰值合并 + 实时监听 + 全局计数）
# ============================================

import pandas as pd
import numpy as np
import os, time, json
from scipy.signal import find_peaks
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ---------- 参数配置 ----------
FILE_PATH = "/home/ljy/project/hand_dec/ljy/ljy_1/pressure_log.csv"
DIR_PATH  = os.path.dirname(FILE_PATH)
SAVE_FILE = "/home/ljy/project/hand_dec/datacap/1/pressure_peaks_valid.csv"
COUNT_FILE = "/home/ljy/project/hand_dec/datacap/1/press_count.json"

STATIC_STD_THRESH = 1000.0        # 静止段判断阈值
VALID_PRESS_THRESH = 2000         # ✅ 有效按压正峰阈值
MIN_DISTANCE = 1                  # 峰值间最小间距（样本点数）
PROMINENCE = 20                   # 峰值显著性要求
MERGE_WINDOW_MS = 10              # ✅ 10ms 内只保留最大峰值
DELAY_AFTER_WRITE = 0.1           # C++ 写入延迟补偿

# ---------- 全局变量 ----------
last_peak_time = 0
total_valid_count = 0
session_count = 0


# ---------- 辅助函数 ----------
def load_press_count():
    """从 JSON 文件恢复累计次数"""
    global total_valid_count
    if os.path.exists(COUNT_FILE):
        try:
            with open(COUNT_FILE, "r") as f:
                data = json.load(f)
                total_valid_count = int(data.get("total_valid_count", 0))
            print(f"[INFO] 已恢复累计按压次数: {total_valid_count}")
        except Exception:
            print("[WARN] 读取累计计数文件失败，重新计数。")
            total_valid_count = 0
    else:
        total_valid_count = 0


def save_press_count():
    """保存累计次数到 JSON 文件"""
    with open(COUNT_FILE, "w") as f:
        json.dump({"total_valid_count": total_valid_count}, f)


# ---------- 核心检测函数 ----------
def detect_global_valid_peaks(file_path: str):
    """融合版核心逻辑：10ms内峰值合并 + 新峰检测 + 文件保存"""
    global last_peak_time, total_valid_count, session_count

    if not os.path.exists(file_path):
        return

    try:
        df = pd.read_csv(file_path, usecols=["time_ms", "press_sum_norm"])
    except Exception as e:
        print(f"[ERROR] 文件读取失败: {e}")
        return

    if df.empty or len(df) < 10:
        return

    df["press_sum_norm"] = pd.to_numeric(df["press_sum_norm"], errors="coerce")
    df = df.dropna(subset=["press_sum_norm"])

    time_ms = df["time_ms"].to_numpy(dtype=float)
    press   = df["press_sum_norm"].to_numpy(dtype=float)

    # ---- 静止段判断 ----
    std_val = np.std(press[-50:])
    if std_val < STATIC_STD_THRESH:
        print(f"🟢 静止段 (STD={std_val:.1f})")
        return

    # ---- 初步峰值检测 ----
    pos_locs, _ = find_peaks(press, prominence=PROMINENCE, distance=MIN_DISTANCE)
    if len(pos_locs) == 0:
        return

    peaks_time = time_ms[pos_locs]
    peaks_val  = press[pos_locs]

    # ---- 合并10ms内的近邻峰，只保留最大值 ----
    merged_times, merged_vals = [], []
    if len(peaks_time) > 0:
        group_start = 0
        for i in range(1, len(peaks_time)):
            if peaks_time[i] - peaks_time[i - 1] <= MERGE_WINDOW_MS:
                continue
            else:
                group_slice = slice(group_start, i)
                max_idx = np.argmax(peaks_val[group_slice]) + group_start
                merged_times.append(peaks_time[max_idx])
                merged_vals.append(peaks_val[max_idx])
                group_start = i
        group_slice = slice(group_start, len(peaks_time))
        max_idx = np.argmax(peaks_val[group_slice]) + group_start
        merged_times.append(peaks_time[max_idx])
        merged_vals.append(peaks_val[max_idx])

    peaks_time = np.array(merged_times)
    peaks_val  = np.array(merged_vals)

    # ---- 筛选有效按压 ----
    valid_mask = peaks_val > VALID_PRESS_THRESH
    peaks_time = peaks_time[valid_mask]
    peaks_val  = peaks_val[valid_mask]
    if len(peaks_time) == 0:
        return

    # ---- 新峰值过滤 ----
    new_idx = peaks_time > last_peak_time
    if not np.any(new_idx):
        return

    new_times = peaks_time[new_idx]
    new_vals  = peaks_val[new_idx]

    # ---- 写入文件 + 更新编号 ----
    press_ids_global = [total_valid_count + i + 1 for i in range(len(new_times))]
    df_valid = pd.DataFrame({
        "press_id_global": press_ids_global,
        "t_pos_ms": new_times,
        "press_pos": new_vals
    })

    df_valid.to_csv(SAVE_FILE, mode='a', header=not os.path.exists(SAVE_FILE), index=False)

    total_valid_count += len(df_valid)
    session_start_id = session_count + 1
    session_count += len(df_valid)
    last_peak_time = new_times[-1]
    save_press_count()

    for i, row in enumerate(df_valid.itertuples(), start=session_start_id):
        print(f"✅ 第 {i} 次有效按压: {row.press_pos:.0f} @ {row.t_pos_ms:.0f} ms")


# ---------- 文件监听 ----------
class PressureWatcher(FileSystemEventHandler):
    """监听 pressure_log.csv 文件变化并触发检测"""
    def on_modified(self, event):
        if not event.src_path.endswith("pressure_log.csv"):
            return
        time.sleep(DELAY_AFTER_WRITE)
        detect_global_valid_peaks(FILE_PATH)


# ---------- 主程序 ----------
if __name__ == "__main__":
    print(f"[INFO] 正在监听文件变化: {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        print(f"⚠️ 文件不存在: {FILE_PATH}")
        exit(1)

    load_press_count()

    event_handler = PressureWatcher()
    observer = Observer()
    observer.schedule(event_handler, DIR_PATH, recursive=False)
    observer.start()
    print("[INFO] 文件监听已启动，等待 C++ 写入中...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print(f"\n[INFO] 手动退出监听。当前会话检测到 {session_count} 次有效按压，总累计 {total_valid_count} 次。")
        save_press_count()

    observer.join()
