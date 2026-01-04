# # ============================================
# # pressure_detector.py
# # —— 模块化实时压力检测，可直接函数调用
# # ============================================

# import pandas as pd
# import numpy as np
# import os, time, json
# from scipy.signal import find_peaks

# # ---------- 参数配置 ----------
# FILE_PATH = "/home/ljy/project/hand_dec/ljy/ljy_1/pressure_log.csv"
# SAVE_FILE = "/home/ljy/project/hand_dec/datacap/1/pressure_peaks_valid.csv"
# COUNT_FILE = "/home/ljy/project/hand_dec/datacap/1/press_count.json"

# STATIC_STD_THRESH = 1000.0        # 静止段判断阈值
# VALID_PRESS_THRESH = 2000         # ✅ 有效按压正峰阈值
# MIN_DISTANCE = 1                  # 峰值间最小间距（样本点数）
# PROMINENCE = 20                   # 峰值显著性要求

# # ---------- 全局状态 ----------
# last_peak_time = 0
# total_valid_count = 0
# session_count = 0


# # ---------- 初始化函数 ----------
# def init_pressure_detector():
#     """初始化并加载计数状态"""
#     global total_valid_count
#     if os.path.exists(COUNT_FILE):
#         try:
#             with open(COUNT_FILE, "r") as f:
#                 data = json.load(f)
#                 total_valid_count = int(data.get("total_valid_count", 0))
#             print(f"[INFO] 已恢复累计按压次数: {total_valid_count}")
#         except Exception:
#             print("[WARN] 计数文件损坏，重置。")
#             total_valid_count = 0
#     else:
#         total_valid_count = 0


# def save_press_count():
#     with open(COUNT_FILE, "w") as f:
#         json.dump({"total_valid_count": total_valid_count}, f)


# # ---------- 核心检测函数 ----------
# def detect_pressure_peaks(file_path=FILE_PATH):
#     """
#     检测最新有效按压，返回列表:
#     [ (local_idx, global_idx, press_value, time_ms), ... ]
#     """
#     global last_peak_time, total_valid_count, session_count

#     if not os.path.exists(file_path):
#         return []

#     try:
#         df = pd.read_csv(file_path, usecols=["time_ms", "press_sum_norm"])
#     except Exception:
#         return []

#     if df.empty or len(df) < 10:
#         return []

#     df["press_sum_norm"] = pd.to_numeric(df["press_sum_norm"], errors="coerce")
#     df = df.dropna(subset=["press_sum_norm"])

#     time_ms = df["time_ms"].to_numpy(dtype=float)
#     press = df["press_sum_norm"].to_numpy(dtype=float)

#     # ---- 静止段判断 ----
#     std_val = np.std(press[-50:])
#     if std_val < STATIC_STD_THRESH:
#         return []

#     # ---- 峰值检测 ----
#     pos_locs, _ = find_peaks(press, prominence=PROMINENCE, distance=MIN_DISTANCE)
#     if len(pos_locs) == 0:
#         return []

#     peaks_time = time_ms[pos_locs]
#     peaks_val = press[pos_locs]

#     # ---- 有效按压筛选 ----
#     valid_mask = peaks_val > VALID_PRESS_THRESH
#     peaks_time = peaks_time[valid_mask]
#     peaks_val = peaks_val[valid_mask]

#     if len(peaks_time) == 0:
#         return []

#     # ---- 新峰值过滤 ----
#     new_idx = peaks_time > last_peak_time
#     if not np.any(new_idx):
#         return []

#     new_times = peaks_time[new_idx]
#     new_vals = peaks_val[new_idx]

#     # ---- 构造返回结果 ----
#     press_ids_global = [total_valid_count + i + 1 for i in range(len(new_times))]
#     session_start_id = session_count + 1
#     session_count += len(new_times)
#     total_valid_count += len(new_times)
#     last_peak_time = new_times[-1]
#     save_press_count()

#     results = []
#     for i, (gid, val, t_ms) in enumerate(zip(press_ids_global, new_vals, new_times), start=session_start_id):
#         results.append((i, gid, val, t_ms))
#         print(f"✅ 第 {i} 次有效按压: {val:.0f} @ {t_ms:.0f} ms")

#     return results


# def get_total_count():
#     """返回当前累计按压次数"""
#     return total_valid_count
# ============================================
# pressure_detector.py
# —— 模块化实时压力检测，可直接函数调用（含10ms峰值合并）
# ============================================

import pandas as pd
import numpy as np
import os, time, json
from scipy.signal import find_peaks

# ---------- 参数配置 ----------
FILE_PATH = "/home/ljy/project/hand_dec/ljy/ljy_1/pressure_log.csv"
SAVE_FILE = "/home/ljy/project/hand_dec/datacap/1/pressure_peaks_valid.csv"
COUNT_FILE = "/home/ljy/project/hand_dec/datacap/1/press_count.json"

STATIC_STD_THRESH = 1000.0        # 静止段判断阈值
VALID_PRESS_THRESH = 5000         # ✅ 有效按压正峰阈值
MIN_DISTANCE = 1                  # 峰值间最小间距（样本点数）
PROMINENCE = 20                   # 峰值显著性要求
MERGE_WINDOW_MS = 10              # ✅ 10ms 内只保留最大峰值

# ---------- 全局状态 ----------
last_peak_time = 0
total_valid_count = 0
session_count = 0


# ---------- 初始化函数 ----------
def init_pressure_detector():
    """初始化并加载计数状态"""
    global total_valid_count
    if os.path.exists(COUNT_FILE):
        try:
            with open(COUNT_FILE, "r") as f:
                data = json.load(f)
                total_valid_count = int(data.get("total_valid_count", 0))
            print(f"[INFO] 已恢复累计按压次数: {total_valid_count}")
        except Exception:
            print("[WARN] 计数文件损坏，重置。")
            total_valid_count = 0
    else:
        total_valid_count = 0


def save_press_count():
    """保存累计次数到文件"""
    with open(COUNT_FILE, "w") as f:
        json.dump({"total_valid_count": total_valid_count}, f)


# ---------- 核心检测函数 ----------
def detect_pressure_peaks(file_path=FILE_PATH):
    """
    检测最新有效按压峰值，返回列表:
    [ (local_idx, global_idx, press_value, time_ms), ... ]
    """
    global last_peak_time, total_valid_count, session_count

    if not os.path.exists(file_path):
        return []

    try:
        df = pd.read_csv(file_path, usecols=["time_ms", "press_sum_norm"])
    except Exception:
        return []

    if df.empty or len(df) < 10:
        return []

    df["press_sum_norm"] = pd.to_numeric(df["press_sum_norm"], errors="coerce")
    df = df.dropna(subset=["press_sum_norm"])

    time_ms = df["time_ms"].to_numpy(dtype=float)
    press = df["press_sum_norm"].to_numpy(dtype=float)

    # ---- 静止段判断 ----
    std_val = np.std(press[-50:])
    if std_val < STATIC_STD_THRESH:
        return []

    # ---- 初步峰值检测 ----
    pos_locs, _ = find_peaks(press, prominence=PROMINENCE, distance=MIN_DISTANCE)
    if len(pos_locs) == 0:
        return []

    peaks_time = time_ms[pos_locs]
    peaks_val  = press[pos_locs]

    # ---- 合并10ms内的近邻峰，只保留最大值 ----
    merged_times = []
    merged_vals = []
    if len(peaks_time) > 0:
        group_start = 0
        for i in range(1, len(peaks_time)):
            # 若时间间隔 <= MERGE_WINDOW_MS，继续归入同组
            if peaks_time[i] - peaks_time[i - 1] <= MERGE_WINDOW_MS:
                continue
            else:
                # 结束一组 [group_start, i)
                group_slice = slice(group_start, i)
                max_idx = np.argmax(peaks_val[group_slice]) + group_start
                merged_times.append(peaks_time[max_idx])
                merged_vals.append(peaks_val[max_idx])
                group_start = i
        # 最后一组
        group_slice = slice(group_start, len(peaks_time))
        max_idx = np.argmax(peaks_val[group_slice]) + group_start
        merged_times.append(peaks_time[max_idx])
        merged_vals.append(peaks_val[max_idx])

    peaks_time = np.array(merged_times)
    peaks_val  = np.array(merged_vals)

    # ---- 有效按压筛选 ----
    valid_mask = peaks_val > VALID_PRESS_THRESH
    peaks_time = peaks_time[valid_mask]
    peaks_val  = peaks_val[valid_mask]

    if len(peaks_time) == 0:
        return []

    # ---- 新峰值过滤（防止重复统计）----
    new_idx = peaks_time > last_peak_time
    if not np.any(new_idx):
        return []

    new_times = peaks_time[new_idx]
    new_vals = peaks_val[new_idx]

    # ---- 更新计数 ----
    press_ids_global = [total_valid_count + i + 1 for i in range(len(new_times))]
    session_start_id = session_count + 1
    session_count += len(new_times)
    total_valid_count += len(new_times)
    last_peak_time = new_times[-1]
    save_press_count()

    # ---- 构造结果 ----
    results = []
    for i, (gid, val, t_ms) in enumerate(zip(press_ids_global, new_vals, new_times), start=session_start_id):
        results.append((i, gid, val, t_ms))
        print(f"✅ 第 {i} 次有效按压: {val:.0f} @ {t_ms:.0f} ms")

    return results


def get_total_count():
    """返回当前累计按压次数"""
    return total_valid_count


# ---------- 测试入口 ----------
# if __name__ == "__main__":
#     init_pressure_detector()
#     while True:
#         peaks = detect_pressure_peaks()
#         if len(peaks) == 0:
#             print("⏸ 无新有效峰值")
#         time.sleep(0.5)
if __name__ == "__main__":
    FILE_PATH = "/home/ljy/project/hand_dec/ljy/ljy_1/pressure_log.csv"
    init_pressure_detector()
    for _ in range(3):
        detect_pressure_peaks(FILE_PATH)
        time.sleep(0.2)


