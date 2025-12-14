#!/usr/bin/env python3
# analyze_peak_sync_offline.py
#
# 目的：
#   1. 从 pressure_series.csv 里做“离线压力峰值检测”
#   2. 从 vision_peaks.csv 里读出视觉峰
#   3. 按顺序一一对应，计算 Δt = t_press - t_vis 的分布
#   4. 输出统计量 + 保存几张图（SVG）

import csv
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt


PRESSURE_CSV = "pressure_series.csv"
VISION_CSV = "hand_depth_plane_avg.csv"


def load_pressure_series(path):
    host_ms = []
    vals = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # 找一下列索引（兼容列顺序)
        col_idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            try:
                host = int(row[col_idx["host_ms"]])
                v = int(row[col_idx["val"]])
            except (KeyError, ValueError, IndexError):
                continue
            host_ms.append(host)
            vals.append(v)
    return np.array(host_ms, dtype=float), np.array(vals, dtype=float)


def load_vision_peaks(path):
    vis_ts_ms = []
    vis_idx = []
    vis_depth = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        col_idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            try:
                # 修改列名匹配实际CSV文件：frame_idx, timestamp_ms, depth_corr_mm
                idx = int(row[col_idx["frame_idx"]])
                depth = float(row[col_idx["depth_corr_mm"]])  # 使用校正后的深度
                ts_ms = int(float(row[col_idx["timestamp_ms"]]))
            except (KeyError, ValueError, IndexError):
                continue
            vis_idx.append(idx)
            vis_depth.append(depth)
            vis_ts_ms.append(ts_ms)
    return np.array(vis_idx), np.array(vis_depth), np.array(vis_ts_ms, dtype=float)


def detect_peaks_offline(times_ms, vals,
                         smooth_window=5,
                         threshold=None,
                         min_distance_ms=150):
    """
    离线版本压力峰值检测：
      1. 对 val 做 smooth_window 点滑动平均
      2. 找局部极大值
      3. 振幅阈值：大于 threshold（如果没给，就用均值+1std 或类似）
      4. 不应期：两个峰之间时间间隔 >= min_distance_ms

    返回：list of (t_peak_ms, val_peak)
    """
    n = len(vals)
    if n < 3:
        return []

    # 1) 简单滑动平均
    # 为了简单，首尾直接复制，内部卷积
    smooth = np.copy(vals).astype(float)
    if smooth_window > 1:
        k = smooth_window
        kernel = np.ones(k) / k
        # 用 'same' 卷积
        smooth = np.convolve(smooth, kernel, mode="same")

    # 2) 默认阈值：如果没给，就用 (均值 + 0.5 * std)
    if threshold is None:
        mu = float(np.mean(smooth))
        sigma = float(np.std(smooth))
        threshold = mu + 0.5 * sigma
        print(f"[INFO] auto threshold = {threshold:.1f} (mu={mu:.1f}, sigma={sigma:.1f})")

    peaks = []
    last_peak_t = None

    for i in range(1, n - 1):
        prev_v = smooth[i - 1]
        cur_v = smooth[i]
        next_v = smooth[i + 1]

        # 局部极大值
        if not (cur_v >= prev_v and cur_v >= next_v):
            continue

        # 振幅阈值
        if cur_v < threshold:
            continue

        t = times_ms[i]
        v_raw = vals[i]

        # 不应期
        if last_peak_t is not None and (t - last_peak_t) < min_distance_ms:
            continue

        peaks.append((float(t), float(v_raw)))
        last_peak_t = t

    print(f"[INFO] detected {len(peaks)} pressure peaks (offline).")
    return peaks


def pair_peaks_by_order(press_peaks, vis_ts_ms, vis_idx, vis_depth):
    """
    按顺序一一对应：
      press_peaks[i] ↔ visu_peaks[i]
    直到 min(len(press_peaks), len(vis_ts_ms))

    返回：
      pairs: list of dict{
        'vis_idx', 'vis_depth',
        't_vis_ms', 't_press_ms', 'dt_ms'
      }
    """
    n = min(len(press_peaks), len(vis_ts_ms))
    pairs = []
    for i in range(n):
        t_press, v_press = press_peaks[i]
        t_vis = vis_ts_ms[i]
        pairs.append({
            "vis_idx": int(vis_idx[i]),
            "vis_depth": float(vis_depth[i]),
            "t_vis_ms": float(t_vis),
            "t_press_ms": float(t_press),
            "dt_ms": float(t_press - t_vis),
        })
    print(f"[INFO] paired {n} peaks (by order).")
    return pairs


def stats_and_plots(pairs):
    if not pairs:
        print("[ERR] no peak pairs, check your CSVs.")
        return

    dt = np.array([p["dt_ms"] for p in pairs], dtype=float)
    n = len(dt)
    mu = float(np.mean(dt))
    sigma = float(np.std(dt))
    dt_min = float(np.min(dt))
    dt_max = float(np.max(dt))

    print("=== 峰值时间差 Δt = t_press - t_vis (ms) 统计 ===")
    print(f"样本数 n           = {n}")
    print(f"均值 mean(Δt)      = {mu:.3f} ms")
    print(f"标准差 std(Δt)     = {sigma:.3f} ms")
    print(f"最小值 min(Δt)     = {dt_min:.3f} ms")
    print(f"最大值 max(Δt)     = {dt_max:.3f} ms")

    # 画图：1) Δt 随按压序号变化；2) 直方图
    # 图1：Δt vs 视觉按压序号
    vis_idx = np.array([p["vis_idx"] for p in pairs], dtype=int)

    plt.figure()
    plt.plot(vis_idx, dt, marker="o", linestyle="-")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Visual compression index")
    plt.ylabel("Δt = t_press - t_vis (ms)")
    plt.title("Peak time difference (pressure vs vision)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("peak_dt_vs_idx.svg")
    print("[FIG] saved SVG: peak_dt_vs_idx.svg")

    # 图2：Δt 直方图
    plt.figure()
    plt.hist(dt, bins=20)
    plt.xlabel("Δt (ms)")
    plt.ylabel("Count")
    plt.title("Histogram of peak time difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("peak_dt_hist.svg")
    print("[FIG] saved SVG: peak_dt_hist.svg")


def main():
    print(f"[INFO] loading pressure series from '{PRESSURE_CSV}' ...")
    t_press_ms, vals = load_pressure_series(PRESSURE_CSV)
    print(f"[INFO] {len(vals)} pressure samples loaded.")

    print(f"[INFO] loading vision peaks from '{VISION_CSV}' ...")
    vis_idx, vis_depth, vis_ts_ms = load_vision_peaks(VISION_CSV)
    print(f"[INFO] {len(vis_idx)} vision peaks loaded.")

    # 1) 离线检测压力峰
    press_peaks = detect_peaks_offline(
        t_press_ms, vals,
        smooth_window=5,
        threshold=None,        # 让脚本自动给一个阈值
        min_distance_ms=150,   # 按压间隔 ~500-600ms，150ms 不会把真峰压掉
    )

    # 2) 顺序配对峰值
    pairs = pair_peaks_by_order(press_peaks, vis_ts_ms, vis_idx, vis_depth)

    # 3) 统计 & 画图
    stats_and_plots(pairs)


if __name__ == "__main__":
    main()
