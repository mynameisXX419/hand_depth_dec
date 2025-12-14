#!/usr/bin/env python3
# analyze_peak_sync_corrected.py
#
# 带时间偏移校正的峰值同步分析
# 自动检测并应用最佳时间偏移量

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate


PRESSURE_CSV = "pressure_series.csv"
VISION_CSV = "hand_depth_plane_avg.csv"


def load_pressure_series(path):
    host_ms = []
    vals = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
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


def load_vision_series(path):
    """加载完整视觉时间序列（用于互相关分析）"""
    times = []
    depths = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        col_idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            try:
                t = int(float(row[col_idx["timestamp_ms"]]))
                d = float(row[col_idx["depth_corr_mm"]])
                times.append(t)
                depths.append(d)
            except (KeyError, ValueError, IndexError):
                continue
    return np.array(times, dtype=float), np.array(depths, dtype=float)


def detect_peaks_offline(times_ms, vals,
                         smooth_window=5,
                         threshold=None,
                         min_distance_ms=150):
    """离线峰值检测"""
    n = len(vals)
    if n < 3:
        return []

    smooth = np.copy(vals).astype(float)
    if smooth_window > 1:
        k = smooth_window
        kernel = np.ones(k) / k
        smooth = np.convolve(smooth, kernel, mode="same")

    if threshold is None:
        mu = float(np.mean(smooth))
        sigma = float(np.std(smooth))
        # 降低阈值系数，从 0.5 改为 0.3，以检测更多峰值
        threshold = mu + 0.3 * sigma
        print(f"[INFO] auto threshold = {threshold:.1f} (mu={mu:.1f}, sigma={sigma:.1f})")

    peaks = []
    last_peak_t = None

    for i in range(1, n - 1):
        prev_v = smooth[i - 1]
        cur_v = smooth[i]
        next_v = smooth[i + 1]

        if not (cur_v >= prev_v and cur_v >= next_v):
            continue

        if cur_v < threshold:
            continue

        t = times_ms[i]
        v_raw = vals[i]

        if last_peak_t is not None and (t - last_peak_t) < min_distance_ms:
            continue

        peaks.append((float(t), float(v_raw)))
        last_peak_t = t

    print(f"[INFO] detected {len(peaks)} peaks.")
    return peaks


def estimate_time_offset_cross_correlation(t_press, v_press, t_vis, d_vis, dt_ms=10):
    """
    使用互相关估计最佳时间偏移
    返回：偏移量（毫秒），正值表示压力滞后于视觉
    """
    # 重采样到统一时间轴
    t_start = max(np.min(t_press), np.min(t_vis))
    t_end = min(np.max(t_press), np.max(t_vis))
    
    if t_end <= t_start:
        print("[WARN] No overlapping time, cannot estimate offset")
        return 0.0
    
    t_common = np.arange(t_start, t_end, dt_ms)
    
    f_press = interpolate.interp1d(t_press, v_press, kind='linear', bounds_error=False, fill_value=0)
    f_vis = interpolate.interp1d(t_vis, d_vis, kind='linear', bounds_error=False, fill_value=0)
    
    s_press = f_press(t_common)
    s_vis = f_vis(t_common)
    
    # 归一化
    s_press = (s_press - np.min(s_press)) / (np.max(s_press) - np.min(s_press) + 1e-6)
    s_vis = (s_vis - np.min(s_vis)) / (np.max(s_vis) - np.min(s_vis) + 1e-6)
    
    # 互相关
    max_lag_samples = int(5000 / dt_ms)  # ±5 秒
    correlation = signal.correlate(s_press, s_vis, mode='same')
    lags = signal.correlation_lags(len(s_press), len(s_vis), mode='same')
    
    mask = np.abs(lags) <= max_lag_samples
    lags_limited = lags[mask]
    correlation_limited = correlation[mask]
    
    best_idx = np.argmax(correlation_limited)
    best_lag_samples = lags_limited[best_idx]
    best_lag_ms = best_lag_samples * dt_ms
    
    print(f"[INFO] Estimated time offset: {best_lag_ms:.0f} ms (pressure {'lags' if best_lag_ms > 0 else 'leads'} vision)")
    
    return best_lag_ms


def pair_peaks_by_closest_time(press_peaks, vis_peaks, max_dt_ms=500):
    """
    按照时间最接近原则配对峰值
    每个视觉峰值找最近的压力峰值，时间差 < max_dt_ms
    
    返回：配对列表
    """
    if not press_peaks or not vis_peaks:
        return []
    
    press_times = np.array([p[0] for p in press_peaks])
    press_vals = np.array([p[1] for p in press_peaks])
    
    vis_times = np.array([p[0] for p in vis_peaks])
    vis_vals = np.array([p[1] for p in vis_peaks])
    
    pairs = []
    used_press_idx = set()
    
    for i, t_vis in enumerate(vis_times):
        # 找最近的压力峰值
        time_diffs = np.abs(press_times - t_vis)
        closest_idx = np.argmin(time_diffs)
        min_diff = time_diffs[closest_idx]
        
        if min_diff <= max_dt_ms and closest_idx not in used_press_idx:
            pairs.append({
                "vis_idx": i,
                "press_idx": closest_idx,
                "vis_time": float(t_vis),
                "press_time": float(press_times[closest_idx]),
                "vis_val": float(vis_vals[i]),
                "press_val": float(press_vals[closest_idx]),
                "dt_ms": float(press_times[closest_idx] - t_vis),
            })
            used_press_idx.add(closest_idx)
    
    print(f"[INFO] Paired {len(pairs)} peaks (by closest time, tolerance={max_dt_ms}ms)")
    return pairs


def stats_and_plots(pairs, offset_ms=0):
    if not pairs:
        print("[ERR] No peak pairs to analyze")
        return
    
    dt = np.array([p["dt_ms"] for p in pairs], dtype=float)
    n = len(dt)
    
    mu = float(np.mean(dt))
    sigma = float(np.std(dt))
    dt_min = float(np.min(dt))
    dt_max = float(np.max(dt))
    
    print("=" * 60)
    print("峰值时间差 Δt = t_press - t_vis (ms) 统计")
    print("=" * 60)
    print(f"配对数量            = {n}")
    print(f"均值 mean(Δt)       = {mu:.3f} ms")
    print(f"标准差 std(Δt)      = {sigma:.3f} ms")
    print(f"最小值 min(Δt)      = {dt_min:.3f} ms")
    print(f"最大值 max(Δt)      = {dt_max:.3f} ms")
    print(f"建议的时间偏移补偿   = {offset_ms:.0f} ms")
    print(f"校正后的均值        = {mu - offset_ms:.3f} ms")
    
    # 图1：Δt vs 峰值序号
    vis_idx = np.array([p["vis_idx"] for p in pairs], dtype=int)
    
    plt.figure(figsize=(12, 5))
    plt.plot(vis_idx, dt, marker='o', linestyle='-', label='Measured Δt')
    plt.axhline(0.0, color='k', linestyle='--', linewidth=0.5)
    plt.axhline(mu, color='r', linestyle='--', label=f'Mean = {mu:.1f} ms')
    if offset_ms != 0:
        plt.axhline(offset_ms, color='g', linestyle='--', label=f'Estimated offset = {offset_ms:.1f} ms')
    plt.xlabel('Peak Index')
    plt.ylabel('Δt = t_press - t_vis (ms)')
    plt.title('Peak Time Difference (corrected pairing)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('peak_sync_corrected_vs_idx.svg')
    print("[FIG] saved: peak_sync_corrected_vs_idx.svg")
    
    # 图2：Δt 直方图
    plt.figure(figsize=(10, 6))
    plt.hist(dt, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'Mean = {mu:.1f} ms')
    if offset_ms != 0:
        plt.axvline(offset_ms, color='g', linestyle='--', linewidth=2, 
                   label=f'Estimated offset = {offset_ms:.1f} ms')
    plt.xlabel('Δt (ms)')
    plt.ylabel('Count')
    plt.title('Histogram of Peak Time Difference (corrected)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('peak_sync_corrected_hist.svg')
    print("[FIG] saved: peak_sync_corrected_hist.svg")
    
    # 图3：散点图（压力时间 vs 视觉时间）
    press_times = np.array([p["press_time"] for p in pairs])
    vis_times = np.array([p["vis_time"] for p in pairs])
    
    # 转换为相对时间（秒）
    t0 = min(press_times[0], vis_times[0])
    press_times_rel = (press_times - t0) / 1000
    vis_times_rel = (vis_times - t0) / 1000
    
    plt.figure(figsize=(10, 10))
    plt.scatter(vis_times_rel, press_times_rel, alpha=0.6, s=50)
    
    # 绘制 y=x 线（完美对齐）
    min_t = min(vis_times_rel.min(), press_times_rel.min())
    max_t = max(vis_times_rel.max(), press_times_rel.max())
    plt.plot([min_t, max_t], [min_t, max_t], 'r--', label='Perfect alignment', linewidth=2)
    
    # 绘制实际拟合线
    from numpy.polynomial import polynomial as P
    coefs = P.polyfit(vis_times_rel, press_times_rel, 1)
    fit_line = P.polyval(vis_times_rel, coefs)
    plt.plot(vis_times_rel, fit_line, 'g-', label=f'Actual fit (slope={coefs[1]:.3f})', linewidth=2)
    
    plt.xlabel('Vision Peak Time (s)')
    plt.ylabel('Pressure Peak Time (s)')
    plt.title('Peak Time Correlation')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('peak_time_correlation.svg')
    print("[FIG] saved: peak_time_correlation.svg")


def main():
    print("=" * 60)
    print("校正版峰值同步分析")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] Loading data...")
    t_press, v_press = load_pressure_series(PRESSURE_CSV)
    t_vis, d_vis = load_vision_series(VISION_CSV)
    
    print(f"  Pressure: {len(t_press)} samples")
    print(f"  Vision:   {len(t_vis)} samples")
    
    # 2. 估计时间偏移
    print("\n[2] Estimating time offset...")
    offset_ms = estimate_time_offset_cross_correlation(t_press, v_press, t_vis, d_vis)
    
    # 3. 检测峰值
    print("\n[3] Detecting peaks...")
    press_peaks = detect_peaks_offline(
        t_press, v_press,
        smooth_window=5,
        threshold=None,
        min_distance_ms=150,
    )
    
    vis_peaks = detect_peaks_offline(
        t_vis, d_vis,
        smooth_window=5,
        threshold=None,
        min_distance_ms=150,
    )
    
    # 4. 配对峰值（按最近时间）
    print("\n[4] Pairing peaks by closest time...")
    pairs = pair_peaks_by_closest_time(press_peaks, vis_peaks, max_dt_ms=500)
    
    # 5. 统计与可视化
    print("\n[5] Statistics and visualization...")
    stats_and_plots(pairs, offset_ms)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
