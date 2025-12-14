#!/usr/bin/env python3
# analyze_peak_sync_improved.py
#
# 改进版峰值同步分析：
#   - 更智能的峰值检测（考虑视觉数据特点）
#   - 自适应阈值
#   - 更详细的峰值信息

import argparse
import csv
import json
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
    """加载完整视觉时间序列"""
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


def detect_peaks_adaptive(times_ms, vals, 
                         smooth_window=5,
                         percentile_threshold=60,
                         min_distance_ms=150,
                         min_prominence_ratio=0.2):
    """
    自适应峰值检测：
    1. 滑动平均平滑
    2. 使用百分位数设定阈值
    3. 检测局部极大值
    4. 要求峰值显著性（prominence）
    5. 时间不应期
    
    返回：list of (time_ms, value)
    """
    n = len(vals)
    if n < 3:
        return []
    
    # 1) 平滑
    smooth = np.copy(vals).astype(float)
    if smooth_window > 1:
        k = smooth_window
        kernel = np.ones(k) / k
        smooth = np.convolve(smooth, kernel, mode="same")
    
    # 2) 自适应阈值：使用百分位数
    threshold = np.percentile(smooth, percentile_threshold)
    
    # 计算全局范围，用于显著性判断
    signal_range = np.max(smooth) - np.min(smooth)
    min_prominence = signal_range * min_prominence_ratio
    
    print(f"[INFO] Adaptive threshold:")
    print(f"  - Value threshold (P{percentile_threshold}): {threshold:.2f}")
    print(f"  - Min prominence ({min_prominence_ratio*100}% of range): {min_prominence:.2f}")
    
    peaks = []
    last_peak_t = None
    
    for i in range(1, n - 1):
        prev_v = smooth[i - 1]
        cur_v = smooth[i]
        next_v = smooth[i + 1]
        
        # 局部极大值
        if not (cur_v >= prev_v and cur_v >= next_v):
            continue
        
        # 阈值过滤
        if cur_v < threshold:
            continue
        
        # 计算峰值显著性（prominence）
        # 向左找到最近的谷值
        left_min = cur_v
        for j in range(i - 1, max(0, i - 50), -1):
            if smooth[j] < left_min:
                left_min = smooth[j]
            if j > 0 and smooth[j] < smooth[j-1]:  # 找到谷底
                break
        
        # 向右找到最近的谷值
        right_min = cur_v
        for j in range(i + 1, min(n, i + 50)):
            if smooth[j] < right_min:
                right_min = smooth[j]
            if j < n - 1 and smooth[j] < smooth[j+1]:  # 找到谷底
                break
        
        prominence = cur_v - max(left_min, right_min)
        
        if prominence < min_prominence:
            continue
        
        t = times_ms[i]
        v_raw = vals[i]
        
        # 时间不应期
        if last_peak_t is not None and (t - last_peak_t) < min_distance_ms:
            continue
        
        peaks.append((float(t), float(v_raw)))
        last_peak_t = t
    
    print(f"[INFO] Detected {len(peaks)} peaks")
    return peaks


def estimate_time_offset_cross_correlation(t_press, v_press, t_vis, d_vis, dt_ms=10):
    """使用互相关估计最佳时间偏移"""
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
    max_lag_samples = int(5000 / dt_ms)
    correlation = signal.correlate(s_press, s_vis, mode='same')
    lags = signal.correlation_lags(len(s_press), len(s_vis), mode='same')
    
    mask = np.abs(lags) <= max_lag_samples
    lags_limited = lags[mask]
    correlation_limited = correlation[mask]
    
    best_idx = np.argmax(correlation_limited)
    best_lag_samples = lags_limited[best_idx]
    best_lag_ms = best_lag_samples * dt_ms
    
    print(f"[INFO] Estimated time offset: {best_lag_ms:.0f} ms")
    
    return best_lag_ms


def pair_peaks_by_closest_time(press_peaks, vis_peaks, max_dt_ms=500):
    """按照时间最接近原则配对峰值

    如果 `max_dt_ms` 为 None，则表示不在此处施加硬阈值（用于先行匹配以
    计算鲁棒统计量）；否则使用指定的阈值进行配对。
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
        time_diffs = np.abs(press_times - t_vis)
        closest_idx = np.argmin(time_diffs)
        min_diff = time_diffs[closest_idx]

        if (max_dt_ms is None) or (min_diff <= max_dt_ms):
            if closest_idx not in used_press_idx:
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

    if max_dt_ms is None:
        print(f"[INFO] Paired {len(pairs)} peaks (tolerance=auto)")
    else:
        print(f"[INFO] Paired {len(pairs)} peaks (tolerance={max_dt_ms}ms)")

    # 打印未配对的峰值
    unpaired_vis = len(vis_peaks) - len(pairs)
    unpaired_press = len([i for i in range(len(press_peaks)) if i not in used_press_idx])

    if unpaired_vis > 0:
        print(f"[WARN] {unpaired_vis} vision peaks were not paired")
    if unpaired_press > 0:
        print(f"[WARN] {unpaired_press} pressure peaks were not paired")

    return pairs


def compute_data_driven_tolerance(press_peaks, vis_peaks,
                                  min_tol_ms=100,
                                  max_tol_ms=1000,
                                  k=3.0):
    """
    根据数据自适应计算时间容忍区间（毫秒）。

    算法：
      1) 对每个视觉峰找到最近的压力峰（不施加阈值），计算初始 Δt 列表
      2) 使用中位数绝对偏差(MAD)估计鲁棒标准差：robust_std = 1.4826 * MAD
      3) 推荐容忍区间 = max(min_tol_ms, min(max_tol_ms, k * robust_std))

    设计理由：MAD 对异常值不敏感，乘以 k≈3 可覆盖大部分常见误差分布；
    同时设置下限以考虑压力采样的量化误差，以及上限避免过大容忍导致错误配对。
    """
    if not press_peaks or not vis_peaks:
        return float(min_tol_ms)

    press_times = np.array([p[0] for p in press_peaks])
    vis_times = np.array([p[0] for p in vis_peaks])

    # 建立最近邻匹配以获得初始 Δt（press - vis）
    dts = []
    for t_vis in vis_times:
        idx = np.argmin(np.abs(press_times - t_vis))
        dts.append(press_times[idx] - t_vis)
    dts = np.array(dts, dtype=float)

    if dts.size == 0:
        return float(min_tol_ms)

    med = np.median(dts)
    mad = np.median(np.abs(dts - med))
    robust_std = 1.4826 * mad

    # 若 robust_std 极小，使用压力采样量化作为下限
    if robust_std < 1.0 and len(press_times) >= 2:
        median_dt = np.median(np.diff(np.sort(press_times)))
        sample_quant_ms = max(1.0, median_dt * 0.5)
        min_tol_ms = max(min_tol_ms, sample_quant_ms)

    tol = max(min_tol_ms, min(max_tol_ms, k * max(robust_std, 1.0)))

    print(f"[INFO] Data-driven tolerance calculation:")
    print(f"  - med(Δt) = {med:.1f} ms, MAD = {mad:.1f} ms, robust_std ≈ {robust_std:.1f} ms")
    print(f"  - recommended tolerance = {tol:.0f} ms (k={k}, min={min_tol_ms}, max={max_tol_ms})")

    return float(tol)


def stats_and_plots(pairs, offset_ms=0, out_json=None):
    if not pairs:
        print("[ERR] No peak pairs to analyze")
        return None
    
    dt = np.array([p["dt_ms"] for p in pairs], dtype=float)
    n = len(dt)
    
    mu = float(np.mean(dt))
    sigma = float(np.std(dt))
    dt_min = float(np.min(dt))
    dt_max = float(np.max(dt))
    median_dt = float(np.median(dt))
    
    print("=" * 60)
    print("峰值时间差统计 Δt = t_press - t_vis (ms)")
    print("=" * 60)
    print(f"配对数量            = {n}")
    print(f"均值 mean(Δt)       = {mu:.3f} ms")
    print(f"中位数 median(Δt)   = {median_dt:.3f} ms")
    print(f"标准差 std(Δt)      = {sigma:.3f} ms")
    print(f"最小值 min(Δt)      = {dt_min:.3f} ms")
    print(f"最大值 max(Δt)      = {dt_max:.3f} ms")
    if offset_ms != 0:
        print(f"建议偏移补偿        = {offset_ms:.0f} ms")
        print(f"校正后的均值        = {mu - offset_ms:.3f} ms")
    
    # 详细配对表
    print("\n详细配对表（前10对）:")
    print("Vis# | Press# |   Vis_t   |  Press_t  |    Δt(ms)")
    print("-" * 55)
    for i, p in enumerate(pairs[:10]):
        print(f"{p['vis_idx']:4d} | {p['press_idx']:6d} | "
              f"{p['vis_time']/1000:9.2f} | {p['press_time']/1000:9.2f} | "
              f"{p['dt_ms']:8.1f}")
    if len(pairs) > 10:
        print(f"... ({len(pairs) - 10} more pairs)")
    
    # 图表
    vis_idx = np.array([p["vis_idx"] for p in pairs], dtype=int)
    
    plt.figure(figsize=(14, 9))
    
    # 子图1：时间差 vs 序号
    plt.subplot(2, 2, 1)
    plt.plot(vis_idx, dt, marker='o', linestyle='-', label='Measured Δt', alpha=0.7)
    plt.axhline(0.0, color='k', linestyle='--', linewidth=0.5)
    plt.axhline(mu, color='r', linestyle='--', label=f'Mean = {mu:.1f} ms', linewidth=1.5)
    plt.axhline(median_dt, color='orange', linestyle='--', label=f'Median = {median_dt:.1f} ms', linewidth=1.5)
    plt.fill_between(vis_idx, mu - sigma, mu + sigma, alpha=0.2, color='red', label=f'±1σ')
    plt.xlabel('Vision Peak Index')
    plt.ylabel('Δt (ms)')
    plt.title('Time Difference vs Peak Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：直方图
    plt.subplot(2, 2, 2)
    plt.hist(dt, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'Mean = {mu:.1f} ms')
    plt.axvline(median_dt, color='orange', linestyle='--', linewidth=2, label=f'Median = {median_dt:.1f} ms')
    plt.xlabel('Δt (ms)')
    plt.ylabel('Count')
    plt.title('Distribution of Time Differences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：散点图（时间相关性）
    plt.subplot(2, 2, 3)
    press_times = np.array([p["press_time"] for p in pairs])
    vis_times = np.array([p["vis_time"] for p in pairs])
    t0 = min(press_times[0], vis_times[0])
    press_times_rel = (press_times - t0) / 1000
    vis_times_rel = (vis_times - t0) / 1000
    
    plt.scatter(vis_times_rel, press_times_rel, alpha=0.6, s=50)
    min_t = min(vis_times_rel.min(), press_times_rel.min())
    max_t = max(vis_times_rel.max(), press_times_rel.max())
    plt.plot([min_t, max_t], [min_t, max_t], 'r--', label='Perfect alignment', linewidth=2)
    
    from numpy.polynomial import polynomial as P
    coefs = P.polyfit(vis_times_rel, press_times_rel, 1)
    fit_line = P.polyval(vis_times_rel, coefs)
    plt.plot(vis_times_rel, fit_line, 'g-', 
             label=f'Fit (slope={coefs[1]:.4f})', linewidth=2)
    
    plt.xlabel('Vision Peak Time (s)')
    plt.ylabel('Pressure Peak Time (s)')
    plt.title('Peak Time Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 子图4：误差分布（箱线图）
    plt.subplot(2, 2, 4)
    plt.boxplot(dt, vert=True)
    plt.ylabel('Δt (ms)')
    plt.title('Time Difference Distribution (Boxplot)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('peak_sync_error_analysis.svg')
    print("\n[FIG] saved: peak_sync_error_analysis.svg")

    # Prepare summary dict
    summary = {
        'n_pairs': int(n),
        'mean_dt_ms': float(mu),
        'median_dt_ms': float(median_dt),
        'std_dt_ms': float(sigma),
        'min_dt_ms': float(dt_min),
        'max_dt_ms': float(dt_max),
        'offset_ms': float(offset_ms),
    }

    if out_json:
        try:
            with open(out_json, 'w') as jf:
                json.dump(summary, jf, indent=2)
            print(f"[INFO] summary written to {out_json}")
        except Exception as e:
            print(f"[WARN] failed to write summary json: {e}")

    return summary


def main():
    print("=" * 60)
    print("改进版峰值同步分析")
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
    
    # 3. 检测峰值（压力数据：使用简单方法，不用显著性检测）
    print("\n[3] Detecting pressure peaks...")
    press_peaks = detect_peaks_adaptive(
        t_press, v_press,
        smooth_window=5,
        percentile_threshold=60,
        min_distance_ms=150,
        min_prominence_ratio=0.0  # 不使用显著性要求
    )
    
    print("\n[4] Detecting vision peaks...")
    vis_peaks = detect_peaks_adaptive(
        t_vis, d_vis,
        smooth_window=5,
        percentile_threshold=50,  # 视觉数据用更低的阈值，检测第一个峰
        min_distance_ms=150,
        min_prominence_ratio=0.1  # 视觉数据用较低的显著性要求
    )
    
    # 4. 配对峰值
    parser = argparse.ArgumentParser(description='Peak sync improved with optional fixed tolerance')
    parser.add_argument('--fixed-tol', type=int, default=None,
                        help='Use a fixed tolerance (ms) for pairing instead of data-driven recommendation')
    parser.add_argument('--out-json', type=str, default=None,
                        help='Write per-run summary JSON to this path')
    args = parser.parse_args()

    print("\n[5] Pairing peaks...")
    # 计算数据驱动的配对容忍区间
    print("\n[5a] Computing data-driven time tolerance...")
    recommended_tol = compute_data_driven_tolerance(press_peaks, vis_peaks,
                                                   min_tol_ms=100,
                                                   max_tol_ms=1000,
                                                   k=3.0)
    if args.fixed_tol is not None:
        use_tol = int(args.fixed_tol)
        print(f"[INFO] Using fixed tolerance = {use_tol} ms for pairing (overrides data-driven)")
    else:
        use_tol = int(recommended_tol)
        print(f"[INFO] Using recommended tolerance = {use_tol} ms for pairing")

    pairs = pair_peaks_by_closest_time(press_peaks, vis_peaks, max_dt_ms=use_tol)
    
    # 5. 统计与可视化
    print("\n[6] Statistics and visualization...")
    summary = stats_and_plots(pairs, offset_ms, out_json=args.out_json)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
