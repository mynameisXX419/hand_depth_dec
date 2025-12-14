#!/usr/bin/env python3
# analyze_waveform_alignment.py
#
# 最终版 + 峰序号配对：
#   ✓ Pressure: local maxima
#   ✓ Vision: segmented valley→peak detection
#   ✓ Vision 强峰过滤
#   ✓ 一对一配对，输出总对数 + 每对峰序号
#   ✓ 输出 CSV: peak_pairs.csv
#   ✓ 输出两张 SVG 图

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from scipy import interpolate
import pandas as pd


# ============================================================
# 数据加载
# ============================================================

def load_csv(path, t_key, v_key):
    t, x = [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t.append(float(row[t_key]))
                x.append(float(row[v_key]))
            except:
                pass
    return np.array(t), np.array(x)

def normalize(x):
    return (x - x.min()) / (x.ptp() + 1e-8)


# ============================================================
# 重采样到共同时间轴
# ============================================================

def resample(t1, s1, t2, s2, dt_ms=10):
    t0, t1e = max(t1.min(), t2.min()), min(t1.max(), t2.max())
    if t1e <= t0:
        return None, None, None
    t = np.arange(t0, t1e, dt_ms)
    f1 = interpolate.interp1d(t1, s1, fill_value="extrapolate")
    f2 = interpolate.interp1d(t2, s2, fill_value="extrapolate")
    return t, f1(t), f2(t)


# ============================================================
# 整体互相关估计延迟
# ============================================================

def best_lag(a, b, dt_ms):
    a_n = normalize(a)
    b_n = normalize(b)
    corr = correlate(a_n, b_n, mode="same")
    lags = correlation_lags(len(a_n), len(b_n), mode="same")
    i = np.argmax(corr)
    return lags[i] * dt_ms, corr[i] / len(a_n)


# ============================================================
# 压力峰检测：尖峰 → local maxima
# ============================================================

def detect_peaks_pressure(t, x, thr_percentile=60, min_dt_ms=250):
    thr = np.percentile(x, thr_percentile)
    peaks = []
    last = -1e12

    for i in range(1, len(x)-1):
        if x[i] >= x[i-1] and x[i] >= x[i+1] and x[i] >= thr:
            if t[i] - last >= min_dt_ms:
                peaks.append((t[i], x[i], i))
                last = t[i]

    if not peaks:
        return np.array([]), np.array([]), np.array([])

    pt, pv, pi = zip(*peaks)
    return np.array(pt), np.array(pv), np.array(pi)


# ============================================================
# 视觉峰检测：valley→peak 分段
# ============================================================

def detect_peaks_vision_segmented(t, x, min_dt_ms=150):
    n = len(x)
    valleys = []

    for i in range(1, n-1):
        if x[i] <= x[i-1] and x[i] <= x[i+1]:
            valleys.append(i)

    if len(valleys) == 0 or valleys[0] != 0:
        valleys = [0] + valleys
    if valleys[-1] != n-1:
        valleys.append(n-1)

    peaks = []
    last_t = -1e12

    for k in range(len(valleys)-1):
        a = valleys[k]
        b = valleys[k+1]
        if b - a < 5:
            continue

        seg = x[a:b+1]
        peak_i = a + np.argmax(seg)

        if t[peak_i] - last_t < min_dt_ms:
            continue

        peaks.append((t[peak_i], x[peak_i], peak_i))
        last_t = t[peak_i]

    if not peaks:
        return np.array([]), np.array([]), np.array([])

    pt, pv, pi = zip(*peaks)
    return np.array(pt), np.array(pv), np.array(pi)


# ============================================================
# 视觉峰过滤
# ============================================================

def filter_visual_peaks(t, x, pt, pv, pi,
                        min_prominence=3.0,
                        min_rise=2.0,
                        min_cycle_ms=200):

    keep_pt, keep_pv, keep_pi = [], [], []

    for k in range(len(pt)):
        idx = pi[k]

        L = max(0, idx - 50)
        R = min(len(x)-1, idx + 50)

        valley_l = np.min(x[L:idx+1])
        valley_r = np.min(x[idx:R+1])
        valley = min(valley_l, valley_r)

        prominence = pv[k] - valley
        rise = pv[k] - valley_l

        if prominence < min_prominence:
            continue
        if rise < min_rise:
            continue
        if k > 0 and pt[k] - pt[k-1] < min_cycle_ms:
            continue

        keep_pt.append(pt[k])
        keep_pv.append(pv[k])
        keep_pi.append(pi[k])

    return np.array(keep_pt), np.array(keep_pv), np.array(keep_pi)


# ============================================================
# Δt 配对 + 峰序号配对
# ============================================================
def pair_peaks_with_index(tp, ip, tv, iv, tol_ms=400):
    """
    严格序号配对：
      Pressure[i] → Vision[i]
    Vision 若更多，只取前 len(tp) 个。
    返回：
      pairs = [ (pressure_idx, vision_idx, dt_ms) ]
    """
    nP = len(tp)
    nV = len(tv)

    n = min(nP, nV)   # 只匹配前 n 个

    pairs = []
    for k in range(n):
        dt = tv[k] - tp[k]    # vision - pressure 时间差
        pairs.append((ip[k], iv[k], dt))

    return pairs



# ============================================================
# 绘图
# ============================================================

def plot_overlay(t, p, v):
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t/1000, p, 'b-', label="Pressure raw")
    ax1.set_ylabel("Pressure")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t/1000, v, 'r-', label="Vision depth")
    ax2.set_ylabel("Vision depth (mm)")

    plt.title("Raw Signal Overlay")
    fig.tight_layout()
    plt.savefig("overlay_raw_dual_axis.svg")


def plot_peaks(t, p, v, ip, iv):
    plt.figure(figsize=(12,4))
    pn = normalize(p)
    vn = normalize(v)

    plt.plot(t/1000, pn, 'b-', label="Pressure")
    plt.plot(t/1000, vn, 'r-', label="Vision")

    if len(ip)>0: plt.plot(t[ip]/1000, pn[ip], 'bo', label="P peaks")
    if len(iv)>0: plt.plot(t[iv]/1000, vn[iv], 'ro', label="V peaks")

    plt.grid()
    plt.legend()
    plt.title("Peak Alignment")
    plt.tight_layout()
    plt.savefig("peaks_overlay.svg")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=== Final Dual Peak Alignment + Index Pairing ===")

    tP, xP = load_csv("pressure_series.csv", "host_ms", "val")
    tV, xV = load_csv("hand_depth_plane_avg.csv", "timestamp_ms", "depth_corr_mm")

    t, p, v = resample(tP, xP, tV, xV, dt_ms=10)

    if t is None:
        print("ERROR: timestamp ranges do not overlap")
        return

    lag, corr = best_lag(p, v, 10)
    print(f"[Xcorr] Best lag = {lag:.1f} ms | corr={corr:.3f}")

    # 压力峰
    tp, _, ip = detect_peaks_pressure(t, p)
    print("Pressure peaks =", len(tp))

    # 视觉峰
    tv, vv, iv = detect_peaks_vision_segmented(t, v)
    print("Vision raw peaks =", len(tv))

    tv, vv, iv = filter_visual_peaks(t, v, tv, vv, iv)
    print("Vision filtered peaks =", len(tv))

    # ★ 一对一配对（含峰序号）
    pairs = pair_peaks_with_index(tp, ip, tv, iv, tol_ms=400)

    print("\n=== Peak Pairing Result ===")
    print("Total matched =", len(pairs))

    for (idx_p, idx_v, dt) in pairs:
        print(f"Pressure[{idx_p}]  <-->  Vision[{idx_v}]     dt = {dt:.1f} ms")

    # 导出 CSV
    if len(pairs) > 0:
        df = pd.DataFrame({
            "pressure_idx": [p[0] for p in pairs],
            "vision_idx":   [p[1] for p in pairs],
            "dt_ms":        [p[2] for p in pairs]
        })
        df.to_csv("peak_pairs.csv", index=False)
        print("\nSaved: peak_pairs.csv")

    # 图
    plot_overlay(t, p, v)
    plot_peaks(t, p, v, ip, iv)

    print("Saved SVG files: overlay_raw_dual_axis.svg , peaks_overlay.svg")


if __name__ == "__main__":
    main()
