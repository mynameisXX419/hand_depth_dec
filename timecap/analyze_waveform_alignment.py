#!/usr/bin/env python3
# analyze_waveform_alignment_with_pairing_print.py

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
            except Exception:
                pass
    return np.array(t), np.array(x)


def load_pressure_series(path="pressure_series.csv",
                         t_key="host_ms",
                         prefer_key="val_filt",
                         fallback_key="val_raw"):
    t, x = [], []
    used_key = None

    with open(path) as f:
        r = csv.DictReader(f)
        if prefer_key in r.fieldnames:
            used_key = prefer_key
        elif fallback_key in r.fieldnames:
            used_key = fallback_key
        else:
            raise RuntimeError("pressure CSV missing val_filt / val_raw")

        for row in r:
            try:
                t.append(float(row[t_key]))
                x.append(float(row[used_key]))
            except Exception:
                pass

    return np.array(t), np.array(x), used_key


def normalize(x):
    return (x - x.min()) / (x.ptp() + 1e-8)


# ============================================================
# 重采样
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
# 互相关
# ============================================================

def best_lag(a, b, dt_ms):
    a_n = normalize(a)
    b_n = normalize(b)
    corr = correlate(a_n, b_n, mode="same")
    lags = correlation_lags(len(a_n), len(b_n), mode="same")
    i = np.argmax(corr)
    return lags[i] * dt_ms, corr[i] / len(a_n)


# ============================================================
# 峰检测
# ============================================================

def detect_peaks_pressure(t, x, thr_percentile=60, min_dt_ms=250):
    thr = np.percentile(x, thr_percentile)
    peaks, last = [], -1e12

    for i in range(1, len(x)-1):
        if x[i] >= x[i-1] and x[i] >= x[i+1] and x[i] >= thr:
            if t[i] - last >= min_dt_ms:
                peaks.append((t[i], x[i], i))
                last = t[i]

    if not peaks:
        return np.array([]), np.array([]), np.array([])

    pt, pv, pi = zip(*peaks)
    return np.array(pt), np.array(pv), np.array(pi)


def detect_peaks_vision_segmented(t, x, min_dt_ms=150):
    valleys = []
    for i in range(1, len(x)-1):
        if x[i] <= x[i-1] and x[i] <= x[i+1]:
            valleys.append(i)

    if not valleys or valleys[0] != 0:
        valleys = [0] + valleys
    if valleys[-1] != len(x)-1:
        valleys.append(len(x)-1)

    peaks, last_t = [], -1e12
    for a, b in zip(valleys[:-1], valleys[1:]):
        if b - a < 5:
            continue
        seg = x[a:b+1]
        idx = a + int(np.argmax(seg))
        if t[idx] - last_t >= min_dt_ms:
            peaks.append((t[idx], x[idx], idx))
            last_t = t[idx]

    if not peaks:
        return np.array([]), np.array([]), np.array([])

    pt, pv, pi = zip(*peaks)
    return np.array(pt), np.array(pv), np.array(pi)


def filter_visual_peaks(t, x, pt, pv, pi,
                        min_prominence=3.0,
                        min_rise=2.0,
                        min_cycle_ms=200):
    keep_pt, keep_pv, keep_pi = [], [], []

    for k in range(len(pt)):
        idx = int(pi[k])
        L, R = max(0, idx-50), min(len(x)-1, idx+50)
        valley = min(np.min(x[L:idx+1]), np.min(x[idx:R+1]))

        if pv[k] - valley < min_prominence:
            continue
        if pv[k] - np.min(x[L:idx+1]) < min_rise:
            continue
        if k > 0 and pt[k] - pt[k-1] < min_cycle_ms:
            continue

        keep_pt.append(pt[k])
        keep_pv.append(pv[k])
        keep_pi.append(pi[k])

    return np.array(keep_pt), np.array(keep_pv), np.array(keep_pi)


# ============================================================
# 严格序号配对
# ============================================================

def pair_peaks_with_index(tp, ip, tv, iv):
    n = min(len(tp), len(tv))
    pairs = []
    for k in range(n):
        pairs.append((
            k,
            int(ip[k]),
            int(iv[k]),
            float(tp[k]),
            float(tv[k]),
            float(tv[k] - tp[k])
        ))
    return pairs


# ============================================================
# 绘图（带配对线）
# ============================================================

def plot_peaks_with_pairs(t, p, v, ip, iv, pairs, label_p):
    pn, vn = normalize(p), normalize(v)

    plt.figure(figsize=(12, 4))
    plt.plot(t/1000, pn, label=label_p, alpha=0.7)
    plt.plot(t/1000, vn, label="Vision (norm)", alpha=0.7)

    plt.plot(t[ip]/1000, pn[ip], 'bo', label="P peaks")
    plt.plot(t[iv]/1000, vn[iv], 'ro', label="V peaks")

    for k, pi, vi, tp, tv, dt in pairs:
        plt.plot([tp/1000, tv/1000],
                 [pn[pi], vn[vi]],
                 'k--', alpha=0.6)
        if k < 6:
            plt.text((tp+tv)/2000,
                     (pn[pi]+vn[vi])/2,
                     f"{k}", fontsize=8)

    plt.grid(True)
    plt.legend()
    plt.title("Peak Alignment with Pairing Index")
    plt.tight_layout()
    plt.savefig("peaks_overlay_with_pairs.svg")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=== Dual Peak Alignment + Pair Print ===")

    tP, xP, used_key = load_pressure_series()
    tV, xV = load_csv("hand_depth_plane_avg.csv", "timestamp_ms", "depth_corr_mm")

    t, p, v = resample(tP, xP, tV, xV)
    if t is None:
        print("Timestamp range mismatch")
        return

    lag, corr = best_lag(p, v, 10)
    print(f"[Xcorr] lag={lag:.1f} ms | corr={corr:.3f}")

    tp, _, ip = detect_peaks_pressure(t, p)
    tv, vv, iv = detect_peaks_vision_segmented(t, v)
    tv, vv, iv = filter_visual_peaks(t, v, tv, vv, iv)

    pairs = pair_peaks_with_index(tp, ip, tv, iv)

    print("\n=== Peak Pairing Result ===")
    print(f"Total matched pairs: {len(pairs)}\n")

    for k, pi, vi, tp_k, tv_k, dt in pairs:
        print(
            f"[PAIR {k:02d}] "
            f"P_idx={pi:<4d}  V_idx={vi:<4d}  "
            f"P_t={tp_k/1000:.3f}s  "
            f"V_t={tv_k/1000:.3f}s  "
            f"Δt={dt:.1f} ms"
        )

    pd.DataFrame(pairs,
                 columns=["pair_id", "pressure_idx", "vision_idx",
                          "pressure_t_ms", "vision_t_ms", "dt_ms"]
                 ).to_csv("peak_pairs.csv", index=False)

    plot_peaks_with_pairs(
        t, p, v, ip, iv, pairs,
        label_p=f"Pressure ({used_key})"
    )

    print("\nSaved:")
    print(" - peak_pairs.csv")
    print(" - peaks_overlay_with_pairs.svg")


if __name__ == "__main__":
    main()
