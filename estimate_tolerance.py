import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, correlate
import matplotlib.pyplot as plt   # ✅ 新增：画图用

# ================== 基本配置 ==================

# 根目录：包含 1,2,...,10 这些子目录
BASE_DIR = "/home/ljy/project/hand_dec/timetongbuexam"

# 要处理的轮次
RUN_IDS = [str(i) for i in range(1, 11)]  # 1~10 轮

# 文件与列名
PRESSURE_FILE = "pressure_series.csv"
PRESS_TIME_COL = "host_ms"
PRESS_VAL_COL = "val"

VISION_FILE = "hand_depth_plane_avg.csv"
VISION_TIME_COL = "timestamp_ms"
VISION_VAL_COL = "depth_corr_mm"

# 峰检测参数（可与论文中写的保持一致）
PRESS_PERCENTILE = 60     # 压力阈值：P60
VISION_PERCENTILE = 50    # 视觉阈值：P50
MIN_INTERVAL_MS = 150     # 峰最小间隔，防止一波多峰
PROMINENCE_RATIO = 0.10   # 视觉端 prominence = range * 0.10

# 互相关参数（仅在没有手工 offset 的 run 上作为 fallback）
RESAMPLE_DT_MS = 10       # 重采样步长（ms）
XCORR_MAX_LAG_MS = 800    # 互相关允许的最大时移（±0.8 s）

# 初次配对使用的“宽松窗口”（相对于 offset）
INIT_TOL_MS = 800         # 例如 ±800ms

MAD_SCALE = 1.4826
K_TOL = 3.0               # tol = K_TOL * robust_std

# 手工标定好的每轮 offset（pressure 相对 vision，单位 ms）
MANUAL_OFFSETS = {
    "1": 200.0,
    "2": 180.0,
    "3": 190.0,
    "4": 250.0,
    "5": 260.0,
    "6": 170.0,
    "7": 280.0,
    "8": 230.0,
    "9": 290.0,
    "10": 170.0,
}


# ================== 工具函数 ==================

def detect_peaks(time_ms, values, percentile=60, min_interval_ms=150,
                 use_prominence=False, prominence_ratio=0.1):
    """
    峰检测：
    - 使用百分位阈值过滤
    - 用 scipy.signal.find_peaks 查峰
    - 返回：峰时间数组、峰索引、properties
    """
    time_ms = np.asarray(time_ms)
    values = np.asarray(values)

    thr = np.percentile(values, percentile)

    if len(time_ms) < 2:
        raise ValueError("时间点太少，无法估计采样间隔")

    dt = np.median(np.diff(time_ms))
    if dt <= 0:
        raise ValueError("时间轴不单调，请先排序或检查数据")

    min_distance = int(np.round(min_interval_ms / dt))
    min_distance = max(min_distance, 1)

    kwargs = {}
    if use_prominence:
        vmin, vmax = float(np.min(values)), float(np.max(values))
        prom = (vmax - vmin) * prominence_ratio
        kwargs["prominence"] = prom

    peaks, properties = find_peaks(values, height=thr, distance=min_distance, **kwargs)
    peak_times = time_ms[peaks]
    return peak_times, peaks, properties


def estimate_offset_by_xcorr(press_t, press_v, vis_t, vis_v,
                             dt_ms=10, max_lag_ms=800):
    """
    使用互相关估计压力 vs 视觉的整体时间偏移（pressure 相对 vision 的延迟）。
    仅在没有手工 offset 的 run 上作为后备方案。
    """
    press_t = np.asarray(press_t)
    press_v = np.asarray(press_v)
    vis_t = np.asarray(vis_t)
    vis_v = np.asarray(vis_v)

    # 取两路时间轴的交集区间
    t_min = max(press_t[0], vis_t[0])
    t_max = min(press_t[-1], vis_t[-1])
    if t_max - t_min < 2 * max_lag_ms:
        # 交集太短，互相关不稳定，返回 0
        return 0.0

    t_uniform = np.arange(t_min, t_max, dt_ms)
    p_uniform = np.interp(t_uniform, press_t, press_v)
    v_uniform = np.interp(t_uniform, vis_t, vis_v)

    # 去均值，避免 DC 影响
    p_uniform -= np.mean(p_uniform)
    v_uniform -= np.mean(v_uniform)

    corr = correlate(p_uniform, v_uniform, mode="full")
    lags = np.arange(-len(v_uniform) + 1, len(p_uniform))

    # 限制最大时移
    max_lag_steps = int(max_lag_ms / dt_ms)
    center = len(corr) // 2
    idx_min = max(center - max_lag_steps, 0)
    idx_max = min(center + max_lag_steps + 1, len(corr))

    corr_window = corr[idx_min:idx_max]
    lags_window = lags[idx_min:idx_max]

    best_lag_steps = lags_window[np.argmax(corr_window)]
    # lag>0 表示 pressure 相对 vision 需要往右移（晚），单位 ms
    offset_ms = best_lag_steps * dt_ms
    return float(offset_ms)


def mad(x):
    """Median Absolute Deviation"""
    x = np.asarray(x)
    med = np.median(x)
    return np.median(np.abs(x - med))


def one_to_one_nearest_pairing(press_times, vis_times, offset_ms=0.0, tol_ms=800.0):
    """
    一对一最近邻配对：
    - 对每个视觉峰，在“应用 offset 后”的压力峰中找最近的一个
    - 要求 |(t_press - t_vis - offset_ms)| <= tol_ms
    - 每个压力峰最多被匹配一次
    返回:
      dts: Δt = t_press - t_vis（原始时间差）
      pairs: (t_press, t_vis)
    """
    press_times = np.asarray(press_times)
    vis_times = np.asarray(vis_times)

    # 应用 offset 后再比较
    press_shifted = press_times - offset_ms

    used_press = np.zeros(len(press_times), dtype=bool)
    dts = []
    pairs = []

    j = 0  # pressure 索引

    for t_v in vis_times:
        # 从上次位置开始，跳过明显过早的 pressure 峰
        while j < len(press_shifted) and press_shifted[j] < t_v - tol_ms:
            j += 1

        # 在 [t_v - tol_ms, t_v + tol_ms] 范围内搜最近的未使用 pressure 峰
        best_idx = -1
        best_abs_diff = None
        k = j
        while k < len(press_shifted) and press_shifted[k] <= t_v + tol_ms:
            if not used_press[k]:
                diff = press_shifted[k] - t_v
                ad = abs(diff)
                if (best_abs_diff is None) or (ad < best_abs_diff):
                    best_abs_diff = ad
                    best_idx = k
            k += 1

        if best_idx >= 0 and best_abs_diff is not None and best_abs_diff <= tol_ms:
            used_press[best_idx] = True
            t_p = press_times[best_idx]
            dt = t_p - t_v
            dts.append(dt)
            pairs.append((t_p, t_v))

    return np.asarray(dts), np.asarray(pairs)


def estimate_tolerance_from_dts(dts, k=K_TOL):
    """
    在清洗后的 Δt 上估计容忍区间：
    - 先用 MAD 剔除远离中位数的异常点
    - 再对 “中心化的 |dt - median(dt)|” 求 MAD → robust_std
    - tol = k * robust_std
    """
    dts = np.asarray(dts)
    if len(dts) == 0:
        raise ValueError("没有任何配对得到 Δt，无法估计容忍区间")

    # 先以 median 为中心做一次 outlier 剔除
    med = np.median(dts)
    mad_val = mad(dts)
    if mad_val > 0:
        thresh = 3.0 * MAD_SCALE * mad_val
        clean = dts[np.abs(dts - med) <= thresh]
        if len(clean) < 3:
            clean = dts.copy()
    else:
        clean = dts.copy()

    # 再根据 “中心化后的绝对偏差” 估计 robust_std
    center = np.median(clean)
    abs_centered = np.abs(clean - center)
    mad_abs = mad(abs_centered)
    robust_std = MAD_SCALE * mad_abs
    tol_ms = k * robust_std

    stats = {
        "n_pairs": int(len(clean)),
        "mean_dt": float(np.mean(clean)),
        "median_dt": float(np.median(clean)),
        "std_dt": float(np.std(clean, ddof=1)),
        "mad_abs_centered": float(mad_abs),
        "robust_std": float(robust_std),
        "recommended_tol_ms": float(tol_ms),
    }
    return tol_ms, stats, clean


def plot_global_dt_boxplot(all_dts, out_dir):
    """
    画一个全局 Δt 的箱线图（SVG 矢量），类似你发的那张图。
    all_dts: list 或 ndarray，包含所有 session 的 Δt（可以用 cleaned Δt 拼起来）
    """
    all_dts = np.asarray(all_dts)
    if all_dts.size == 0:
        print("[WARN] No dt data for boxplot, skip.")
        return

    plt.figure(figsize=(6, 3.5))
    # 这里传入一个列表 [all_dts]，画单个箱子
    plt.boxplot([all_dts], vert=True, showfliers=True)
    plt.ylabel("Δt (ms)")
    plt.xlabel("All sessions")
    plt.title("Time Difference Distribution (Boxplot)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "dt_boxplot_all.svg")
    plt.savefig(out_path, format="svg")  # ✅ SVG 矢量格式
    plt.close()
    print(f"[SAVE] Global Δt boxplot saved to: {out_path}")


# ================== 主流程 ==================

def main():
    summary_rows = []
    all_clean_dts = []   # ✅ 新增：收集所有轮次的 Δt，用于画全局箱线图

    for run_id in RUN_IDS:
        run_dir = os.path.join(BASE_DIR, run_id)
        press_path = os.path.join(run_dir, PRESSURE_FILE)
        vision_path = os.path.join(run_dir, VISION_FILE)

        print(f"\n=== Run {run_id} ===")
        df_p = pd.read_csv(press_path).sort_values(by=PRESS_TIME_COL)
        df_v = pd.read_csv(vision_path).sort_values(by=VISION_TIME_COL)

        press_t = df_p[PRESS_TIME_COL].values
        press_v = df_p[PRESS_VAL_COL].values
        vis_t = df_v[VISION_TIME_COL].values
        vis_v = df_v[VISION_VAL_COL].values

        # 1) 决定使用哪种 offset：优先手工表，没有则 fallback 互相关
        if run_id in MANUAL_OFFSETS:
            offset_ms = MANUAL_OFFSETS[run_id]
            print(f"[INFO] Use MANUAL offset for run {run_id}: {offset_ms:.1f} ms")
        else:
            offset_ms = estimate_offset_by_xcorr(
                press_t, press_v, vis_t, vis_v,
                dt_ms=RESAMPLE_DT_MS,
                max_lag_ms=XCORR_MAX_LAG_MS
            )
            print(f"[INFO] Estimated offset (press relative to vision): {offset_ms:.1f} ms")

        # 2) 峰检测
        press_peaks_t, _, _ = detect_peaks(
            press_t, press_v,
            percentile=PRESS_PERCENTILE,
            min_interval_ms=MIN_INTERVAL_MS,
            use_prominence=False
        )
        vis_peaks_t, _, _ = detect_peaks(
            vis_t, vis_v,
            percentile=VISION_PERCENTILE,
            min_interval_ms=MIN_INTERVAL_MS,
            use_prominence=True,
            prominence_ratio=PROMINENCE_RATIO
        )
        print(f"[INFO] Detected {len(press_peaks_t)} pressure peaks, {len(vis_peaks_t)} vision peaks.")

        # 3) 一对一最近邻配对（在 offset 附近 ±INIT_TOL_MS）
        dts, pairs = one_to_one_nearest_pairing(
            press_peaks_t, vis_peaks_t,
            offset_ms=offset_ms,
            tol_ms=INIT_TOL_MS
        )
        print(f"[INFO] One-to-one pairing (±{INIT_TOL_MS} ms around offset) gives {len(dts)} pairs.")

        # 4) 在这些 Δt 上做 robust 容忍区间估计
        tol_ms, stats, clean_dts = estimate_tolerance_from_dts(dts, k=K_TOL)

        print("[STATS] Δt = t_press - t_vis (after pairing)")
        for k_name, v in stats.items():
            print(f"  {k_name}: {v}")

        print(f"[RESULT] Data-driven recommended tolerance for Run {run_id}: ~ {tol_ms:.1f} ms")

        # 收集清洗后的 Δt，用于全局箱线图
        all_clean_dts.append(clean_dts)

        # 保存配对明细
        if len(pairs) > 0:
            out_pairs = pd.DataFrame({
                "t_press_ms": pairs[:, 0],
                "t_vis_ms": pairs[:, 1],
                "dt_ms": dts
            })
            out_csv = os.path.join(run_dir, f"pairing_one2one_run{run_id}.csv")
            out_pairs.to_csv(out_csv, index=False)
            print(f"[SAVE] Pairing details saved to: {out_csv}")
        else:
            print("[WARN] No valid pairs for this run, skip saving pair file.")

        # 保存汇总信息
        summary_rows.append({
            "run": run_id,
            "n_press_peaks": len(press_peaks_t),
            "n_vis_peaks": len(vis_peaks_t),
            "n_pairs_clean": stats["n_pairs"],
            "mean_dt": stats["mean_dt"],
            "median_dt": stats["median_dt"],
            "std_dt": stats["std_dt"],
            "robust_std": stats["robust_std"],
            "recommended_tol_ms": stats["recommended_tol_ms"],
            "offset_ms": offset_ms
        })

    # 保存每轮的汇总表
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(BASE_DIR, "tolerance_summary_one2one.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY] Saved per-run tolerance summary to: {summary_path}")

    # ====== 全局 Δt 箱线图（SVG） ======
    if all_clean_dts:
        all_clean_dts = np.concatenate(all_clean_dts)
        plot_global_dt_boxplot(all_clean_dts, BASE_DIR)


if __name__ == "__main__":
    main()
