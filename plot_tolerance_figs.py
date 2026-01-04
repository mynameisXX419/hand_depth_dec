import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== 基本配置 ==================

BASE_DIR = "/home/ljy/project/hand_dec/timetongbuexam"
RUN_IDS = [str(i) for i in range(1, 11)]  # 1~10 轮

SUMMARY_FILE = os.path.join(BASE_DIR, "tolerance_summary_one2one.csv")


def load_tolerance_summary():
    """读取每轮的 recommended_tol_ms 等信息"""
    if not os.path.exists(SUMMARY_FILE):
        raise FileNotFoundError(f"Summary file not found: {SUMMARY_FILE}")
    df = pd.read_csv(SUMMARY_FILE)
    # 确保 run 为字符串，便于索引
    df["run"] = df["run"].astype(str)
    return df.set_index("run")


# ======================================================================
# 1) 单轮：Δt vs index，可视化「峰值匹配情况 + 最优容忍带」
# ======================================================================

def plot_dt_vs_index_with_tol(run_id, tol_ms, out_dir):
    """
    对单个 session 画 Δt vs pair index：
      - 蓝色点：在容忍带内的配对
      - 橙色点：被容忍带排除的 outlier
      - 中位数，median ± tol 用水平线表示
    """
    csv_path = os.path.join(out_dir, run_id, f"pairing_one2one_run{run_id}.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] Pairing file not found for run {run_id}: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "dt_ms" not in df.columns:
        print(f"[WARN] dt_ms not found in {csv_path}")
        return

    dts = df["dt_ms"].values
    idx = np.arange(1, len(dts) + 1)
    med = np.median(dts)

    # 以 median 为中心的最优时间容忍带
    mask_in = np.abs(dts - med) <= tol_ms
    dts_in = dts[mask_in]
    idx_in = idx[mask_in]
    dts_out = dts[~mask_in]
    idx_out = idx[~mask_in]

    plt.figure(figsize=(6, 4))
    # 容忍带内
    plt.scatter(idx_in, dts_in, s=18, label="Within tolerance")
    # 容忍带外
    if len(dts_out) > 0:
        plt.scatter(idx_out, dts_out, s=18, marker="x", label="Outside tolerance")

    # 中位数与容忍带
    plt.axhline(med, linestyle="--", linewidth=1.2, label=f"Median = {med:.1f} ms")
    plt.axhline(med + tol_ms, linestyle=":", linewidth=1.0,
                label=f"Median ± tol ({tol_ms:.0f} ms)")
    plt.axhline(med - tol_ms, linestyle=":", linewidth=1.0)

    plt.xlabel("Pair index")
    plt.ylabel("Δt = t_press - t_vis (ms)")
    plt.title(f"Timing difference per pair with tolerance (Run {run_id})")
    plt.tight_layout()

    out_path = os.path.join(out_dir, run_id, f"dt_vs_index_tol_run{run_id}.svg")
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"[SAVE] Δt vs index with tolerance for run {run_id} -> {out_path}")


# ======================================================================
# 2) 单轮：Δt 直方图 + 最优容忍带，展示「阈值前后分布」
# ======================================================================

def plot_dt_hist_with_tol(run_id, tol_ms, out_dir):
    """
    单个 session 的 Δt 直方图：
      - 灰色柱：所有配对
      - 垂直线：median 与 median ± tol
    """
    csv_path = os.path.join(out_dir, run_id, f"pairing_one2one_run{run_id}.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] Pairing file not found for run {run_id}: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "dt_ms" not in df.columns:
        print(f"[WARN] dt_ms not found in {csv_path}")
        return

    dts = df["dt_ms"].values
    med = np.median(dts)

    plt.figure(figsize=(6, 4))
    plt.hist(dts, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(med, linestyle="--", linewidth=1.2, label=f"Median = {med:.1f} ms")
    plt.axvline(med - tol_ms, linestyle=":", linewidth=1.0,
                label=f"Median ± tol ({tol_ms:.0f} ms)")
    plt.axvline(med + tol_ms, linestyle=":", linewidth=1.0)

    plt.xlabel("Δt = t_press - t_vis (ms)")
    plt.ylabel("Count")
    plt.title(f"Histogram of timing differences with tolerance (Run {run_id})")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, run_id, f"dt_hist_tol_run{run_id}.svg")
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"[SAVE] Δt histogram with tolerance for run {run_id} -> {out_path}")


# ======================================================================
# 3) 全局：加入容忍带前后，每轮 std(Δt) 对比（条形图）
# ======================================================================

def compute_std_before_after(summary_df):
    """
    根据 pairing_one2one_runX.csv 和 recommended_tol_ms，计算：
      - 每轮原始 std_all
      - 应用 median±tol 之后 std_in
      - 以及「保留的配对比例」
    返回 DataFrame: columns = [run, std_all, std_in, keep_ratio]
    """
    rows = []
    for run_id in RUN_IDS:
        # tol_ms 从 summary 取
        if run_id not in summary_df.index:
            continue
        tol_ms = summary_df.loc[run_id, "recommended_tol_ms"]

        csv_path = os.path.join(BASE_DIR, run_id, f"pairing_one2one_run{run_id}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] Pairing file not found for run {run_id}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "dt_ms" not in df.columns:
            print(f"[WARN] dt_ms not found in {csv_path}")
            continue

        dts = df["dt_ms"].values
        if len(dts) < 2:
            continue

        std_all = float(np.std(dts, ddof=1))
        med = np.median(dts)
        mask_in = np.abs(dts - med) <= tol_ms
        dts_in = dts[mask_in]
        if len(dts_in) >= 2:
            std_in = float(np.std(dts_in, ddof=1))
        else:
            std_in = np.nan

        keep_ratio = float(len(dts_in) / len(dts))

        rows.append({
            "run": run_id,
            "std_all": std_all,
            "std_in": std_in,
            "keep_ratio": keep_ratio,
            "tol_ms": tol_ms,
        })

    return pd.DataFrame(rows)


def plot_std_before_after(effect_df):
    """
    画每轮「应用最优时间容忍前后 std(Δt) 的对比条形图」，
    直观看出：加入 tolerance 后时间一致性变紧了多少。
    """
    if effect_df.empty:
        print("[WARN] No data for std comparison.")
        return

    effect_df = effect_df.sort_values(by="run")
    runs = effect_df["run"].values
    x = np.arange(len(runs))

    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, effect_df["std_all"].values,
            width=width, label="STD before tolerance")
    plt.bar(x + width/2, effect_df["std_in"].values,
            width=width, label="STD within tolerance")

    plt.xticks(x, runs)
    plt.xlabel("Session (run)")
    plt.ylabel("STD of Δt (ms)")
    plt.title("Effect of data-driven tolerance on timing variability")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, "std_before_after_tol.svg")
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"[SAVE] STD before/after tolerance figure -> {out_path}")


# ======================================================================
# 4) 文本结果分析（终端 + txt 文件）
# ======================================================================

def analyze_and_save_text(summary_df, effect_df):
    """
    汇总一些关键统计量，并输出成自然语言分析，方便论文写作。
    """
    lines = []

    # 1) 时间容忍区间统计
    tol_vals = summary_df["recommended_tol_ms"].values
    tol_mean = float(np.mean(tol_vals))
    tol_min = float(np.min(tol_vals))
    tol_max = float(np.max(tol_vals))

    lines.append("=== Data-driven timing tolerance summary ===")
    lines.append(f"Number of sessions: {len(tol_vals)}")
    lines.append(f"Recommended tolerance: mean = {tol_mean:.1f} ms, "
                 f"min = {tol_min:.1f} ms, max = {tol_max:.1f} ms")

    # 2) STD 前后对比
    if not effect_df.empty:
        std_all_mean = float(np.nanmean(effect_df["std_all"].values))
        std_in_mean = float(np.nanmean(effect_df["std_in"].values))
        reduction_abs = std_all_mean - std_in_mean
        reduction_pct = reduction_abs / std_all_mean * 100.0

        keep_ratio_mean = float(np.nanmean(effect_df["keep_ratio"].values))

        lines.append("")
        lines.append("=== Effect of tolerance on Δt variability ===")
        lines.append(f"Average STD of Δt before tolerance: {std_all_mean:.1f} ms")
        lines.append(f"Average STD of Δt within tolerance: {std_in_mean:.1f} ms")
        lines.append(f"Average STD reduction: {reduction_abs:.1f} ms "
                     f"({reduction_pct:.1f} %)")
        lines.append(f"Average proportion of pairs kept within tolerance: "
                     f"{keep_ratio_mean*100:.1f} %")
    else:
        lines.append("")
        lines.append("No effect_df data, skip STD analysis.")

    text = "\n".join(lines)

    # 打印到终端
    print("\n" + text + "\n")

    # 写入 txt 文件
    out_txt = os.path.join(BASE_DIR, "tolerance_effect_analysis.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[SAVE] Text analysis saved to: {out_txt}")


# ======================================================================
# 主入口
# ======================================================================

def main():
    summary_df = load_tolerance_summary()

    # 1 & 2: 对每轮 session 做「配对情况 + 阈值」可视化
    for run_id in RUN_IDS:
        if run_id not in summary_df.index:
            continue
        tol_ms = summary_df.loc[run_id, "recommended_tol_ms"]
        plot_dt_vs_index_with_tol(run_id, tol_ms, BASE_DIR)
        plot_dt_hist_with_tol(run_id, tol_ms, BASE_DIR)

    # 3: 全局 std(Δt) 在应用 tol 前后的对比
    effect_df = compute_std_before_after(summary_df)
    effect_df.to_csv(os.path.join(BASE_DIR, "tolerance_effect_summary.csv"),
                     index=False)
    print(f"[SAVE] Tolerance effect summary -> "
          f"{os.path.join(BASE_DIR, 'tolerance_effect_summary.csv')}")

    plot_std_before_after(effect_df)

    # 4: 做一段总体文字分析，输出到终端 + txt
    analyze_and_save_text(summary_df, effect_df)


if __name__ == "__main__":
    main()
