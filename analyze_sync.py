#!/usr/bin/env python3
# analyze_and_plot_sync.py
#
# 读取 sync_debug.csv：
#   1) 计算 soft-sync 误差统计：err = recv_pc_ms - host_ms
#   2) 拟合 MCU 时钟到 PC 时钟：recv_pc_ms ≈ a + b * mcu_ms
#   3) 画三张图并保存为 EMF + SVG：
#        - 时间误差曲线（err vs. idx）
#        - 误差直方图
#        - 拟合残差曲线（residual vs. idx）
#
# 注：在 Windows + Office/WPS 下，用 EMF 嵌入 Word/PowerPoint 效果最好；
#     如果在 Linux 上不支持 EMF，可以改成 PDF/SVG 输出。

import csv
from statistics import mean, stdev

import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = "sync_debug.csv"  # 如果 CSV 不在当前目录，改这里


def load_sync_csv(path):
    mcu_ms = []
    host_ms = []
    recv_pc_ms = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        for row in reader:
            if len(row) < 4:
                continue
            idx, mcu, host, recv_pc = row[:4]
            try:
                mcu_ms.append(int(mcu))
                host_ms.append(int(host))
                recv_pc_ms.append(int(recv_pc))
            except ValueError:
                continue

    return mcu_ms, host_ms, recv_pc_ms


def basic_error_stats(host_ms, recv_pc_ms):
    """计算 soft sync 误差: err = recv_pc_ms - host_ms"""
    errs = [rp - h for rp, h in zip(recv_pc_ms, host_ms)]
    if not errs:
        print("[ERR] no data for error statistics")
        return None

    n = len(errs)
    err_mean = mean(errs)
    err_min = min(errs)
    err_max = max(errs)
    if n > 1:
        err_std = stdev(errs)
    else:
        err_std = 0.0

    print("=== [1] Soft-sync 误差统计: err = recv_pc_ms - host_ms ===")
    print(f"样本数 n = {n}")
    print(f"均值 mean(err)         = {err_mean:.3f} ms")
    print(f"标准差 std(err)        = {err_std:.3f} ms")
    print(f"最小值 min(err)        = {err_min:.3f} ms")
    print(f"最大值 max(err)        = {err_max:.3f} ms")
    print()

    return np.array(errs, dtype=float)


def fit_mcu_to_pc_clock(mcu_ms, recv_pc_ms):
    """
    拟合关系: recv_pc_ms ≈ a + b * mcu_ms

    返回:
      a, b, residuals(np.array), res_std, drift_ppm, drift_per_sec_ms, drift_per_min_ms
    """
    if len(mcu_ms) < 2:
        print("[ERR] not enough data for linear fit")
        return None

    x = np.array(mcu_ms, dtype=float)
    y = np.array(recv_pc_ms, dtype=float)

    # 用 numpy 拟合一条直线
    b, a = np.polyfit(x, y, 1)  # y ≈ a + b*x
    # 计算拟合残差
    y_pred = a + b * x
    residuals = y - y_pred
    res_std = float(np.std(residuals))

    print("=== [2] MCU -> PC 时钟线性拟合: recv_pc_ms ≈ a + b * mcu_ms ===")
    print(f"a (截距)            = {a:.3f} ms")
    print(f"b (斜率, 频率比)    = {b:.9f}")
    print(f"残差标准差 std(res) = {res_std:.3f} ms")

    # 解释一下 b 的意义
    drift_ppm = (b - 1.0) * 1_000_000  # 百万分之一
    print()
    print("--- 频率偏差估计 ---")
    print(f"相对频率偏差 ≈ {drift_ppm:.1f} ppm")
    print("  (正数: MCU 走得比 PC 慢; 负数: MCU 走得比 PC 快)")

    # 估算每秒累计的时间误差
    one_sec_pc_ms = 1000.0 * b
    drift_per_sec_ms = one_sec_pc_ms - 1000.0
    print(f"每秒累计时间误差 ≈ {drift_per_sec_ms:.3f} ms/s")

    # 估算 1 分钟内可能积累的误差
    drift_per_min_ms = drift_per_sec_ms * 60.0
    print(f"一分钟内可能累计误差 ≈ {drift_per_min_ms:.3f} ms/min")
    print()

    return a, b, residuals, res_std, drift_ppm, drift_per_sec_ms, drift_per_min_ms


def save_figure_emf_and_svg(fig, base_name: str):
    """同时保存为 EMF 和 SVG，方便论文排版"""
    emf_path = f"{base_name}.emf"
    svg_path = f"{base_name}.svg"

    # 有的环境可能不支持 EMF，这里 try 一下
    try:
        fig.savefig(emf_path, format="emf")
        print(f"[FIG] saved EMF: {emf_path}")
    except Exception as e:
        print(f"[WARN] save EMF failed ({e}), 请在 Windows 环境下再导出 EMF")

    # 兜底保存一份 SVG（通用矢量格式）
    fig.savefig(svg_path, format="svg")
    print(f"[FIG] saved SVG: {svg_path}")


def main():
    print(f"[INFO] loading sync data from '{CSV_PATH}' ...")
    mcu_ms, host_ms, recv_pc_ms = load_sync_csv(CSV_PATH)

    if not mcu_ms:
        print("[ERR] no valid rows in CSV, please check path and content.")
        return

    print(f"[INFO] loaded {len(mcu_ms)} samples.")

    # 1) 误差统计
    errs = basic_error_stats(host_ms, recv_pc_ms)

    # 2) 时钟线性拟合
    fit_res = fit_mcu_to_pc_clock(mcu_ms, recv_pc_ms)
    if fit_res is None:
        return
    a, b, residuals, res_std, drift_ppm, drift_per_sec_ms, drift_per_min_ms = fit_res

    # === 画图部分 ===
    # 图一：err 随 index 的变化
    fig1 = plt.figure()
    plt.plot(errs, linewidth=0.8)
    plt.title("Soft-sync Time Error (recv_pc_ms - host_ms)")
    plt.xlabel("Sample Index")
    plt.ylabel("Error (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_figure_emf_and_svg(fig1, "sync_error_curve")

    # 图二：err 直方图
    fig2 = plt.figure()
    plt.hist(errs, bins=40, edgecolor="black")
    plt.title("Soft-sync Time Error Distribution")
    plt.xlabel("Error (ms)")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_figure_emf_and_svg(fig2, "sync_error_hist")

    # 图三：线性拟合残差 vs index（反映抖动）
    fig3 = plt.figure()
    plt.plot(residuals, linewidth=0.8)
    plt.title("Fit Residuals: recv_pc_ms - (a + b * mcu_ms)")
    plt.xlabel("Sample Index")
    plt.ylabel("Residual (ms)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_figure_emf_and_svg(fig3, "sync_fit_residuals")

    print("[INFO] All figures saved.")
    # 如需交互查看，可最后加：
    # plt.show()


if __name__ == "__main__":
    main()
