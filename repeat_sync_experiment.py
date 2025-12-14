#!/usr/bin/env python3
# repeat_sync_experiment.py
#
# 自动化重复实验工具
# 目的：快速收集多次实验数据，用于时间同步分析的统计推断

import os
import json
import time
import csv
from datetime import datetime
import numpy as np


EXPERIMENTS_DIR = "sync_experiments"
SUMMARY_FILE = "sync_experiments_summary.json"


def create_experiment_dir():
    """创建实验数据目录"""
    if not os.path.exists(EXPERIMENTS_DIR):
        os.makedirs(EXPERIMENTS_DIR)
        print(f"[INFO] Created directory: {EXPERIMENTS_DIR}")


def get_next_experiment_id():
    """获取下一个实验ID"""
    if not os.path.exists(SUMMARY_FILE):
        return 1
    
    with open(SUMMARY_FILE, 'r') as f:
        summary = json.load(f)
    
    return len(summary.get('experiments', [])) + 1


def wait_for_experiment_start():
    """等待用户准备开始实验"""
    exp_id = get_next_experiment_id()
    
    print("\n" + "=" * 60)
    print(f"实验 #{exp_id} 准备")
    print("=" * 60)
    print("\n请确保:")
    print("  1. listener_fusion_3_1.py 正在运行")
    print("  2. hand_dec_3.py 正在运行")
    print("  3. 压力传感器已连接")
    print("  4. 视觉系统工作正常")
    print("\n准备好后，请:")
    print("  - 按Enter开始实验（建议40-50次按压）")
    print("  - 完成按压后，按Ctrl+C停止数据采集")
    print("  - 或输入 'q' 退出\n")
    
    response = input(">>> 按Enter开始实验 (或输入'q'退出): ")
    
    if response.lower() == 'q':
        return False
    
    return True


def backup_current_data(exp_id):
    """备份当前实验数据"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 备份文件
    files_to_backup = [
        "pressure_series.csv",
        "hand_depth_plane_avg.csv"
    ]
    
    backed_up = []
    for fname in files_to_backup:
        if os.path.exists(fname):
            new_name = f"{EXPERIMENTS_DIR}/exp{exp_id:02d}_{fname.replace('.csv', '')}_{timestamp}.csv"
            os.rename(fname, new_name)
            backed_up.append(new_name)
            print(f"[BACKUP] {fname} -> {new_name}")
    
    return backed_up, timestamp


def analyze_experiment(exp_id, pressure_file, vision_file):
    """分析单次实验的时间偏移"""
    from scipy import signal, interpolate
    
    # 加载数据
    def load_pressure(path):
        times, vals = [], []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                try:
                    times.append(int(row[2]))
                    vals.append(int(row[3]))
                except:
                    continue
        return np.array(times, dtype=float), np.array(vals, dtype=float)
    
    def load_vision(path):
        times, depths = [], []
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                try:
                    times.append(int(float(row[1])))
                    depths.append(float(row[5]))
                except:
                    continue
        return np.array(times, dtype=float), np.array(depths, dtype=float)
    
    t_press, v_press = load_pressure(pressure_file)
    t_vis, d_vis = load_vision(vision_file)
    
    # 互相关估计时间偏移
    t_start = max(np.min(t_press), np.min(t_vis))
    t_end = min(np.max(t_press), np.max(t_vis))
    
    if t_end <= t_start:
        return None
    
    dt_ms = 10
    t_common = np.arange(t_start, t_end, dt_ms)
    
    f_press = interpolate.interp1d(t_press, v_press, kind='linear', 
                                   bounds_error=False, fill_value=0)
    f_vis = interpolate.interp1d(t_vis, d_vis, kind='linear', 
                                 bounds_error=False, fill_value=0)
    
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
    best_corr = correlation_limited[best_idx] / len(s_press)
    
    # 峰值检测和配对
    def detect_peaks_simple(times, vals):
        peaks = []
        threshold = np.percentile(vals, 60)
        for i in range(1, len(vals) - 1):
            if vals[i] >= vals[i-1] and vals[i] >= vals[i+1] and vals[i] >= threshold:
                peaks.append((times[i], vals[i]))
        return peaks
    
    press_peaks = detect_peaks_simple(t_press, v_press)
    vis_peaks = detect_peaks_simple(t_vis, d_vis)
    
    # 简单配对
    n_pairs = min(len(press_peaks), len(vis_peaks))
    
    return {
        'exp_id': exp_id,
        'n_pressure_samples': len(t_press),
        'n_vision_samples': len(t_vis),
        'n_pressure_peaks': len(press_peaks),
        'n_vision_peaks': len(vis_peaks),
        'n_paired_peaks': n_pairs,
        'time_offset_ms': float(best_lag_ms),
        'correlation_coef': float(best_corr),
        'duration_s': float((t_end - t_start) / 1000),
    }


def save_experiment_summary(experiments):
    """保存实验汇总"""
    with open(SUMMARY_FILE, 'w') as f:
        json.dump({
            'total_experiments': len(experiments),
            'last_updated': datetime.now().isoformat(),
            'experiments': experiments
        }, f, indent=2)
    
    print(f"\n[SAVE] Summary saved to {SUMMARY_FILE}")


def print_summary_statistics(experiments):
    """打印汇总统计"""
    if not experiments:
        print("\n[WARN] No experiments to analyze")
        return
    
    offsets = [e['time_offset_ms'] for e in experiments]
    corrs = [e['correlation_coef'] for e in experiments]
    pairs = [e['n_paired_peaks'] for e in experiments]
    
    print("\n" + "=" * 60)
    print("实验汇总统计")
    print("=" * 60)
    print(f"\n总实验次数: {len(experiments)}")
    print(f"总配对峰值数: {sum(pairs)}")
    
    print(f"\n时间偏移 (ms):")
    print(f"  均值:     {np.mean(offsets):.1f}")
    print(f"  中位数:   {np.median(offsets):.1f}")
    print(f"  标准差:   {np.std(offsets):.1f}")
    print(f"  范围:     [{np.min(offsets):.1f}, {np.max(offsets):.1f}]")
    
    # 95% 置信区间
    sem = np.std(offsets) / np.sqrt(len(offsets))
    ci_95 = 1.96 * sem
    mean_offset = np.mean(offsets)
    print(f"  95% CI:   [{mean_offset - ci_95:.1f}, {mean_offset + ci_95:.1f}]")
    
    print(f"\n互相关系数:")
    print(f"  均值:     {np.mean(corrs):.3f}")
    print(f"  范围:     [{np.min(corrs):.3f}, {np.max(corrs):.3f}]")
    
    print(f"\n配对峰值数/实验:")
    print(f"  均值:     {np.mean(pairs):.1f}")
    print(f"  范围:     [{np.min(pairs)}, {np.max(pairs)}]")
    
    # 变异系数
    cv = np.std(offsets) / np.mean(offsets) * 100
    print(f"\n变异系数 (CV): {cv:.1f}%")
    
    if cv < 20:
        print("  ✓ 变异性低，时间偏移稳定")
    elif cv < 50:
        print("  ~ 变异性中等，需要自适应补偿")
    else:
        print("  ✗ 变异性高，建议检查系统")
    
    # 统计检验建议
    print("\n" + "=" * 60)
    print("论文撰写建议")
    print("=" * 60)
    
    if len(experiments) >= 5:
        print("✓ 数据量足够用于会议论文/一般期刊")
        print(f"  建议表述: '基于{len(experiments)}次独立实验（n={sum(pairs)}对峰值）'")
    elif len(experiments) >= 3:
        print("~ 数据量基本够用（最小要求）")
        print(f"  建议表述: '基于初步实验（N={len(experiments)}次，n={sum(pairs)}对）'")
    else:
        print("✗ 数据量不足，建议至少3-5次实验")
    
    print(f"\n推荐表述:")
    print(f"\"压力传感器相对于视觉传感器存在{mean_offset:.0f}±{np.std(offsets):.0f}ms")
    print(f" 的平均延迟（95% CI: [{mean_offset - ci_95:.0f}, {mean_offset + ci_95:.0f}]ms，")
    print(f" N={len(experiments)}次实验，n={sum(pairs)}对峰值）。\"")


def generate_latex_table(experiments):
    """生成LaTeX表格"""
    print("\n" + "=" * 60)
    print("LaTeX 表格代码")
    print("=" * 60)
    print("""
\\begin{table}[h]
\\centering
\\caption{Time Synchronization Experiments Summary}
\\label{tab:sync_experiments}
\\begin{tabular}{cccccc}
\\hline
Exp. & Pressure & Vision & Paired & Offset & Corr. \\\\
ID & Peaks & Peaks & Peaks & (ms) & Coef. \\\\
\\hline""")
    
    for exp in experiments:
        print(f"{exp['exp_id']} & "
              f"{exp['n_pressure_peaks']} & "
              f"{exp['n_vision_peaks']} & "
              f"{exp['n_paired_peaks']} & "
              f"{exp['time_offset_ms']:.0f} & "
              f"{exp['correlation_coef']:.3f} \\\\")
    
    offsets = [e['time_offset_ms'] for e in experiments]
    mean_offset = np.mean(offsets)
    std_offset = np.std(offsets)
    
    print("\\hline")
    print(f"\\textbf{{Mean}} & - & - & "
          f"{np.mean([e['n_paired_peaks'] for e in experiments]):.0f} & "
          f"\\textbf{{{mean_offset:.0f}}} & "
          f"{np.mean([e['correlation_coef'] for e in experiments]):.3f} \\\\")
    print(f"\\textbf{{Std}} & - & - & - & "
          f"\\textbf{{{std_offset:.0f}}} & "
          f"{np.std([e['correlation_coef'] for e in experiments]):.3f} \\\\")
    print("""\\hline
\\end{tabular}
\\end{table}
""")


def main():
    """主程序"""
    print("=" * 60)
    print("时间同步重复实验自动化工具")
    print("=" * 60)
    print("\n目标: 收集足够的数据用于时间同步分析的统计推断")
    print("推荐: 至少进行3-5次独立实验")
    
    create_experiment_dir()
    
    # 加载已有实验
    experiments = []
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r') as f:
            data = json.load(f)
            experiments = data.get('experiments', [])
        print(f"\n[INFO] 已有 {len(experiments)} 次实验记录")
    
    while True:
        # 等待用户开始
        if not wait_for_experiment_start():
            break
        
        exp_id = get_next_experiment_id()
        
        print(f"\n[实验 #{exp_id}] 开始数据采集...")
        print("[提示] 请进行40-50次按压，然后按Ctrl+C停止\n")
        
        # 等待用户完成实验
        try:
            input(">>> 数据采集中... 完成后按Enter继续")
        except KeyboardInterrupt:
            print("\n[INFO] 数据采集完成")
        
        # 备份数据
        print(f"\n[实验 #{exp_id}] 备份数据...")
        backed_up, timestamp = backup_current_data(exp_id)
        
        if len(backed_up) < 2:
            print("[ERROR] 数据文件不完整，跳过此次实验")
            continue
        
        # 分析数据
        print(f"\n[实验 #{exp_id}] 分析数据...")
        result = analyze_experiment(exp_id, backed_up[0], backed_up[1])
        
        if result is None:
            print("[ERROR] 分析失败，跳过此次实验")
            continue
        
        result['timestamp'] = timestamp
        result['files'] = backed_up
        
        experiments.append(result)
        
        # 显示当前实验结果
        print(f"\n[实验 #{exp_id}] 结果:")
        print(f"  时间偏移:    {result['time_offset_ms']:.0f} ms")
        print(f"  相关系数:    {result['correlation_coef']:.3f}")
        print(f"  配对峰值:    {result['n_paired_peaks']}")
        print(f"  实验时长:    {result['duration_s']:.1f} s")
        
        # 保存汇总
        save_experiment_summary(experiments)
        
        # 显示累积统计
        if len(experiments) >= 2:
            print_summary_statistics(experiments)
        
        # 询问是否继续
        print("\n" + "-" * 60)
        if len(experiments) >= 5:
            print("✓ 已完成5次实验，数据量充足！")
            response = input(">>> 是否继续添加更多实验? (y/N): ")
            if response.lower() != 'y':
                break
        elif len(experiments) >= 3:
            print("~ 已完成3次实验，基本够用。建议再进行1-2次。")
        else:
            print(f"⚠ 已完成{len(experiments)}次实验，建议至少进行{3-len(experiments)}次以上。")
    
    # 最终汇总
    if experiments:
        print("\n" + "=" * 60)
        print("所有实验完成！")
        print("=" * 60)
        print_summary_statistics(experiments)
        generate_latex_table(experiments)
        
        print(f"\n数据文件保存在: {EXPERIMENTS_DIR}/")
        print(f"汇总文件: {SUMMARY_FILE}")
    else:
        print("\n[INFO] 未完成任何实验")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] 程序被用户中断")
