import pandas as pd
import numpy as np
from bisect import bisect_left

CSV_PATH = "hand_depth_plane_avg.csv"
SIG_COL  = "depth_kf_mm"   # 你用于峰值检测的信号列（按需改：depth_corr_mm 等）

# 可选：过滤峰/谷的合理范围（防止跑飞点）
PEAK_MIN, PEAK_MAX = 8.0, 80.0        # 正峰阈值范围
TROUGH_MIN, TROUGH_MAX = -20.0, 30.0  # 负峰(谷)阈值范围，按你的系统实际调

df = pd.read_csv(CSV_PATH)
y = df[SIG_COL].to_numpy()

# 1) 局部正峰（max）
peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
# 2) 局部负峰/波谷（min）
troughs = np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1

# 3) 合理性过滤（可按需要关闭）
peaks = [i for i in peaks if PEAK_MIN < y[i] < PEAK_MAX]
troughs = [i for i in troughs if TROUGH_MIN < y[i] < TROUGH_MAX]
troughs = sorted(troughs)

rows = []
for p in peaks:
    pos = bisect_left(troughs, p)
    if pos == 0 or pos >= len(troughs):
        continue

    prev_tr = troughs[pos - 1]
    next_tr = troughs[pos]

    # 峰-谷幅值差（你要的“正负峰值间隔”）
    amp_prev = float(y[p] - y[prev_tr])
    amp_next = float(y[p] - y[next_tr])
    amp_mean = 0.5 * (amp_prev + amp_next)

    rows.append({
        "peak_idx": int(p),
        "peak_val": float(y[p]),
        "prev_trough_idx": int(prev_tr),
        "prev_trough_val": float(y[prev_tr]),
        "amp_prev": amp_prev,
        "next_trough_idx": int(next_tr),
        "next_trough_val": float(y[next_tr]),
        "amp_next": amp_next,
        "amp_mean": amp_mean,   # 推荐作为单次按压幅值
    })

res = pd.DataFrame(rows)

print("\n=== 正负峰值“幅值间隔”(peak-trough) 统计 ===")
print(res[["amp_prev", "amp_next", "amp_mean"]].describe())

print("\n=== 前 10 条明细 ===")
print(res.head(10))
