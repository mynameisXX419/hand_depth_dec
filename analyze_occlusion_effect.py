import pandas as pd
import numpy as np

base = "."

df_all   = pd.read_csv(f"{base}/occlusion_all_pressure_predict.csv")
df_first = pd.read_csv(f"{base}/occlusion_first_pressure_predict.csv")
df_gt    = pd.read_csv(f"{base}/vision_peaks_gt.csv")

# 1. 和 GT 按压序号对齐（idx 是视觉的“真·第几次按压”）
df_first_merged = df_first.merge(
    df_gt[["idx", "depth_mm"]],
    left_on="cc_index_est",
    right_on="idx",
    how="inner",
    suffixes=("_pred", "_gt")
)

# 2. 计算误差
df_first_merged["err_mm"] = df_first_merged["pred_depth_mm"] - df_first_merged["depth_mm"]

mae  = df_first_merged["err_mm"].abs().mean()
rmse = np.sqrt((df_first_merged["err_mm"]**2).mean())
bias = df_first_merged["err_mm"].mean()

print("===== 首次接管（first predict）整体效果 =====")
print("样本数 n =", len(df_first_merged))
print(f"MAE  = {mae:.2f} mm")
print(f"RMSE = {rmse:.2f} mm")
print(f"Bias = {bias:.2f} mm")

# 3. 按遮挡段看误差
grp = df_first_merged.groupby("occl_seq_id")["err_mm"].agg(["count", "mean"])
print("\n按遮挡段统计：")
print(grp)
