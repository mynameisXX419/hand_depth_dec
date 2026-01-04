import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

ROOT = 'timetongbuexam'
MODES = ['auto', '350', '450', '500']

# collect data
data = {}
for run in sorted(os.listdir(ROOT), key=lambda x: int(x) if x.isdigit() else x):
    run_dir = os.path.join(ROOT, run, 'summary_outputs')
    if not os.path.isdir(run_dir):
        continue
    data[run] = {}
    for m in MODES:
        fname = os.path.join(run_dir, f'summary_{m}.json') if m != 'auto' else os.path.join(run_dir, 'summary_auto.json')
        if os.path.exists(fname):
            try:
                with open(fname, 'r') as f:
                    data[run][m] = json.load(f)
            except Exception as e:
                data[run][m] = {'error': str(e)}
        else:
            data[run][m] = None

# write CSV-like markdown table
md_lines = []
md_lines.append('# 10-Round Aggregated Summary\n')
md_lines.append('| Run | Mode | n_pairs | mean_dt(ms) | median_dt(ms) | std_dt(ms) | offset(ms) |')
md_lines.append('|---|---:|---:|---:|---:|---:|---:|')

for run in sorted(data.keys(), key=lambda x: int(x)):
    for m in MODES:
        rec = data[run].get(m)
        if rec is None:
            md_lines.append(f'| {run} | {m} | - | - | - | - | - |')
        else:
            md_lines.append(f"| {run} | {m} | {rec.get('n_pairs', '-') } | {rec.get('mean_dt_ms','-'):.1f} | {rec.get('median_dt_ms','-'):.1f} | {rec.get('std_dt_ms','-'):.1f} | {rec.get('offset_ms','-'):.1f} |")

# overall totals per mode
md_lines.append('\n## Overall totals by mode\n')
md_lines.append('| Mode | Total pairs | Mean of means (ms) | Median of medians (ms) | Mean std (ms) |')
md_lines.append('|---|---:|---:|---:|---:|')
for m in MODES:
    ns = []
    means = []
    meds = []
    stds = []
    for run in data:
        rec = data[run].get(m)
        if rec:
            ns.append(int(rec.get('n_pairs',0)))
            means.append(float(rec.get('mean_dt_ms',0)))
            meds.append(float(rec.get('median_dt_ms',0)))
            stds.append(float(rec.get('std_dt_ms',0)))
    if len(ns)==0:
        md_lines.append(f'| {m} | - | - | - | - |')
    else:
        md_lines.append(f'| {m} | {sum(ns)} | {np.mean(means):.1f} | {np.median(meds):.1f} | {np.mean(stds):.1f} |')

md_path = '10_ROUND_FINAL_SUMMARY.md'
with open(md_path, 'w') as f:
    f.write('\n'.join(md_lines))
print('Wrote', md_path)

# Quick plots: pairs per run per mode
runs = sorted([r for r in data.keys()], key=lambda x: int(x))
fig, ax = plt.subplots(figsize=(10,6))
indices = np.arange(len(runs))
width = 0.18
for i,m in enumerate(MODES):
    vals = []
    for r in runs:
        rec = data[r].get(m)
        vals.append(rec.get('n_pairs',0) if rec else 0)
    ax.bar(indices + (i-1.5)*width, vals, width, label=m)
ax.set_xticks(indices)
ax.set_xticklabels(runs)
ax.set_xlabel('Run')
ax.set_ylabel('Paired peaks')
ax.set_title('Paired peaks per run and mode')
ax.legend()
plt.tight_layout()
plt.savefig('pairs_by_mode.png')
print('Wrote pairs_by_mode.png')

# SD trend
fig, ax = plt.subplots(figsize=(10,6))
for m in MODES:
    vals = []
    for r in runs:
        rec = data[r].get(m)
        vals.append(rec.get('std_dt_ms',np.nan) if rec else np.nan)
    ax.plot(runs, vals, marker='o', label=m)
ax.set_xlabel('Run')
ax.set_ylabel('Std(Δt) ms')
ax.set_title('Std(Δt) per run by mode')
ax.legend()
plt.tight_layout()
plt.savefig('std_trend_by_mode.png')
print('Wrote std_trend_by_mode.png')
