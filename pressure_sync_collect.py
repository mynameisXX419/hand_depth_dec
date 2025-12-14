#!/usr/bin/env python3
# pressure_sync_collect.py
# 用于仅采集 MCU → PC 时间同步数据，不做推理融合

import time
import pressure_detector_3 as pressure_detector

print("=== Time Sync Data Collector ===")
print("Waiting MCU TCP connection...")
pressure_detector.init_pressure_detector()

try:
    while True:
        # 每秒打印当前缓存大小，确认在收数据
        with pressure_detector._lock:
            print(f"Samples: {len(pressure_detector._samples)}")
        time.sleep(1)

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C received, exporting CSV...")
    pressure_detector.export_sync_to_csv("sync_debug1.csv", max_n=10000)
    print("[INFO] Done.")
