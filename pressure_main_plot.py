import time
import threading
import matplotlib.pyplot as plt
from collections import deque

from pressure_detector_4 import (
    init_pressure_detector,
    detect_pressure_peaks,
    _samples,          # 只读使用
    _lock
)

# ================== 参数 ==================
FPS_EST = 34.0
WINDOW_SEC = 10                 # 显示最近 10 秒
MAX_POINTS = int(FPS_EST * WINDOW_SEC)

PEAK_THRESHOLD = 5000           # 按你的系统经验
MIN_PEAK_DIST_MS = 180          # 34Hz -> ~30ms/帧，6帧≈180ms

# ================== 缓存 ==================
x_time = deque(maxlen=MAX_POINTS)
y_val  = deque(maxlen=MAX_POINTS)

peak_x = deque(maxlen=100)
peak_y = deque(maxlen=100)


def update_data():
    """后台线程：拉取 samples"""
    last_idx = -1
    while True:
        time.sleep(0.03)  # ~30 ms
        with _lock:
            if not _samples:
                continue
            s = _samples[-1]
            if s["idx"] == last_idx:
                continue
            last_idx = s["idx"]
            x_time.append(s["host_ms"])
            y_val.append(s["val"])


def update_peaks():
    """后台线程：检测峰值"""
    while True:
        time.sleep(0.05)
        peaks = detect_pressure_peaks(
            threshold=PEAK_THRESHOLD,
            min_distance_ms=MIN_PEAK_DIST_MS,
            smooth_window=3,
        )
        for _, _, val, t_ms in peaks:
            peak_x.append(t_ms)
            peak_y.append(val)
            print(f"[PEAK] t={t_ms:.0f} ms, val={val}")


def main():
    print("[main] start pressure detector")
    init_pressure_detector()

    threading.Thread(target=update_data, daemon=True).start()
    threading.Thread(target=update_peaks, daemon=True).start()

    # ================== Matplotlib ==================
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], label="Pressure Sum")
    sc = ax.scatter([], [], c="r", s=50, label="Peaks")

    ax.set_xlabel("PC Time (ms)")
    ax.set_ylabel("Pressure Scalar (sum)")
    ax.set_title("Pressure Curve @ ~34 Hz")
    ax.legend()
    ax.grid(True)

    while True:
        if len(x_time) < 2:
            time.sleep(0.05)
            continue

        line.set_data(x_time, y_val)
        sc.set_offsets(list(zip(peak_x, peak_y)))

        ax.set_xlim(x_time[0], x_time[-1])
        ymin = min(y_val)
        ymax = max(y_val)
        ax.set_ylim(ymin * 0.95, ymax * 1.05)

        plt.pause(0.03)


if __name__ == "__main__":
    main()
