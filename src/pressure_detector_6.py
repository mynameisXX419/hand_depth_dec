# pressure_detector_6.py
# ------------------------------------------------------------
# MCU -> PC 串口 16x16 压力阵列
#   - 滑动窗口均值滤波（MA）
#   - FSM（单阈值 + 回弹确认）
#   - 一次按压 = 一个峰
#   - 峰后下降触发确认（鲁棒）
#   - 支持 baseline CROC + fallback（不会卡死）
#   - ✅ 新增：启动阶段 baseline 初始化（只改这一点，其它逻辑不变）
# ------------------------------------------------------------

import serial
import struct
import threading
import time
import csv
from typing import Optional
from collections import deque

# ================= 串口协议 =================
HEADER = 0xAABB
PAYLOAD_LEN = 512
PACKET_SIZE = 2 + 2 + PAYLOAD_LEN + 2

SERIAL_DEV = "/dev/ttyUSB0"
BAUDRATE = 2000000
TIMEOUT_S = 1.0

# ================= 核心参数 =================
MA_WINDOW = 5

PRESS_ON = 100000
MIN_PRESS_DURATION_MS = 250
MIN_INTER_PRESS_MS = 450

# fallback（一定能跑）
RELEASE_RATIO = 0.95

# baseline + CROC
BASELINE_WINDOW = 10
BASELINE_UPDATE_RATIO = 0.50
RECOIL_RATIO = 1.05  #基线回弹比例
RECOIL_HOLD_MS = 120

# ✅ 新增：启动阶段 baseline 采集时长（秒）
BASELINE_INIT_SEC = 3.0

# ================= 全局数据 =================
_samples = []
_peaks = []

_global_idx = 0
_lock = threading.Lock()
_running = False

_ma_buf = deque(maxlen=MA_WINDOW)

# FSM
STATE_IDLE = 0
STATE_PRESSING = 1
_state = STATE_IDLE

_press_start_t = None
_last_peak_t = None

# 峰值跟踪
_peak_max_val_raw = None
_peak_max_val_filt = None
_peak_max_idx = None
_peak_max_t = None

# baseline
_baseline_buf = deque(maxlen=BASELINE_WINDOW)
_baseline_press = None

# ✅ 新增：baseline 初始化状态
_baseline_init_done = False
_start_time_ms = None

# recoil tracking
_recoil_start_ms = None
_recoil_ok = False
_release_min_val_filt = None

# ================= 工具函数 =================
def update_baseline(val_filt: float) -> float:
    _baseline_buf.append(val_filt)
    return sum(_baseline_buf) / len(_baseline_buf)

def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF

def _read_exact(ser, length) -> Optional[bytes]:
    buf = b""
    while len(buf) < length and _running:
        chunk = ser.read(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf if len(buf) == length else None

def _find_header(ser) -> bool:
    while _running:
        if ser.read(1) == b"\xBB" and ser.read(1) == b"\xAA":
            return True
    return False

# ================= RX 线程 =================
def _rx_thread():
    global _global_idx, _state
    global _press_start_t, _last_peak_t
    global _peak_max_val_raw, _peak_max_val_filt
    global _peak_max_idx, _peak_max_t
    global _baseline_press, _recoil_start_ms, _recoil_ok
    global _release_min_val_filt
    global _baseline_init_done, _start_time_ms

    print(f"[pressure] open {SERIAL_DEV} @ {BAUDRATE}")
    ser = serial.Serial(SERIAL_DEV, BAUDRATE, timeout=TIMEOUT_S)

    _start_time_ms = int(time.time() * 1000)

    try:
        while _running:
            if not _find_header(ser):
                break

            recv_ms = int(time.time() * 1000)
            rest = _read_exact(ser, PACKET_SIZE - 2)
            if rest is None:
                continue

            packet = b"\xBB\xAA" + rest
            header, length = struct.unpack("<HH", packet[:4])
            if header != HEADER or length != PAYLOAD_LEN:
                continue

            payload = packet[4:4 + PAYLOAD_LEN]
            crc_recv = struct.unpack("<H", packet[-2:])[0]
            if _crc16(packet[:-2]) != crc_recv:
                continue

            vals = struct.unpack("<256H", payload)
            frame_256 = list(vals)  
            val_raw = int(sum(vals))
            _ma_buf.append(val_raw)
            val_filt = sum(_ma_buf) / len(_ma_buf)

            with _lock:
                _samples.append({
                    "idx": _global_idx,
                    "host_ms": recv_ms,
                    "val_raw": val_raw,
                    "val_filt": val_filt,
                })

                # =========================================================
                # ✅ 新增：启动阶段 baseline 初始化（只新增这一段）
                #   - 启动后 BASELINE_INIT_SEC 秒内，不进入 FSM，不检测峰
                #   - 用 val_filt 填充 baseline_buf
                # =========================================================
                if not _baseline_init_done:
                    if (recv_ms - _start_time_ms) <= int(BASELINE_INIT_SEC * 1000):
                        _baseline_press = update_baseline(val_filt)
                        _global_idx += 1
                        continue
                    else:
                        _baseline_init_done = True
                        # baseline 可能仍为 None（极端情况），这里安全打印
                        if _baseline_press is not None:
                            print(f"[pressure] baseline initialized: {_baseline_press:.0f}")
                        else:
                            print("[pressure] baseline initialized: NA")
                # =========================================================

                # ================= FSM =================
                if _state == STATE_IDLE:
                    # baseline 更新（保留你原逻辑，不删不改）
                    if val_filt < PRESS_ON * BASELINE_UPDATE_RATIO:
                        _baseline_press = update_baseline(val_filt)

                    # 进入按压
                    if val_filt >= PRESS_ON:
                        if (_last_peak_t is None) or (recv_ms - _last_peak_t >= MIN_INTER_PRESS_MS):
                            _state = STATE_PRESSING
                            _press_start_t = recv_ms

                            _peak_max_val_raw = val_raw
                            _peak_max_val_filt = val_filt
                            _peak_max_idx = _global_idx
                            _peak_max_t = recv_ms

                            _recoil_start_ms = None
                            _recoil_ok = False
                            _release_min_val_filt = val_filt

                elif _state == STATE_PRESSING:
                    # 记录释放阶段最低压力
                    _release_min_val_filt = min(_release_min_val_filt, val_filt)

                    # 峰值更新
                    if val_raw > _peak_max_val_raw:
                        _peak_max_val_raw = val_raw
                        _peak_max_val_filt = val_filt
                        _peak_max_idx = _global_idx
                        _peak_max_t = recv_ms

                    if (recv_ms - _press_start_t) >= MIN_PRESS_DURATION_MS:
                        # -------- 回弹判定（双通道）--------
                        baseline_ok = (
                            (_baseline_press is not None) and
                            (val_filt <= _baseline_press * RECOIL_RATIO)
                        )

                        fallback_ok = (
                            val_filt <= RELEASE_RATIO * _peak_max_val_filt
                        )

                        if baseline_ok or fallback_ok:
                            if _recoil_start_ms is None:
                                _recoil_start_ms = recv_ms
                            elif (recv_ms - _recoil_start_ms) >= RECOIL_HOLD_MS:
                                _recoil_ok = True
                        else:
                            _recoil_start_ms = None

                        if _recoil_ok:
                            interval = None
                            bpm = None
                            if _last_peak_t is not None:
                                interval = _peak_max_t - _last_peak_t
                                if interval > 0:
                                    bpm = 60000.0 / interval

                            _peaks.append({
                                "idx": _peak_max_idx,
                                "val": int(_peak_max_val_raw),
                                "t": _peak_max_t,
                                "bpm": bpm,
                                "recoil_ok": _recoil_ok,
                                "recoil_mode": "baseline" if baseline_ok else "fallback",
                                "baseline": _baseline_press,
                                "frame_256": frame_256,          # ⭐ 核心新增
                            })


                            release_min_str = (
                                f"{_release_min_val_filt:.0f}"
                                if _release_min_val_filt is not None else "NA"
                            )

                            baseline_str = (
                                f"{_baseline_press:.0f}"
                                if _baseline_press is not None else "NA"
                            )

                            print(
                                f"[PEAK #{len(_peaks):02d}] "
                                f"idx={_peak_max_idx:5d} "
                                f"val={int(_peak_max_val_raw):6d} "
                                f"{'' if bpm is None else f'bpm={bpm:.1f}'} "
                                f"recoil={'OK' if baseline_ok else 'FB'} "
                                f"(release_min={release_min_str}, baseline={baseline_str})"
                            )

                            _last_peak_t = _peak_max_t
                            _state = STATE_IDLE

                _global_idx += 1

    finally:
        ser.close()
        print("[pressure] rx exit")

# ================= API =================
def init_pressure_detector():
    global _running
    _running = True
    threading.Thread(target=_rx_thread, daemon=True).start()
    print("[pressure] detector started")

def export_series_csv(path="pressure_series.csv"):
    with _lock:
        data = list(_samples)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "host_ms", "val_raw", "val_filt"])
        for s in data:
            w.writerow([s["idx"], s["host_ms"], s["val_raw"], int(s["val_filt"])])

def export_peaks_csv(path="pressure_peaks_test.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "peak_val", "t_ms", "bpm", "recoil_ok", "baseline"])
        for p in _peaks:
            w.writerow([
                p["idx"], p["val"], p["t"],
                p.get("bpm"), p.get("recoil_ok"), p.get("baseline")
            ])
# ================= Peak Fetch API =================
_last_peak_export_idx = 0

def fetch_new_peaks():
    global _last_peak_export_idx

    with _lock:
        if _last_peak_export_idx >= len(_peaks):
            return []

        new_peaks = _peaks[_last_peak_export_idx:]
        start_idx = _last_peak_export_idx
        _last_peak_export_idx = len(_peaks)

    results = []
    for i, p in enumerate(new_peaks):
        results.append({
            "local_idx": start_idx + i,
            "global_idx": p["idx"],
            "press_val": p["val"],
            "t_ms": p["t"],
            "bpm": p.get("bpm"),
            "recoil_ok": p.get("recoil_ok"),
            "recoil_mode": p.get("recoil_mode"),
            "baseline": p.get("baseline"),
            "frame_256": p.get("frame_256"),     # ⭐ 核心
        })
    return results


# ================= main =================
if __name__ == "__main__":
    print("========== pressure_detector test ==========")

    init_pressure_detector()
    time.sleep(20)

    export_series_csv()
    export_peaks_csv()

    print("========== test finished ==========")
    print(f"Detected peaks: {len(_peaks)}")
