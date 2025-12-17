# pressure_detector_4.py
# ------------------------------------------------------------
# MCU -> PC 串口 16x16 压力阵列
#   - 滑动窗口均值滤波（MA）
#   - FSM（单阈值）
#   - 一次按压 = 一个峰
#   - 实时打印峰值 / 间隔 / BPM
# ------------------------------------------------------------

import serial
import struct
import threading
import time
from typing import Optional
from collections import deque

# ================= 串口协议 =================
HEADER = 0xAABB
PAYLOAD_LEN = 512
PACKET_SIZE = 2 + 2 + PAYLOAD_LEN + 2

SERIAL_DEV = "/dev/ttyUSB0"
BAUDRATE = 2000000
TIMEOUT_S = 1.0

# ================= 参数（核心） =================
MA_WINDOW = 5

PRESS_ON = 50000                 # 进入按压阈值
MIN_PRESS_DURATION_MS = 250     # 最短按压
MIN_INTER_PRESS_MS    = 450     # 相邻按压最小间隔

# ================= 全局数据 =================
_samples = []   # {idx, host_ms, val_raw, val_filt}
_peaks   = []   # {idx, val, t_ms}

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
_peak_max_val = None
_peak_max_idx = None
_peak_max_t   = None


# ================= Peak Fetch API =================
_last_peak_export_idx = 0

def fetch_new_peaks():
    """
    返回自上次调用以来的新压力峰
    return: List of (local_idx, global_idx, press_val, t_ms, frame_256)
    """
    global _last_peak_export_idx

    with _lock:
        if _last_peak_export_idx >= len(_peaks):
            return []

        new_peaks = _peaks[_last_peak_export_idx:]
        start_idx = _last_peak_export_idx
        _last_peak_export_idx = len(_peaks)

    results = []
    for i, p in enumerate(new_peaks):
        results.append((
            start_idx + i,          # local_idx（pressure侧序号）
            p["idx"],               # global_idx（采样 idx）
            p["val"],               # 峰值（滤波后）
            p["t"],                 # PC 时间戳 ms
            p.get("frame", None),   # ★ 256 个压力值（list[int]）
        ))

    return results



# ================= CRC16 =================
def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


# ================= 串口工具 =================
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
    global _global_idx
    global _state, _press_start_t, _last_peak_t
    global _peak_max_val, _peak_max_idx, _peak_max_t

    # ★ 新增：缓存最近一帧 256 点压力图
    _last_frame_256 = None

    print(f"[pressure] open {SERIAL_DEV} @ {BAUDRATE}")
    ser = serial.Serial(SERIAL_DEV, BAUDRATE, timeout=TIMEOUT_S)

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

        # ================= 解析 256 点压力阵列 =================
        vals = struct.unpack("<256H", payload)

        # ★ 关键：缓存当前 256 点（list，方便 JSON / deepcopy / Qt）
        _last_frame_256 = list(vals)

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

            # ================= FSM =================
            if _state == STATE_IDLE:
                if val_filt >= PRESS_ON:
                    if (_last_peak_t is None) or (recv_ms - _last_peak_t >= MIN_INTER_PRESS_MS):
                        _state = STATE_PRESSING
                        _press_start_t = recv_ms
                        _peak_max_val = val_filt
                        _peak_max_idx = _global_idx
                        _peak_max_t   = recv_ms

            elif _state == STATE_PRESSING:
                if val_filt > _peak_max_val:
                    _peak_max_val = val_filt
                    _peak_max_idx = _global_idx
                    _peak_max_t   = recv_ms

                # ===== 一次按压结束（时间判定）=====
                if recv_ms - _press_start_t >= MIN_PRESS_DURATION_MS:
                    interval = None
                    bpm = None
                    if _last_peak_t is not None:
                        interval = _peak_max_t - _last_peak_t
                        bpm = 60000 / interval

                    # ★ 关键修改：把 256 点 frame 一起存入峰值
                    _peaks.append({
                        "idx": _peak_max_idx,
                        "val": int(_peak_max_val),
                        "t": _peak_max_t,
                        "frame": _last_frame_256,   # ✅ 现在一定是 256 长度
                    })

                    print(
                        f"[PEAK #{len(_peaks):02d}] "
                        f"idx={_peak_max_idx:5d} "
                        f"val={int(_peak_max_val):6d} "
                        f"t={_peak_max_t} "
                        f"{'' if bpm is None else f'interval={interval}ms bpm={bpm:.1f}'}"
                    )

                    _last_peak_t = _peak_max_t
                    _state = STATE_IDLE

            _global_idx += 1

    ser.close()
    print("[pressure] rx exit")

# ================= API =================
def init_pressure_detector():
    global _running
    _running = True
    threading.Thread(target=_rx_thread, daemon=True).start()
    print("[pressure] detector started")


def export_series_csv(path="pressure_series_test.csv"):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "host_ms", "val_raw", "val_filt"])
        for s in _samples:
            w.writerow([s["idx"], s["host_ms"], s["val_raw"], int(s["val_filt"])])


def export_peaks_csv(path="pressure_peaks_test.csv"):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "peak_val", "t_ms"])
        for p in _peaks:
            w.writerow([p["idx"], p["val"], p["t"]])


# ================= main =================
if __name__ == "__main__":
    print("========== pressure_detector test ==========")

    init_pressure_detector()
    time.sleep(20)

    export_series_csv()
    export_peaks_csv()

    print("========== test finished ==========")
    print(f"Detected peaks: {len(_peaks)}")
