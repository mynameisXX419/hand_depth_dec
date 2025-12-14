# pressure_detector_4.py
# -----------------------------------------
# 从 MCU 通过串口接收 16x16 压力阵列数据
#   - 帧格式：BB AA | length | 512B payload | crc16
#   - 使用 PC 接收时间（ms）作为时间轴
#   - detect_pressure_peaks：在压力标量序列中找峰值（增量式）
#   - export_sync_to_csv：导出最近的压力序列（用于实验日志）
# -----------------------------------------

import serial
import struct
import threading
import time
from typing import Optional

# ================= 协议定义 =================
HEADER = 0xAABB
PAYLOAD_LEN = 512                  # 256 * uint16
PACKET_SIZE = 2 + 2 + PAYLOAD_LEN + 2  # = 518 bytes

# ================= 串口参数 =================
SERIAL_DEV = "/dev/ttyUSB0"
BAUDRATE   = 921600
TIMEOUT_S  = 1.0

# ================= 全局缓存 =================
_samples = []   # {"idx", "host_ms", "recv_pc_ms", "val"}
_global_sample_idx = 0
_last_peak_sample_idx = -1

_lock = threading.Lock()
_running = False


def _crc16(data: bytes) -> int:
    """CRC-16 (Modbus A001)"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


def _read_exact(ser: serial.Serial, length: int) -> Optional[bytes]:
    """从串口精确读取 length 字节；读不到则返回 None（依赖 timeout）"""
    buf = b""
    while len(buf) < length and _running:
        chunk = ser.read(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf if len(buf) == length else None


def _find_header(ser: serial.Serial) -> bool:
    """在串口字节流中寻找 BB AA 帧头"""
    while _running:
        b = ser.read(1)
        if not b:
            continue
        if b == b"\xBB":
            b2 = ser.read(1)
            if b2 == b"\xAA":
                return True
    return False


def _rx_thread_main():
    global _global_sample_idx, _running

    print(f"[pressure_detector] open serial {SERIAL_DEV} @ {BAUDRATE}")

    try:
        ser = serial.Serial(
            SERIAL_DEV,
            BAUDRATE,
            timeout=TIMEOUT_S
        )
    except Exception as e:
        print(f"[pressure_detector] open serial failed: {e}")
        _running = False
        return

    while _running:
        # 1) 同步帧头
        if not _find_header(ser):
            break

        # 2) 包接收起始时间（PC时间轴，ms）
        recv_pc_ms = int(time.time() * 1000)

        # 3) 读取剩余字节
        rest = _read_exact(ser, PACKET_SIZE - 2)
        if rest is None:
            continue

        packet = b"\xBB\xAA" + rest

        # 4) 解包 header + length
        try:
            header, length = struct.unpack("<HH", packet[:4])
        except struct.error:
            continue

        if header != HEADER or length != PAYLOAD_LEN:
            continue

        payload = packet[4:4 + PAYLOAD_LEN]
        if len(payload) != PAYLOAD_LEN:
            continue

        crc_recv = struct.unpack("<H", packet[-2:])[0]
        if _crc16(packet[:-2]) != crc_recv:
            continue

        # 5) payload -> 256 x uint16 -> 标量（sum）
        try:
            vals = struct.unpack("<256H", payload)
        except struct.error:
            continue

        press_scalar = int(sum(vals))

        with _lock:
            _samples.append({
                "idx": _global_sample_idx,
                "host_ms": recv_pc_ms,   # 统一PC时间轴
                "recv_pc_ms": recv_pc_ms,
                "val": press_scalar,
            })
            _global_sample_idx += 1

        if _global_sample_idx % 50 == 0:
            print(
                f"[pressure_detector] sample#{_global_sample_idx}: "
                f"host_ms={recv_pc_ms}, val={press_scalar}"
            )

    try:
        ser.close()
    except Exception:
        pass
    print("[pressure_detector] rx thread exit")


def init_pressure_detector():
    global _running
    if _running:
        return
    _running = True
    th = threading.Thread(target=_rx_thread_main, daemon=True)
    th.start()
    print("[pressure_detector] init done, serial rx thread started")


def detect_pressure_peaks(
    threshold: int = 5000,
    min_distance_ms: int = 150,
    smooth_window: int = 3,
    debug: bool = False,   # ★ 保留这个形参以兼容旧代码，但不影响逻辑；你现在 listener 已不再传 debug
):
    """
    返回新检测到的压力峰：
      (local_idx, global_idx, press_val, t_ms)

    - local_idx：本次调用的局部序号
    - global_idx：全局样本序号（递增）
    - press_val：压力标量（sum）
    - t_ms：PC时间轴（ms）
    """
    global _last_peak_sample_idx

    results = []
    with _lock:
        n = len(_samples)
        if n < 3:
            return results

        # 当前只实现 smooth_window=3（与你原逻辑一致）
        smooth_vals = [0.0] * n
        for i in range(1, n - 1):
            smooth_vals[i] = (
                _samples[i - 1]["val"] +
                _samples[i]["val"] +
                _samples[i + 1]["val"]
            ) / 3.0

        start = max(_last_peak_sample_idx + 1, 1)
        end = n - 1
        if start >= end:
            return results

        local_idx = 0
        last_peak_time = None

        for i in range(start, end):
            if not (smooth_vals[i] >= smooth_vals[i - 1] and
                    smooth_vals[i] >= smooth_vals[i + 1]):
                continue

            if smooth_vals[i] < threshold:
                continue

            host_ms = _samples[i]["host_ms"]
            if last_peak_time is not None and (host_ms - last_peak_time) < min_distance_ms:
                continue

            results.append((
                local_idx,
                _samples[i]["idx"],
                _samples[i]["val"],
                float(host_ms),
            ))
            local_idx += 1
            last_peak_time = host_ms

        _last_peak_sample_idx = n - 2

    return results


def export_sync_to_csv(path: str = "pressure_series.csv", max_n: int = 20000):
    """导出最近 max_n 个样本：idx, host_ms, val"""
    import csv
    with _lock:
        data = _samples[-max_n:]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "host_ms", "val"])
        for s in data:
            w.writerow([s["idx"], s["host_ms"], s["val"]])
