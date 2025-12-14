# pressure_detector.py
# -----------------------------------------
# 从 MCU 通过 TCP 接收 16x16 压力阵列数据：
#   - 帧格式：BB AA | length | t_mcu_ms | 512B payload | crc16
#   - soft sync：t_mcu_ms 对齐到 PC 时间轴
#   - detect_pressure_peaks：在压力标量序列中找峰值
# -----------------------------------------

import socket
import struct
import threading
import time
from typing import Optional


HEADER = 0xAABB
PACKET_SIZE = 518  # 2(header) + 2(len) + + 512(payload) + 2(crc)

HOST = "0.0.0.0"
PORT = 8000

# 全局缓存
_samples = []  # 每个元素: {"idx", "mcu_ms", "host_ms", "val"}
_global_sample_idx = 0
_last_peak_sample_idx = -1
_lock = threading.Lock()
_running = False


def _crc16(data: bytes) -> int:
    """和 MCU 一样的 CRC-16(A001) 实现"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _recv_exact(conn: socket.socket, length: int) -> Optional[bytes]:
    """从 TCP 流中精确读取 length 字节"""
    buf = b""
    while len(buf) < length:
        chunk = conn.recv(length - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _find_header(conn: socket.socket) -> bool:
    """在 TCP 流里寻找 BB AA 帧头"""
    while True:
        b = conn.recv(1)
        if not b:
            return False
        if b == b"\xBB":
            b2 = conn.recv(1)
            if not b2:
                return False
            if b2 == b"\xAA":
                return True  # 帧头 OK，后面继续收剩余 520 字节


def _rx_thread_main():
    global _global_sample_idx, _running

    print(f"[pressure_detector] TCP server listen on {HOST}:{PORT} ...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    conn, addr = server.accept()
    print(f"[pressure_detector] MCU connected from {addr}")

    first_packet = True
    t0_mcu = 0
    t0_pc_ms = 0

    while _running:
        # 1) 同步帧头 BB AA
        ok = _find_header(conn)
        if not ok:
            print("[pressure_detector] connection closed while sync header")
            break

        # 2) 收剩余 520 字节
                # 先记录一下“这个包真正开始接收的 PC 时间”
        recv_pc_ms = int(time.time() * 1000)
        rest = _recv_exact(conn, PACKET_SIZE - 2)
        if rest is None:
            print("[pressure_detector] connection closed while recv body")
            break

        full_packet = b"\xBB\xAA" + rest

        # 3) 解包
        try:
            header, length, t_mcu_ms = struct.unpack("<HHI", full_packet[:8])
        except struct.error:
            print("[pressure_detector] struct unpack failed, skip")
            continue

        if header != HEADER:
            print("[pressure_detector] header mismatch, skip")
            continue

        if length != 512:
            print(f"[pressure_detector] length mismatch: {length}, expect 512")
            continue

        payload_bytes = full_packet[8:520]
        crc_recv = struct.unpack("<H", full_packet[-2:])[0]
        crc_calc = _crc16(full_packet[:-2])

        if crc_calc != crc_recv:
            print("[pressure_detector] CRC mismatch, drop packet")
            continue

        # payload -> 256 个 uint16
        vals = struct.unpack("<256H", payload_bytes)
        # 这里取 max 作为一个标量压力值，你可以改成 sum/均值/中心区域等
        press_scalar = sum(vals)

        # soft sync: 第一个包建立 MCU 时间和 PC 时间映射
        now_pc_ms = int(time.time() * 1000)
        if first_packet:
            t0_mcu = t_mcu_ms
            t0_pc_ms = now_pc_ms
            first_packet = False
            print(
                f"[pressure_detector] soft sync set: "
                f"t0_mcu={t0_mcu} ms, t0_pc={t0_pc_ms} ms"
            )

        host_ms = t0_pc_ms + (int(t_mcu_ms) - int(t0_mcu))

        # 存入全局样本列表
        with _lock:
            _samples.append(
                {
                    "idx": _global_sample_idx,
                    "mcu_ms": int(t_mcu_ms),
                    "host_ms": int(host_ms),      # 软同步后的时间
                    "recv_pc_ms": recv_pc_ms,     # 实际到达 PC 的时间
                    "val": int(press_scalar),
                }
            )
            _global_sample_idx += 1

        # 可选：偶尔打印一下，调试用
        if _global_sample_idx % 50 == 0:
            print(
                f"[pressure_detector] sample#{_global_sample_idx}: "
                f"mcu={t_mcu_ms} ms, host={host_ms} ms, val={press_scalar}"
            )

    conn.close()
    server.close()
    print("[pressure_detector] rx thread exit")


def init_pressure_detector():
    """在 listener_fusion_predict.py 中调用：初始化并启动接收线程"""
    global _running
    if _running:
        return
    _running = True
    th = threading.Thread(target=_rx_thread_main, daemon=True)
    th.start()
    print("[pressure_detector] init done, rx thread started")


def detect_pressure_peaks(
    threshold: int = 5000,
    min_distance_ms: int = 150,
    smooth_window: int = 3,
    debug: bool = False,
):
    """
    峰值检测（带简单滤波 + 去重）：
      1. 对 val 做简单平滑（3 点或 5 点滑动平均）
      2. 在平滑后的序列上找局部极大值
      3. 加幅度阈值：val >= threshold
      4. 加“不应期”：两个峰的 host_ms 至少间隔 min_distance_ms
    返回: list of (local_idx, global_idx, press_val, t_ms_host)

    如果 debug=True，当本次没有返回任何峰值时，会打印为什么：
      - 当前区间内最大值是多少
      - 是否都低于 threshold
      - 是否被 min_distance_ms 限制过滤掉
    """
    global _last_peak_sample_idx

    results = []
    with _lock:
        n = len(_samples)
        if n < 3:
            if debug:
                print("[pressure_detector] no peaks: n < 3")
            return results

        # ---------- 1) 构造一个平滑后的序列 ----------
        smooth_vals = [0.0] * n
        for i in range(1, n - 1):
            smooth_vals[i] = (_samples[i - 1]["val"] +
                              _samples[i]["val"] +
                              _samples[i + 1]["val"]) / 3.0

        # ---------- 2) 只在新样本区间内做峰值检测 ----------
        start = max(_last_peak_sample_idx + 1, 1)
        end = n - 1
        if start >= end:
            if debug:
                print(f"[pressure_detector] no peaks: start({start}) >= end({end})")
            return results

        local_idx = 0
        last_peak_time = None  # 上一次接受的峰的 host_ms，用于“不应期”

        # 为了 debug 统计一下候选点情况
        candidate_cnt = 0
        below_threshold_cnt = 0
        refractory_cnt = 0

        # 预先算一下当前区间的平滑最大值
        max_smooth_val = max(smooth_vals[start:end])

        for i in range(start, end):
            prev_v = smooth_vals[i - 1]
            cur_v  = smooth_vals[i]
            next_v = smooth_vals[i + 1]

            # 先看是不是局部极大值
            if not (cur_v >= prev_v and cur_v >= next_v):
                continue

            candidate_cnt += 1

            # 振幅阈值
            if cur_v < threshold:
                below_threshold_cnt += 1
                if debug:
                    raw_v = _samples[i]["val"]
                    print(
                        f"[pressure_detector][DEBUG] peak_candidate@{i} rejected: "
                        f"cur_v={cur_v:.1f} (raw={raw_v}) < threshold={threshold}"
                    )
                continue

            global_idx = _samples[i]["idx"]
            host_ms    = _samples[i]["host_ms"]
            raw_val    = _samples[i]["val"]  # 返回原始的 sum(vals)，不是平滑值

            # ---------- 3) 不应期：与上一个峰至少间隔 min_distance_ms ----------
            if last_peak_time is not None and host_ms - last_peak_time < min_distance_ms:
                refractory_cnt += 1
                if debug:
                    print(
                        f"[pressure_detector][DEBUG] peak_candidate@{i} rejected by refractory: "
                        f"host_ms={host_ms}, last_peak_time={last_peak_time}, "
                        f"Δ={host_ms - last_peak_time} < {min_distance_ms}"
                    )
                continue

            # 通过检查，接受这个峰
            results.append((local_idx, global_idx, raw_val, float(host_ms)))
            local_idx += 1
            last_peak_time = host_ms

        # 记录已经处理到哪一帧了
        _last_peak_sample_idx = n - 2

        # ---------- 4) 如果没有任何峰值，就给个整体 debug 总结 ----------
        if debug and not results:
            print(
                "[pressure_detector] no peaks found in this call: "
                f"n={n}, search_range=[{start},{end}), "
                f"max_smooth_val={max_smooth_val:.1f}, threshold={threshold}, "
                f"candidates={candidate_cnt}, "
                f"below_threshold={below_threshold_cnt}, "
                f"refractory_filtered={refractory_cnt}"
            )

    return results

def export_sync_to_csv(path="sync_debug.csv", max_n=5000):
    import csv
    with _lock:
        data = list(_samples[-max_n:])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # 多加一列 val
        # w.writerow(["idx", "mcu_ms", "host_ms", "recv_pc_ms", "val"])
        w.writerow(["idx", "mcu_ms", "host_ms", "val"])
        for s in data:
            w.writerow([
                s["idx"],
                s["mcu_ms"],
                s["host_ms"],
                #s["recv_pc_ms"],
                s["val"],
            ])
    print(f"[sync_debug] exported {len(data)} samples to {path}")

