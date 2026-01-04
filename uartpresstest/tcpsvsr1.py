import serial
import struct
import time

# ================= 协议定义 =================
HEADER = 0xAABB
PAYLOAD_LEN = 512
PACKET_SIZE = 2 + 2 + PAYLOAD_LEN + 2  # 518

# ================= 串口参数 =================
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 2000000
TIMEOUT = 0.1

# ================= 行为控制 =================
PRINT_GRID = False      # ← 关键！先关掉
STAT_INTERVAL = 1.0     # FPS 统计窗口（秒）


def find_header(ser):
    while True:
        b = ser.read(1)
        if b == b'\xBB':
            if ser.read(1) == b'\xAA':
                return


def read_exact(ser, length, max_wait_ms=200):
    data = b""
    t0 = time.time()
    while len(data) < length:
        more = ser.read(length - len(data))
        if more:
            data += more
        elif (time.time() - t0) * 1000 > max_wait_ms:
            return None
    return data


def crc16_a001(data: bytes):
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc >> 1) ^ 0xA001) if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


def parse_packet(packet: bytes):
    if len(packet) != PACKET_SIZE:
        return None

    header, length = struct.unpack("<HH", packet[:4])
    if header != HEADER or length != PAYLOAD_LEN:
        return None

    payload = packet[4:4 + PAYLOAD_LEN]
    crc_recv = struct.unpack("<H", packet[-2:])[0]
    if crc16_a001(packet[:-2]) != crc_recv:
        return None

    data = struct.unpack("<256H", payload)
    return [data[i * 16:(i + 1) * 16] for i in range(16)]


def receive_loop(ser):
    print("📡 Syncing packet header...")

    ok_cnt = 0
    crc_err_cnt = 0
    t_stat_start = time.monotonic()
    packet_id = 0

    while True:
        find_header(ser)
        rest = read_exact(ser, PACKET_SIZE - 2)
        if rest is None:
            continue

        packet = b'\xBB\xAA' + rest
        grid = parse_packet(packet)

        if grid is None:
            crc_recv = struct.unpack("<H", packet[-2:])[0]
            if crc16_a001(packet[:-2]) != crc_recv:
                crc_err_cnt += 1
            continue

        # ===== 成功一帧 =====
        packet_id += 1
        ok_cnt += 1

        # ===== 可选打印 =====
        if PRINT_GRID:
            print(f"\n[PACKET #{packet_id}]")
            for r in grid:
                print(" ".join(f"{v:4d}" for v in r))

        # ===== FPS 统计（关键）=====
        now = time.monotonic()
        if now - t_stat_start >= STAT_INTERVAL:
            fps = ok_cnt / (now - t_stat_start)
            total = ok_cnt + crc_err_cnt
            loss = (crc_err_cnt / total * 100) if total else 0.0

            print(
                f"📊 FPS={fps:6.1f} | OK={ok_cnt:5d} | "
                f"CRC_ERR={crc_err_cnt:4d} | LOSS={loss:5.2f}%"
            )

            t_stat_start = now
            ok_cnt = 0
            crc_err_cnt = 0


if __name__ == "__main__":
    print(f"Opening serial {SERIAL_PORT} @ {BAUDRATE}")
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=TIMEOUT,
    )

    try:
        receive_loop(ser)
    except KeyboardInterrupt:
        print("\nExit.")
    finally:
        ser.close()
