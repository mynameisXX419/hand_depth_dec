import serial
import struct
import time

HEADER = 0xAABB
PAYLOAD_LEN = 512
PACKET_SIZE = 2 + 2 + PAYLOAD_LEN + 2  # = 518

# ===== 串口参数 =====
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 2000000
TIMEOUT = 0.1


def find_header(ser):
    """在串口字节流中寻找 BB AA"""
    while True:
        b = ser.read(1)
        if not b:
            continue
        if b == b'\xBB':
            b2 = ser.read(1)
            if b2 == b'\xAA':
                return


def read_exact(ser, length, max_wait_ms=200):
    data = b""
    t0 = time.time()
    while len(data) < length:
        more = ser.read(length - len(data))
        if more:
            data += more
        else:
            if (time.time() - t0) * 1000 > max_wait_ms:
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

    payload_bytes = packet[4:4 + PAYLOAD_LEN]
    if len(payload_bytes) != PAYLOAD_LEN:
        return None

    crc_recv = struct.unpack("<H", packet[-2:])[0]
    crc_calc = crc16_a001(packet[:-2])
    if crc_calc != crc_recv:
        print("⚠ CRC mismatch")
        return None

    payload = struct.unpack("<256H", payload_bytes)
    grid = [payload[i * 16:(i + 1) * 16] for i in range(16)]
    return grid


def receive_loop(ser):
    packet_id = 0
    print("📡 Syncing packet header...")

    while True:
        find_header(ser)

        rest = read_exact(ser, PACKET_SIZE - 2)
        if rest is None:
            print("⚠ packet timeout, resync")
            continue

        packet = b'\xBB\xAA' + rest
        grid = parse_packet(packet)
        if grid is None:
            continue

        packet_id += 1
        print(f"\n[PACKET #{packet_id}]")
        for r in grid:
            print("  " + " ".join(f"{v:4d}" for v in r))


if __name__ == "__main__":
    print(f"Opening serial {SERIAL_PORT} @ {BAUDRATE}")
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=TIMEOUT,
    )

    try:
        receive_loop(ser)
    except KeyboardInterrupt:
        print("\nExit.")
    finally:
        ser.close()
