# listener.py — 改进版（彩色输出 + 紫色持续过滤）
import socket, os, json, time

SOCKET_PATH = "/tmp/press_event.sock"
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(SOCKET_PATH)
print(f"[INFO] Listening on {SOCKET_PATH}")

# ANSI颜色定义
COLORS = {
    "peak": "\033[92m",              # 绿色
    "occlusion": "\033[93m",         # 黄色
    "occlusion_lost": "\033[91m",    # 红色
    "occlusion_static": "\033[95m",  # 紫色
    "occlusion_clear": "\033[96m",   # 青色
    "reset": "\033[0m",
}

last_static_time = 0.0      # 上一次输出 occlusion_static 的时间戳
STATIC_INTERVAL = 1.0       # 连续静态遮挡输出的最小间隔秒数

while True:
    data, _ = sock.recvfrom(1024)
    msg = json.loads(data.decode("utf-8"))
    t = msg["type"]
    ts = time.strftime("%H:%M:%S", time.localtime(msg["timestamp"]))
    color = COLORS.get(t, COLORS["reset"])

    if t == "peak":
        print(f"{color}[PEAK] #{msg['idx']}: {msg['depth']} mm at {ts}{COLORS['reset']}")

    elif t == "occlusion_static":
        now = time.time()
        if now - last_static_time >= STATIC_INTERVAL:  # 超过1s才打印
            print(f"{color}[OCCLUSION_STATIC] conf={msg.get('conf', 0):.2f}, depth={msg.get('depth', 0):.2f} at {ts}{COLORS['reset']}")
            last_static_time = now

    elif t in ("occlusion", "occlusion_lost", "occlusion_clear"):
        print(f"{color}[{t.upper()}] conf={msg.get('conf', 0):.2f}, depth={msg.get('depth', 0):.2f} at {ts}{COLORS['reset']}")

    else:
        print(f"{color}[UNKNOWN] {msg}{COLORS['reset']}")
