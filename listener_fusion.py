# ============================================
# listener_fusion_predict.py
# —— 视觉丢失自动用压力预测深度（含RMSE颜色与a,b平滑）
# ============================================

import socket, os, json, time
import numpy as np
from collections import deque
import pressure_detector  # 直接导入你写的压力检测模块

# ---------- 初始化 ----------
pressure_detector.init_pressure_detector()

SOCKET_PATH = "/tmp/press_event.sock"
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(SOCKET_PATH)
print(f"[INFO] Listening on {SOCKET_PATH}")

# ---------- ANSI颜色 ----------
COLORS = {
    "peak": "\033[92m",              # 绿色
    "occlusion": "\033[93m",         # 黄色
    "occlusion_lost": "\033[91m",    # 红色
    "occlusion_static": "\033[95m",  # 紫色
    "occlusion_clear": "\033[96m",   # 青色
    "fit": "\033[94m",               # 蓝色（拟合）
    "predict": "\033[36m",           # 青蓝色（预测）
    "reset": "\033[0m",
}

# ---------- 缓存与参数 ----------
N_WINDOW = 10
depth_queue = deque(maxlen=N_WINDOW)
press_queue = deque(maxlen=N_WINDOW)
a_hist, b_hist = deque(maxlen=5), deque(maxlen=5)  # 平滑系数缓存

occlusion_lost_count = 0
last_static_time = 0.0
STATIC_INTERVAL = 1.0
fit_params = None  # (a,b,rmse)

# ---------- 拟合函数 ----------
def fit_linear(press, depth):
    if len(press) < 2:
        return None
    x = np.array(press)
    y = np.array(depth)
    A = np.vstack([x, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = a + b * x
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return a, b, rmse


# ---------- 主循环 ----------
while True:
    data, _ = sock.recvfrom(1024)
    msg = json.loads(data.decode("utf-8"))
    t = msg["type"]
    ts = time.strftime("%H:%M:%S", time.localtime(msg["timestamp"]))
    color = COLORS.get(t, COLORS["reset"])

    # ============= 视觉 PEAK 事件 =============
    if t == "peak":
        occlusion_lost_count = 0  # 重置遮挡计数
        depth = msg["depth"]
        idx = msg["idx"]

        print(f"{color}[PEAK] #{idx}: {depth:.2f} mm at {ts}{COLORS['reset']}")

        # 检测压力峰值
        press_results = pressure_detector.detect_pressure_peaks()
        if not press_results:
            continue

        for local_idx, global_idx, press_val, t_ms in press_results:
            press_queue.append(press_val)
            depth_queue.append(depth)
            print(f"   ↳ 压力 #{local_idx}: {press_val:.0f} @ {t_ms:.0f} ms")

        # 更新拟合参数
        if len(press_queue) >= 3:
            res = fit_linear(press_queue, depth_queue)
            if res:
                a, b, rmse = res
                # 平滑参数更新
                a_hist.append(a)
                b_hist.append(b)
                a_smooth = np.mean(a_hist)
                b_smooth = np.mean(b_hist)
                fit_params = (a_smooth, b_smooth, rmse)

                # RMSE颜色分级
                if rmse < 0.5:
                    c_rmse = "\033[92m"  # green
                elif rmse < 1.0:
                    c_rmse = "\033[93m"  # yellow
                else:
                    c_rmse = "\033[91m"  # red

                print(
                    f"{COLORS['fit']}[FIT]{COLORS['reset']} depth = {a_smooth:.3f} + {b_smooth:.6f} × pressure | "
                    f"RMSE={c_rmse}{rmse:.3f} mm{COLORS['reset']} (n={len(press_queue)})\n"
                )

    # ============= 连续遮挡检测 =============
    elif t == "occlusion_lost":
        occlusion_lost_count += 1
        print(f"{color}[OCCLUSION_LOST] conf={msg.get('conf', 0):.2f}, cnt={occlusion_lost_count}{COLORS['reset']}")

        # 当连续遮挡10次以上时尝试使用压力预测
        if occlusion_lost_count >= 10 and fit_params is not None:
            press_results = pressure_detector.detect_pressure_peaks()
            if not press_results:
                continue  # 无压力峰值则跳过

            a, b, rmse = fit_params
            for local_idx, global_idx, press_val, t_ms in press_results:
                pred_depth = a + b * press_val
                print(
                    f"{COLORS['predict']}[PREDICT] 压力#{local_idx}: {press_val:.0f} → 预测深度={pred_depth:.2f} mm (遮挡){COLORS['reset']}"
                )
                # 保持拟合序列连续
                press_queue.append(press_val)
                depth_queue.append(pred_depth)

    # ============= 静态遮挡与其他事件 =============
    elif t == "occlusion_static":
        now = time.time()
        if now - last_static_time >= STATIC_INTERVAL:
            print(f"{color}[OCCLUSION_STATIC] conf={msg.get('conf', 0):.2f}, depth={msg.get('depth', 0):.2f} at {ts}{COLORS['reset']}")
            last_static_time = now

    elif t in ("occlusion", "occlusion_clear"):
        occlusion_lost_count = 0
        print(f"{color}[{t.upper()}] conf={msg.get('conf', 0):.2f}, depth={msg.get('depth', 0):.2f} at {ts}{COLORS['reset']}")

    else:
        print(f"{color}[UNKNOWN] {msg}{COLORS['reset']}")
