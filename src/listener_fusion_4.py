# ============================================
# listener_fusion_predict.py
# —— 视觉丢失自动用压力预测深度
# ============================================

import socket
import os
import json
import time
import csv
import signal
import sys
import numpy as np
from collections import deque

import pressure_detector_4 as pressure_detector
last_peak_time_ms = None
last_press_val = None


# ================= 初始化压力检测 =================
pressure_detector.init_pressure_detector()


# ================= Ctrl+C 处理 =================
def signal_handler(sig, frame):
    print("\n[INFO] Ctrl+C received, cleaning up...")

    # ===== CSV =====
    try:
        pressure_detector.export_series_csv("pressure_series.csv")
    except Exception as e:
        print(f"[WARN] export_series_csv failed: {e}")

    for f in (csv_file, csv_all_file, peak_csv_file):
        try:
            f.close()
        except Exception:
            pass

    # ===== sockets =====
    global sock, ui_sock, pressure_sock
    global UI_LOCAL_PATH, PRESSURE_LOCAL_PATH

    if sock:
        try:
            sock.close()
            print("[CLEAN] sock closed")
        except Exception:
            pass

    if ui_sock:
        try:
            ui_sock.close()
            print("[CLEAN] ui_sock closed")
        except Exception:
            pass

    if pressure_sock:
        try:
            pressure_sock.close()
            print("[CLEAN] pressure_sock closed")
        except Exception:
            pass

    # ===== unlink socket paths =====
    for p in (RX_SOCKET_PATH, UI_LOCAL_PATH, PRESSURE_LOCAL_PATH):
        if p and os.path.exists(p):
            try:
                os.unlink(p)
                print(f"[CLEAN] unlinked {p}")
            except Exception:
                pass

    print("[INFO] exit")
    sys.exit(0)



signal.signal(signal.SIGINT, signal_handler)


# ================= Socket =================

# 接收视觉 / 上游事件（你原来的）
RX_SOCKET_PATH = "/tmp/press_event.sock"
if os.path.exists(RX_SOCKET_PATH):
    os.remove(RX_SOCKET_PATH)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(RX_SOCKET_PATH)
sock.settimeout(0.1)
print(f"[INFO] Listening on {RX_SOCKET_PATH}")

# 发送给 Qt UI 的 socket（只发，不 bind）
UI_SOCKET_PATH = "/tmp/ui_event.sock"

ui_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

# ===== 新增：sender 本地地址（必须）=====
UI_LOCAL_PATH = f"/tmp/ui_sender_{os.getpid()}.sock"
try:
    os.unlink(UI_LOCAL_PATH)
except FileNotFoundError:
    pass

ui_sock.bind(UI_LOCAL_PATH)

# ===== 改动 2：等待 Qt UI socket ready =====
print("[INFO] waiting for UI socket...")
while not os.path.exists(UI_SOCKET_PATH):
    time.sleep(0.1)
print("[INFO] UI socket ready")


# ================= 新增：pressure frame socket =================

UI_PRESSURE_SOCKET_PATH = "/tmp/ui_pressure.sock"

print("[INFO] waiting for UI pressure socket...")
while not os.path.exists(UI_PRESSURE_SOCKET_PATH):
    time.sleep(0.1)
print("[INFO] UI pressure socket ready")

pressure_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

PRESSURE_LOCAL_PATH = f"/tmp/ui_pressure_sender_{os.getpid()}.sock"
try:
    os.unlink(PRESSURE_LOCAL_PATH)
except FileNotFoundError:
    pass

pressure_sock.bind(PRESSURE_LOCAL_PATH)

print("[INFO] pressure socket sender ready")


# ================= 颜色 =================
COLORS = {
    "peak": "\033[92m",
    "occlusion": "\033[93m",
    "occlusion_lost": "\033[91m",
    "occlusion_static": "\033[95m",
    "occlusion_clear": "\033[96m",
    "fit": "\033[94m",
    "state": "\033[35m",
    "reset": "\033[0m",
}


# ================= 状态定义 =================
VISION_OK = "OK"
VISION_SOFT = "SOFT"
VISION_HARD = "HARD"
VISION_INVALID = "INVALID"

FUSION_VISION_ONLY = "VISION_ONLY"
FUSION_BLEND = "BLEND"
FUSION_PRESSURE_ONLY = "PRESSURE_ONLY"

vision_state = VISION_OK
fusion_state = FUSION_VISION_ONLY


# ================= 拟合缓存 =================
N_WINDOW = 20
press_queue = deque(maxlen=N_WINDOW)
depth_queue = deque(maxlen=N_WINDOW)
a_hist, b_hist = deque(maxlen=5), deque(maxlen=5)

fit_params = None
MIN_PAIR_FOR_FIT = 3
RMSE_THRESH_UPDATE = 5.0


# ================= 时间对齐 =================
OFFSET_MS = 350.0
MAX_LOOKBACK_MS = 1500.0
WAIT_MS_FOR_PRESS = 500.0

press_peak_buffer = deque()


# ================= CSV =================
csv_file = open("occlusion_first_pressure_predict.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "occl_seq_id", "cc_index_est", "pred_depth_mm",
    "vision_state", "reason", "log_time_s"
])

csv_all_file = open("occlusion_all_pressure_predict.csv", "w", newline="")
csv_all_writer = csv.writer(csv_all_file)
csv_all_writer.writerow([
    "occl_seq_id", "cc_index_est", "pred_depth_mm",
    "vision_state", "reason",
    "press_val", "press_time_ms", "log_time_s"
])

peak_csv_file = open("vision_peaks_gt.csv", "w", newline="")
peak_csv_writer = csv.writer(peak_csv_file)
peak_csv_writer.writerow([
    "idx", "depth_mm", "vis_time_ms", "log_time_s"
])


# ================= 变量 =================
current_cc_index = 0
occl_seq_id = 0
in_occlusion_segment = False
first_predict_in_segment = False


# ================= 工具函数 =================
def fit_linear(press, depth):
    if len(press) < 2:
        return None
    x = np.array(press)
    y = np.array(depth)
    A = np.vstack([x, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = a + b * x
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    return a, b, rmse, mae


def filter_pressure_peaks(vis_time_ms):
    new_peaks = pressure_detector.fetch_new_peaks()

    # ===== 1️⃣ 新 pressure peak =====
    if new_peaks:
        print(
            f"[DEBUG][PRESSURE_PEAK] count={len(new_peaks)} "
            f"last=(press={new_peaks[-1][2]}, t={new_peaks[-1][3]})"
        )

    for item in new_peaks:
        press_peak_buffer.append(item)

    # ===== 2️⃣ 清理过期 peak =====
    while press_peak_buffer:
        l_idx, g_idx, p_val, t_ms, frame_256 = press_peak_buffer[0]
        dt = t_ms - vis_time_ms
        if dt < -MAX_LOOKBACK_MS:
            press_peak_buffer.popleft()
        else:
            break

    # ===== 3️⃣ 构造候选 =====
    candidates = []
    for item in press_peak_buffer:
        l_idx, g_idx, p_val, t_ms, frame_256 = item
        dt = t_ms - vis_time_ms
        if -MAX_LOOKBACK_MS <= dt <= 300:
            candidates.append((item, dt))

    # ===== 4️⃣ 无匹配 =====
    if not candidates:
        print(
            f"[DEBUG][ALIGN] no pressure peak match "
            f"vis_time={vis_time_ms:.1f} "
            f"buffer_size={len(press_peak_buffer)}"
        )
        return []

    # ===== 5️⃣ 选择最优 =====
    best, dt = min(candidates, key=lambda x: abs(x[1] + OFFSET_MS))
    press_peak_buffer.remove(best)

    print(
        f"[DEBUG][ALIGN] matched "
        f"dt={dt:.1f}ms "
        f"press={best[2]}"
    )

    return [best]

def mark_occlusion_start():
    global occl_seq_id, in_occlusion_segment, first_predict_in_segment
    if not in_occlusion_segment:
        occl_seq_id += 1
        in_occlusion_segment = True
        first_predict_in_segment = True
        print(f"{COLORS['state']}[OCCL_START] seq={occl_seq_id}{COLORS['reset']}")
        ui_sock.sendto(json.dumps({
            "type": "occlusion"
        }).encode(), UI_SOCKET_PATH)
        


def mark_occlusion_end():
    global in_occlusion_segment, first_predict_in_segment
    if in_occlusion_segment:
        print(f"{COLORS['state']}[OCCL_END] seq={occl_seq_id}{COLORS['reset']}")
        ui_sock.sendto(json.dumps({
            "type": "occlusion_clear"
        }).encode(), UI_SOCKET_PATH)
    in_occlusion_segment = False
    first_predict_in_segment = False


# ================= 主循环 =================
try:
    while True:
        try:
            data, _ = sock.recvfrom(1024)
        except socket.timeout:
            # 非遮挡：什么都不做
            if not in_occlusion_segment:
                continue

            # 遮挡但还没拟合好：也不做
            if fit_params is None:
                continue

            a, b, _ = fit_params

            new_peaks = pressure_detector.fetch_new_peaks()
            for l_idx, g_idx, p_val, t_ms, frame_256 in new_peaks:
                pred_depth = a + b * float(p_val)
                current_cc_index += 1

                print(
                    f"{COLORS['occlusion']}[PRESSURE_ONLY]{COLORS['reset']} "
                    f"seq={occl_seq_id}, "
                    f"cc_idx={current_cc_index}, "
                    f"press={p_val:.0f}, "
                    f"pred_depth={pred_depth:.2f} mm"
                )

                payload = {
                    "type": "pressure_only",
                    "press": float(p_val),
                    "pred_depth": float(pred_depth),
                    # "frame_256": frame_256,
                    "is_predicted": True
                }

                ui_sock.sendto(
                    json.dumps(payload).encode(),
                    UI_SOCKET_PATH
                )


                if frame_256 is None or len(frame_256) != 256:
                    print(f"[WARN] pressure_only frame_256 invalid: {type(frame_256)}")
                else:
                    pressure_payload = {
                        "type": "pressure_only",
                        "seq": current_cc_index,
                        "frame_256": frame_256,
                        "is_predicted": True
                    }
                    try:
                       pressure_sock.sendto(json.dumps(pressure_payload).encode(), UI_PRESSURE_SOCKET_PATH)
                    except Exception as e:
                        print("[ERROR] send pressure frame failed:", e)

                    print(
                        f"[DEBUG][PRESSURE_ONLY] "
                        f"seq={current_cc_index} "
                        f"press={p_val:.1f} "
                        f"frame_len={len(frame_256)}"
                    )

                # 记录到 CSV
                csv_all_writer.writerow([
                    occl_seq_id,
                    current_cc_index,
                    float(pred_depth),
                    VISION_HARD,
                    "pressure_only",
                    float(p_val),
                    float(t_ms),
                    time.time()
                ])
                csv_all_file.flush()


            continue


        msg = json.loads(data.decode("utf-8"))
        t = msg["type"]

        # ========== 视觉 PEAK ==========
        if t == "peak":
            depth = float(msg["depth"])
            idx = int(msg["idx"])
            current_cc_index = idx

            # --- 1. 时间戳 ---
            vis_time_ms = float(
                msg.get("vis_time_ms", time.time() * 1000)
            )

            # --- 2. BPM 计算 ---
            bpm = None
            if last_peak_time_ms is not None:
                dt_ms = vis_time_ms - last_peak_time_ms
                if dt_ms > 0:
                    bpm = 60_000.0 / dt_ms
            last_peak_time_ms = vis_time_ms

            print(f"{COLORS['peak']}[PEAK] #{idx} depth={depth:.2f}{COLORS['reset']}")

            # --- 3. 先发一次 peak（pressure 可能为空）---
            payload = {
                "type": "peak",
                "idx": idx,
                "depth": depth,
                "press": last_press_val,
                "bpm": bpm
            }
            ui_sock.sendto(json.dumps(payload).encode(), UI_SOCKET_PATH)

            # --- 4. 等待压力峰 ---
            time.sleep(WAIT_MS_FOR_PRESS / 1000.0)

            matched = filter_pressure_peaks(vis_time_ms)
            if not matched:
                print("[WARN] no matched pressure peak")
                continue

            # --- 5. 拿到 pressure ---
            _, _, p_val, _, frame_256 = matched[0]
            last_press_val = float(p_val)

            ui_sock.sendto(json.dumps({
                "type": "peak_update",
                "idx": idx,
                "press": last_press_val
                # "frame_256": frame_256      # ★ 新增：256 个压力值
            }).encode(), UI_SOCKET_PATH)
            # print("[DEBUG] peak_update sent, frame_256 len =", len(frame_256))
            
            # ✅ 只有 frame_256 合法才发压力帧
            if frame_256 is None or len(frame_256) != 256:
                print(f"[WARN] frame_256 invalid: {type(frame_256)} len={0 if frame_256 is None else len(frame_256)}")
            else:
                pressure_payload = {
                    "type": "peak_update",   # ✅ 强烈建议加 type（见第 3 点）
                    "seq": idx,
                    "frame_256": frame_256,
                    "is_predicted": False
                }
                pressure_sock.sendto(json.dumps(pressure_payload).encode(), UI_PRESSURE_SOCKET_PATH)
                print(
                    f"[DEBUG][SEND_PRESSURE] type=peak_update "
                    f"seq={idx} "
                    f"len={len(frame_256)} "
                    f"min={min(frame_256)} "
                    f"max={max(frame_256)}"
                )


            press_queue.append(last_press_val)
            depth_queue.append(depth)

            # --- 6. 拟合 ---
            if len(press_queue) >= MIN_PAIR_FOR_FIT:
                res = fit_linear(press_queue, depth_queue)
                if res:
                    a, b, rmse, mae = res
                    if rmse <= RMSE_THRESH_UPDATE:
                        a_hist.append(a)
                        b_hist.append(b)
                        fit_params = (
                            np.mean(a_hist),
                            np.mean(b_hist),
                            rmse
                        )

                        print(
                            f"{COLORS['fit']}[FIT]{COLORS['reset']} "
                            f"depth = {fit_params[0]:.3f} + "
                            f"{fit_params[1]:.6f} × press | "
                            f"RMSE={rmse:.3f}, MAE={mae:.3f}"
                        )

                        ui_sock.sendto(json.dumps({
                            "type": "fit",
                            "a": fit_params[0],
                            "b": fit_params[1],
                            "rmse": rmse,
                            "mae": mae
                        }).encode(), UI_SOCKET_PATH)




        # ====== 其余 occlusion_* 事件 ======
        elif t.startswith("occlusion"):
            if t in ("occlusion", "occlusion_lost"):
                mark_occlusion_start()
            elif t == "occlusion_clear":
                mark_occlusion_end()
            

except KeyboardInterrupt:
    pass
# finally:
#     signal_handler(None, None)
