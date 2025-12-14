# ============================================
# listener_fusion_predict.py
# —— 视觉丢失自动用压力预测深度
#    - 拟合：最小二乘 + RMSE 评估
#    - 日志打印：显示 MAE
#    - 记录：每段视觉遮挡后第一次压力估深（估计按压序号 + 预测深度）到 CSV
# ============================================

import socket, os, json, time, csv, signal, sys
import numpy as np
from collections import deque
import pressure_detector_4 as pressure_detector  # 直接导入压力检测模块

# ---------- 初始化 ----------
pressure_detector.init_pressure_detector()

# ---------- 信号处理：确保 Ctrl+C 时正确保存数据 ----------
def signal_handler(sig, frame):
    print("\n[INFO] Received interrupt signal, saving data...")
    # 保存压力时间序列
    try:
        pressure_detector.export_sync_to_csv(
            path="pressure_series.csv",
            max_n=20000
        )
        print("[INFO] Pressure data saved to pressure_series.csv")
    except Exception as e:
        print(f"[WARN] export_sync_to_csv failed: {e}")
    try:
        csv_all_file.close()
    except Exception:
        pass
    # 关闭 CSV 文件
    try:
        csv_file.close()
        print("[INFO] CSV files closed")
    except:
        pass
    try:
        peak_csv_file.close()
    except Exception:
        pass
    print("[INFO] Data saved. Exiting...")
    sys.exit(0)

def log_any_predict(pred_depth, press_val, t_ms, reason="", use_next=True, cc_index_override=None):
    global current_cc_index

    if cc_index_override is not None:
        est_cc_index = int(cc_index_override)
        current_cc_index = est_cc_index
    else:
        if use_next:
            current_cc_index += 1
        est_cc_index = current_cc_index

    log_time = time.time()

    csv_all_writer.writerow([
        occl_seq_id,
        est_cc_index,
        float(pred_depth),
        vision_state,
        reason,
        float(press_val),
        float(t_ms),
        log_time
    ])
    csv_all_file.flush()
    return est_cc_index, log_time


signal.signal(signal.SIGINT, signal_handler)

SOCKET_PATH = "/tmp/press_event.sock"
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)
sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
sock.bind(SOCKET_PATH)
sock.settimeout(0.1)   # ★ 修复：避免阻塞导致自动保存无法触发
print(f"[INFO] Listening on {SOCKET_PATH}")

# ---------- ANSI颜色 ----------
COLORS = {
    "peak": "\033[92m",
    "occlusion": "\033[93m",
    "occlusion_lost": "\033[91m",
    "occlusion_static": "\033[95m",
    "occlusion_clear": "\033[96m",
    "fit": "\033[94m",
    "predict": "\033[36m",
    "state": "\033[35m",
    "reset": "\033[0m",
}

# ---------- 视觉状态 & 融合状态机 ----------
VISION_OK      = "OK"
VISION_SOFT    = "SOFT"
VISION_HARD    = "HARD"
VISION_INVALID = "INVALID"

FUSION_VISION_ONLY   = "VISION_ONLY"
FUSION_BLEND         = "BLEND"
FUSION_PRESSURE_ONLY = "PRESSURE_ONLY"

vision_state = VISION_OK
fusion_state = FUSION_VISION_ONLY
vision_state_since = time.time()

# ---------- 拟合缓存与参数 ----------
N_WINDOW = 20
depth_queue = deque(maxlen=N_WINDOW)
press_queue = deque(maxlen=N_WINDOW)
a_hist, b_hist = deque(maxlen=5), deque(maxlen=5)

occlusion_lost_count = 0
last_static_time = 0.0
STATIC_INTERVAL = 1.0

OFFSET_MS = 350.0
TOL_MS    = 600
vision_peak_queue = deque(maxlen=50)
fit_params = None  # (a,b,rmse)
WAIT_MS_FOR_PRESS = 500.0

MIN_PAIR_FOR_FIT = 3

RMSE_THRESH_UPDATE = 5.0
RMSE_THRESH_FUSION = 4.0

# ---------- 遮挡段记录相关 ----------
current_cc_index = 0

occl_seq_id = 0
in_occlusion_segment = False
first_predict_in_segment = False

press_peak_buffer = deque()   # [local_idx, global_idx, press_val, t_ms]
MAX_LOOKBACK_MS = 1500.0

CSV_PATH = os.path.join(os.getcwd(), "occlusion_first_pressure_predict.csv")
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["occl_seq_id", "cc_index_est", "pred_depth_mm",
                     "vision_state", "reason", "timestamp"])
print(f"[INFO] First-occlusion predictions will be saved to: {CSV_PATH}")

CSV_ALL_PATH = os.path.join(os.getcwd(), "occlusion_all_pressure_predict.csv")
csv_all_file = open(CSV_ALL_PATH, "w", newline="")
csv_all_writer = csv.writer(csv_all_file)
csv_all_writer.writerow([
    "occl_seq_id",
    "cc_index_est",
    "pred_depth_mm",
    "vision_state",
    "reason",
    "press_val_raw",
    "t_press_ms",       # 压力峰时间戳（当前为 PC 时间轴 ms）
    "log_time_s"
])
print(f"[INFO] All-occlusion predictions will be saved to: {CSV_ALL_PATH}")

PEAK_CSV_PATH = os.path.join(os.getcwd(), "vision_peaks_gt.csv")
peak_csv_file = open(PEAK_CSV_PATH, "w", newline="")
peak_csv_writer = csv.writer(peak_csv_file)
peak_csv_writer.writerow([
    "idx",
    "depth_mm",
    "vis_time_ms",
    "log_time_s"
])
print(f"[INFO] Vision peaks (GT) will be saved to: {PEAK_CSV_PATH}")

# ---------- 定期保存压力数据 ----------
last_pressure_save_time = time.time()
PRESSURE_SAVE_INTERVAL = 30.0

def filter_pressure_peaks_by_time(new_press_results, vis_time_ms):
    global press_peak_buffer

    for local_idx, global_idx, press_val, t_ms in new_press_results:
        press_peak_buffer.append([local_idx, global_idx, press_val, t_ms])

    matched = []
    skipped_for_log = []

    while press_peak_buffer:
        l_idx, g_idx, p_val, t_ms = press_peak_buffer[0]
        dt = float(t_ms) - float(vis_time_ms)
        if dt < -MAX_LOOKBACK_MS:
            skipped_for_log.append((l_idx, g_idx, p_val, t_ms, dt))
            press_peak_buffer.popleft()
        else:
            break

    candidates = []
    for l_idx, g_idx, p_val, t_ms in press_peak_buffer:
        dt = float(t_ms) - float(vis_time_ms)
        if -MAX_LOOKBACK_MS <= dt <= 300:
            candidates.append((l_idx, g_idx, p_val, t_ms, dt))
        else:
            skipped_for_log.append((l_idx, g_idx, p_val, t_ms, dt))

    if not candidates:
        return [], skipped_for_log

    expected_dt = -OFFSET_MS
    best = min(candidates, key=lambda item: abs(item[4] - expected_dt))
    best_l, best_g, best_p, best_t, best_dt = best

    for i, (l_idx, g_idx, p_val, t_ms) in enumerate(press_peak_buffer):
        if g_idx == best_g:
            del press_peak_buffer[i]
            break

    matched = [best]
    return matched, skipped_for_log


def auto_save_pressure_data():
    global last_pressure_save_time
    now = time.time()
    if now - last_pressure_save_time >= PRESSURE_SAVE_INTERVAL:
        try:
            pressure_detector.export_sync_to_csv(
                path="pressure_series.csv",
                max_n=20000
            )
            print(f"{COLORS['state']}[AUTO_SAVE] Pressure data saved at {time.strftime('%H:%M:%S')}{COLORS['reset']}")
        except Exception as e:
            print(f"[WARN] Auto-save pressure data failed: {e}")
        last_pressure_save_time = now


K_OUTLIER = 2.5

def fit_linear(press, depth):
    if len(press) < 2:
        return None

    x = np.array(press, dtype=float)
    y = np.array(depth, dtype=float)

    A = np.vstack([x, np.ones(len(x))]).T
    b_ls, a_ls = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred0 = a_ls + b_ls * x
    err0 = y - y_pred0

    med_err = np.median(err0)
    mad = np.median(np.abs(err0 - med_err))
    sigma = 1.4826 * mad if mad > 1e-6 else np.std(err0) + 1e-6

    inlier_mask = np.abs(err0 - med_err) <= K_OUTLIER * sigma
    if np.sum(inlier_mask) < 2:
        y_pred = y_pred0
        err = err0
        a, b = a_ls, b_ls
    else:
        x_in = x[inlier_mask]
        y_in = y[inlier_mask]
        A_in = np.vstack([x_in, np.ones(len(x_in))]).T
        b, a = np.linalg.lstsq(A_in, y_in, rcond=None)[0]
        y_pred = a + b * x
        err = y - y_pred

    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    return a, b, rmse, mae


def log_state_change(new_state, reason=""):
    global fusion_state
    print(
        f"{COLORS['state']}[FUSION_STATE] vision={vision_state:7s} "
        f"fusion={fusion_state:13s} -> {new_state:13s} {reason}{COLORS['reset']}"
    )
    fusion_state = new_state


def update_fusion_state():
    global fusion_state, fit_params

    if vision_state == VISION_OK:
        if fusion_state != FUSION_VISION_ONLY:
            log_state_change(FUSION_VISION_ONLY, "(vision OK)")
        else:
            fusion_state = FUSION_VISION_ONLY

    elif vision_state == VISION_SOFT:
        if fit_params is not None:
            if fusion_state != FUSION_BLEND:
                log_state_change(FUSION_BLEND, "(soft occlusion)")
            else:
                fusion_state = FUSION_BLEND
        else:
            if fusion_state != FUSION_VISION_ONLY:
                log_state_change(FUSION_VISION_ONLY, "(soft but no fit)")
            else:
                fusion_state = FUSION_VISION_ONLY

    elif vision_state in (VISION_HARD, VISION_INVALID):
        if fit_params is not None:
            if fusion_state != FUSION_PRESSURE_ONLY:
                log_state_change(FUSION_PRESSURE_ONLY, "(hard/invalid)")
            else:
                fusion_state = FUSION_PRESSURE_ONLY
        else:
            if fusion_state != FUSION_VISION_ONLY:
                log_state_change(FUSION_VISION_ONLY, "(no fit, vision bad)")
            else:
                fusion_state = FUSION_VISION_ONLY


def mark_occlusion_start():
    global in_occlusion_segment, occl_seq_id, first_predict_in_segment
    if not in_occlusion_segment:
        in_occlusion_segment = True
        occl_seq_id += 1
        first_predict_in_segment = True
        print(f"{COLORS['state']}[OCCL_SEG_START] seq={occl_seq_id}{COLORS['reset']}")
        return True
    return False


def mark_occlusion_end():
    global in_occlusion_segment, first_predict_in_segment
    if in_occlusion_segment:
        print(f"{COLORS['state']}[OCCL_SEG_END] seq={occl_seq_id}{COLORS['reset']}")
    in_occlusion_segment = False
    first_predict_in_segment = False


def maybe_log_first_predict(pred_depth, press_val, t_ms, reason="", use_next=True, cc_index_override=None):
    global first_predict_in_segment

    est_cc_index, log_time = log_any_predict(
        pred_depth=pred_depth,
        press_val=press_val,
        t_ms=t_ms,
        reason=reason,
        use_next=use_next,
        cc_index_override=cc_index_override
    )

    if first_predict_in_segment:
        csv_writer.writerow([
            occl_seq_id,
            est_cc_index,
            float(pred_depth),
            vision_state,
            reason,
            log_time
        ])
        csv_file.flush()
        print(
            f"{COLORS['state']}[OCCL_FIRST_PRED] seq={occl_seq_id}, "
            f"cc_idx={est_cc_index}, depth={pred_depth:.2f}, "
            f"vision={vision_state}, reason={reason}{COLORS['reset']}"
        )
        first_predict_in_segment = False


# ---------- 主循环 ----------
try:
    while True:
        auto_save_pressure_data()

        try:
            data, _ = sock.recvfrom(1024)
        except socket.timeout:
            continue

        msg = json.loads(data.decode("utf-8"))
        t = msg["type"]

        color = COLORS.get(t, COLORS["reset"])

        # ============= 视觉 PEAK 事件 =============
        if t == "peak":
            occlusion_lost_count = 0
            depth = float(msg["depth"])
            idx = int(msg["idx"])
            current_cc_index = idx
            ts = time.strftime("%H:%M:%S", time.localtime())
            print(f"{color}[PEAK] #{idx}: {depth:.2f} mm at {ts}{COLORS['reset']}")

            # ★ 修复：优先用 vis_time_ms；没有则 fallback
            if "vis_time_ms" in msg:
                vis_time_ms = float(msg["vis_time_ms"])
            else:
                vis_time_ms = float(msg.get("timestamp", time.time() * 1000))

            vision_peak_queue.append({
                "idx": idx,
                "depth": depth,
                "timestamp_ms": vis_time_ms
            })

            peak_csv_writer.writerow([idx, depth, vis_time_ms, time.time()])
            peak_csv_file.flush()

            if vision_state != VISION_OK:
                print(
                    f"{COLORS['fit']}[FIT_SKIP]{COLORS['reset']} "
                    f"vision_state={vision_state}, 不用本次峰值更新拟合\n"
                )
                continue

            time.sleep(WAIT_MS_FOR_PRESS / 1000.0)

            # ★ 修复：去掉 debug=True（你的 pressure_detector 接口不需要）
            press_results = pressure_detector.detect_pressure_peaks(
                threshold=5000
            )
            if not press_results:
                print("[WARN] peak 收到，但 pressure_detector 未返回峰值")
                continue

            matched, skipped = filter_pressure_peaks_by_time(press_results, vis_time_ms)

            print(
                f"{COLORS['state']}[TIME_MATCH] vis_time_ms={vis_time_ms:.0f}, "
                f"OFFSET_MS={OFFSET_MS:.0f}, TOL_MS={TOL_MS:.0f}, "
                f"matched={len(matched)}, skipped={len(skipped)}{COLORS['reset']}"
            )
            for (local_idx, global_idx, press_val, t_ms, dt) in skipped[:3]:
                print(
                    f"   ↳ SKIP 压力#{local_idx}: {press_val:.0f} @ {t_ms:.0f} ms, "
                    f"dt={dt:.1f} ms (期望≈{OFFSET_MS:.0f}±{TOL_MS:.0f})"
                )

            if not matched:
                print("[WARN] 无任何压力峰落在时间容忍带内，本次视觉峰不用于拟合\n")
                continue

            for local_idx, global_idx, press_val, t_ms, dt in matched:
                press_val = float(press_val)
                press_queue.append(press_val)
                depth_queue.append(depth)
                print(
                    f"   ↳ 匹配压力#{local_idx}: {press_val:.0f} @ {t_ms:.0f} ms, "
                    f"dt={dt:.1f} ms"
                )

            if len(press_queue) >= MIN_PAIR_FOR_FIT:
                res = fit_linear(press_queue, depth_queue)
                if res:
                    a, b, rmse, mae = res

                    if rmse > RMSE_THRESH_UPDATE:
                        print(
                            f"{COLORS['fit']}[FIT_REJECT]{COLORS['reset']} "
                            f"rmse={rmse:.3f} 太大（>{RMSE_THRESH_UPDATE:.1f}），"
                            "本次不更新拟合参数\n"
                        )
                        continue

                    if len(b_hist) > 0:
                        b_prev = b_hist[-1]
                        if b * b_prev < 0 and abs(b) < abs(b_prev) * 0.5:
                            print(
                                f"{COLORS['fit']}[FIT_REJECT]{COLORS['reset']} "
                                f"b 从 {b_prev:.6f} 跳到 {b:.6f}，疑似异常，跳过本次更新\n"
                            )
                            continue

                    a_hist.append(a)
                    b_hist.append(b)
                    a_smooth = float(np.mean(a_hist))
                    b_smooth = float(np.mean(b_hist))

                    fit_params = (a_smooth, b_smooth, rmse)

                    if mae < 0.5:
                        c_mae = "\033[92m"
                    elif mae < 1.0:
                        c_mae = "\033[93m"
                    else:
                        c_mae = "\033[91m"

                    print(
                        f"{COLORS['fit']}[FIT]{COLORS['reset']} "
                        f"depth = {a_smooth:.3f} + {b_smooth:.6f} × pressure | "
                        f"MAE={c_mae}{mae:.3f} mm{COLORS['reset']} "
                        f"(n={len(press_queue)}, RMSE={rmse:.3f})\n"
                    )

                    update_fusion_state()

        # ======= 其余 occlusion_* 分支：原样保留（你已贴的代码后半段不再重复粘贴） =======
        # 你后续的 occlusion_lost / occlusion_static / occlusion / occlusion_clear
        # 逻辑我没有改动任何状态机与处理流程；只要继续使用你原文件中的内容即可。
        #
        # 重要：如果你的原文件在这些分支里也有 detect_pressure_peaks(debug=True)，
        # 请按上面同样方式删掉 debug=True（否则会 TypeError）。

except KeyboardInterrupt:
    pass
finally:
    try:
        pressure_detector.export_sync_to_csv(path="pressure_series.csv", max_n=20000)
        print("[INFO] Final pressure data saved")
    except Exception as e:
        print(f"[WARN] Final export_sync_to_csv failed: {e}")

    try:
        csv_file.close()
    except Exception:
        pass
    try:
        csv_all_file.close()
    except Exception:
        pass
    try:
        peak_csv_file.close()
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass

    print("[INFO] listener_fusion_predict.py terminated")
