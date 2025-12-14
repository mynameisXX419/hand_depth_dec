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
import pressure_detector_3 as pressure_detector# 直接导入你写的压力检测模块

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
    
    # 关闭 CSV 文件
    try:
        csv_file.close()
        print("[INFO] CSV files closed")
    except:
        pass
    
    print("[INFO] Data saved. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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
    "state": "\033[35m",             # 状态机日志
    "reset": "\033[0m",
}

# ---------- 视觉状态 & 融合状态机 ----------
VISION_OK      = "OK"
VISION_SOFT    = "SOFT"
VISION_HARD    = "HARD"
VISION_INVALID = "INVALID"

FUSION_VISION_ONLY   = "VISION_ONLY"
FUSION_BLEND         = "BLEND"         # 轻遮挡/静止时可视为"融合期"
FUSION_PRESSURE_ONLY = "PRESSURE_ONLY" # 视觉不可用/无效时只用压力

vision_state = VISION_OK
fusion_state = FUSION_VISION_ONLY
vision_state_since = time.time()

# ---------- 拟合缓存与参数 ----------
N_WINDOW = 20
depth_queue = deque(maxlen=N_WINDOW)
press_queue = deque(maxlen=N_WINDOW)
a_hist, b_hist = deque(maxlen=5), deque(maxlen=5)  # 平滑系数缓存

occlusion_lost_count = 0
last_static_time = 0.0
STATIC_INTERVAL = 1.0

# ========= 多模态时间容忍参数（根据离线分析结果） =========
# 这里假设你这一轮实验使用的 offset≈200 ms、容忍带≈200 ms
# 如果你以后按轮次区分，可以改成字典或从配置文件读取
OFFSET_MS = -350.0    # 压力时间相对视觉时间的整体偏移（t_press - t_vis）
TOL_MS    = 400    # 数据驱动推荐的时间容忍（大约 400 ms）
vision_peak_queue = deque(maxlen=50)
# 拟合参数：a, b, rmse（内部仍然保留 RMSE）
fit_params = None  # (a,b,rmse)

MIN_PAIR_FOR_FIT = 3  # 最少多少对数据才允许拟合

# ---------- 遮挡段记录相关 ----------
# 估计"第几次按压"。视觉 peak 的 idx 直接用消息里的 idx；
# 遮挡期间压力预测的按压序号：在最后一个 visual idx 基础上累加。
current_cc_index = 0

# 每一段视觉遮挡（连续 occlusion_* ~ clear）给一个序号
occl_seq_id = 0
in_occlusion_segment = False
first_predict_in_segment = False  # 本遮挡段内第一次预测是否已经记录

# CSV：记录每段遮挡的第一次压力估深
CSV_PATH = os.path.join(os.getcwd(), "occlusion_first_pressure_predict.csv")
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["occl_seq_id", "cc_index_est", "pred_depth_mm",
                     "vision_state", "reason", "timestamp"])
print(f"[INFO] First-occlusion predictions will be saved to: {CSV_PATH}")

# ---------- 定期保存压力数据 ----------
last_pressure_save_time = time.time()
PRESSURE_SAVE_INTERVAL = 30.0  # 每30秒自动保存一次

def filter_pressure_peaks_by_time(press_results, vis_time_ms):
    matched = []
    skipped = []
    for local_idx, global_idx, press_val, t_ms in press_results:
        dt = float(t_ms) - float(vis_time_ms)
        # 只要 |dt| < 1500ms 就认为可以
        if (OFFSET_MS - TOL_MS) <= dt <= (OFFSET_MS + TOL_MS):
            matched.append((local_idx, global_idx, press_val, t_ms, dt))
        else:
            skipped.append((local_idx, global_idx, press_val, t_ms, dt))
    return matched, skipped


def auto_save_pressure_data():
    """定期自动保存压力数据，防止数据丢失"""
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


# ---------- 拟合函数 ----------
def fit_linear(press, depth):
    """
    最小二乘线性拟合：
        depth ≈ a + b * press
    返回：a, b, rmse, mae
    """
    if len(press) < 2:
        return None
    x = np.array(press, dtype=float)
    y = np.array(depth, dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    b, a = np.linalg.lstsq(A, y, rcond=None)[0]
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
    """
    根据当前 vision_state + 拟合是否存在，更新 fusion_state。
    拟合本身仍用 RMSE 评估，但此处只看 fit_params 是否存在。
    """
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
    """
    标记一段新的"遮挡段"开始：无论是 soft / hard / invalid。
    只在从"非遮挡状态"进入遮挡时调用一次。
    """
    global in_occlusion_segment, occl_seq_id, first_predict_in_segment
    if not in_occlusion_segment:
        in_occlusion_segment = True
        occl_seq_id += 1
        first_predict_in_segment = True
        print(f"{COLORS['state']}[OCCL_SEG_START] seq={occl_seq_id}{COLORS['reset']}")


def mark_occlusion_end():
    """
    遮挡段结束（收到 occlusion_clear 时调用）。
    """
    global in_occlusion_segment, first_predict_in_segment
    if in_occlusion_segment:
        print(f"{COLORS['state']}[OCCL_SEG_END] seq={occl_seq_id}{COLORS['reset']}")
    in_occlusion_segment = False
    first_predict_in_segment = False


def maybe_log_first_predict(pred_depth, reason=""):
    """
    在当前遮挡段中，如果这是"第一次压力估深"，把
    (occl_seq_id, 估计按压序号, 预测深度, 视觉状态, reason, 时间戳) 写入 CSV。
    同时更新 current_cc_index。
    """
    global first_predict_in_segment, current_cc_index

    # 每次预测都认为是"下一次按压"
    est_cc_index = current_cc_index + 1

    if first_predict_in_segment:
        ts_now = time.time()
        csv_writer.writerow([
            occl_seq_id,
            est_cc_index,
            float(pred_depth),
            vision_state,
            reason,
            ts_now
        ])
        csv_file.flush()
        print(
            f"{COLORS['state']}[OCCL_FIRST_PRED] seq={occl_seq_id}, "
            f"cc_idx={est_cc_index}, depth={pred_depth:.2f}, "
            f"vision={vision_state}, reason={reason}{COLORS['reset']}"
        )
        first_predict_in_segment = False

    # 更新全局"按压计数"
    current_cc_index = est_cc_index


# ---------- 主循环 ----------
try:
    while True:
        # 定期自动保存压力数据
        auto_save_pressure_data()
        
        data, _ = sock.recvfrom(1024)
        msg = json.loads(data.decode("utf-8"))
        t = msg["type"]
        # ★ 视觉时间戳统一从 vis_time_ms 读取（毫秒）
        # if "vis_time_ms" in msg:
        #     vis_time_ms = float(msg["vis_time_ms"])
        # else:
        #     # 兼容旧格式
        #     vis_time_ms = float(msg.get("timestamp", time.time()*1000))

        color = COLORS.get(t, COLORS["reset"])

        # ============= 视觉 PEAK 事件 =============
        if t == "peak":
            occlusion_lost_count = 0  # 重置遮挡计数
            depth = float(msg["depth"])
            idx = int(msg["idx"])
            current_cc_index = idx  # 认为视觉的 idx 就是"第几次按压"
            ts = time.strftime("%H:%M:%S", time.localtime())
            print(f"{color}[PEAK] #{idx}: {depth:.2f} mm at {ts}{COLORS['reset']}")
            # 视觉峰对应的时间（ms），用于和压力峰对齐
            # 统一读取视觉峰的时间戳
            vis_time_ms = float(msg["vis_time_ms"])

            vision_peak_queue.append({
                "idx": idx,
                "depth": depth,
                "timestamp_ms": vis_time_ms
            })
            # time.sleep(0.35)   # 等待压力峰出现
            if vision_state != VISION_OK:
                print(
                    f"{COLORS['fit']}[FIT_SKIP]{COLORS['reset']} "
                    f"vision_state={vision_state}, 不用本次峰值更新拟合\n"
                )
                continue

            # 检测压力峰值（可能返回最近一小段时间内的多个峰）
            press_results = pressure_detector.detect_pressure_peaks(
                threshold=2000,
                debug=True,
            )
            if not press_results:
                print("[WARN] peak 收到，但 pressure_detector 未返回峰值")
                continue

            # ========= ★ 根据时间容忍带筛选压力峰 ★ =========
            matched, skipped = filter_pressure_peaks_by_time(press_results, vis_time_ms)

            # 调试：看有多少峰被匹配/丢弃
            print(
                f"{COLORS['state']}[TIME_MATCH] vis_time_ms={vis_time_ms:.0f}, "
                f"OFFSET_MS={OFFSET_MS:.0f}, TOL_MS={TOL_MS:.0f}, "
                f"matched={len(matched)}, skipped={len(skipped)}{COLORS['reset']}"
            )
            for (local_idx, global_idx, press_val, t_ms, dt) in skipped[:3]:
                # 只打印少数几个被丢弃的例子，防止刷屏
                print(
                    f"   ↳ SKIP 压力#{local_idx}: {press_val:.0f} @ {t_ms:.0f} ms, "
                    f"dt={dt:.1f} ms (期望≈{OFFSET_MS:.0f}±{TOL_MS:.0f})"
                )

            if not matched:
                print("[WARN] 无任何压力峰落在时间容忍带内，本次视觉峰不用于拟合\n")
                continue

            # ========= 只用 matched 的压力峰更新拟合 =========
            for local_idx, global_idx, press_val, t_ms, dt in matched:
                press_val = float(press_val)
                press_queue.append(press_val)
                depth_queue.append(depth)
                print(
                    f"   ↳ 匹配压力#{local_idx}: {press_val:.0f} @ {t_ms:.0f} ms, "
                    f"dt={dt:.1f} ms"
                )

            # 更新拟合参数（基于最近 N_WINDOW 对真实 (press, depth)）
            if len(press_queue) >= MIN_PAIR_FOR_FIT:
                res = fit_linear(press_queue, depth_queue)
                if res:
                    a, b, rmse, mae = res
                    # 平滑参数更新（a,b）
                    a_hist.append(a)
                    b_hist.append(b)
                    a_smooth = float(np.mean(a_hist))
                    b_smooth = float(np.mean(b_hist))
                    fit_params = (a_smooth, b_smooth, rmse)

                    # MAE 颜色分级（打印用 MAE）
                    if mae < 0.5:
                        c_mae = "\033[92m"  # green
                    elif mae < 1.0:
                        c_mae = "\033[93m"  # yellow
                    else:
                        c_mae = "\033[91m"  # red

                    print(
                        f"{COLORS['fit']}[FIT]{COLORS['reset']} "
                        f"depth = {a_smooth:.3f} + {b_smooth:.6f} × pressure | "
                        f"MAE={c_mae}{mae:.3f} mm{COLORS['reset']} "
                        f"(n={len(press_queue)}, RMSE={rmse:.3f})\n"
                    )

                    # 视觉状态良好 + 有拟合，刷新一次融合状态
                    update_fusion_state()

        # ============= 严重遮挡：视觉完全丢失 =============
        elif t == "occlusion_lost":
            occlusion_lost_count += 1
            conf = float(msg.get("conf", 0.0))
            depth_vis = float(msg.get("depth", 0.0))

            mark_occlusion_start()
            # ⭐ 拿最近视觉峰作为新的 current_cc_index
            last_vis = vision_peak_queue[-1] if len(vision_peak_queue) > 0 else None
            if last_vis:
                current_cc_index = last_vis["idx"]

            vision_state = VISION_HARD
            vision_state_since = time.time()
            update_fusion_state()

            print(
                f"{color}[OCCLUSION_LOST] conf={conf:.2f}, "
                f"cnt={occlusion_lost_count}, depth_vis={depth_vis:.2f}{COLORS['reset']}"
            )

            # 在 PRESSURE_ONLY 模式下尝试用压力顶上深度
            if fusion_state == FUSION_PRESSURE_ONLY and fit_params is not None:
                press_results = pressure_detector.detect_pressure_peaks()
                if not press_results:
                    continue  # 无压力峰值则跳过

                a, b, rmse = fit_params
                for local_idx, global_idx, press_val, t_ms in press_results:
                    press_val = float(press_val)
                    pred_depth = a + b * press_val
                    print(
                        f"{COLORS['predict']}[PREDICT] (HARD) 压力#{local_idx}: "
                        f"{press_val:.0f} → 预测深度={pred_depth:.2f} mm{COLORS['reset']}"
                    )
                    # 记录本遮挡段的第一次预测
                    maybe_log_first_predict(pred_depth, reason="hard")

        # ============= 静态遮挡（压着不动） =============
        elif t == "occlusion_static":
            now = time.time()
            conf = float(msg.get("conf", 0.0))
            depth_vis = float(msg.get("depth", 0.0))

            mark_occlusion_start()
            # ⭐ 恢复 index
            last_vis = vision_peak_queue[-1] if len(vision_peak_queue) > 0 else None
            if last_vis:
                current_cc_index = last_vis["idx"]

            vision_state = VISION_SOFT
            vision_state_since = now
            update_fusion_state()
            ts = time.strftime("%H:%M:%S", time.localtime())

            # 打印频率限制
            if now - last_static_time >= STATIC_INTERVAL:
                print(
                    f"{color}[OCCLUSION_STATIC] conf={conf:.2f}, "
                    f"depth_vis={depth_vis:.2f} at {ts}{COLORS['reset']}"
                )
                last_static_time = now

            # 轻度遮挡/静止下，可以选择性预测一下（融合期）
            if fusion_state in (FUSION_BLEND, FUSION_PRESSURE_ONLY) and fit_params is not None:
                press_results = pressure_detector.detect_pressure_peaks()
                if not press_results:
                    continue
                a, b, rmse = fit_params
                for local_idx, global_idx, press_val, t_ms in press_results:
                    press_val = float(press_val)
                    pred_depth = a + b * press_val
                    print(
                        f"{COLORS['predict']}[PREDICT] (STATIC) 压力#{local_idx}: "
                        f"{press_val:.0f} → 预测深度={pred_depth:.2f} mm{COLORS['reset']}"
                    )
                    maybe_log_first_predict(pred_depth, reason="static")

        # ============= 一般遮挡（低置信度 / 他人手 / 深度异常） =============
        elif t == "occlusion":
            conf = float(msg.get("conf", 0.0))
            depth_vis = float(msg.get("depth", 0.0))
            reason = msg.get("reason", "")

            mark_occlusion_start()
            # ⭐ 恢复 index
            last_vis = vision_peak_queue[-1] if len(vision_peak_queue) > 0 else None
            if last_vis:
                current_cc_index = last_vis["idx"]

            # 根据 reason 判断是 soft 还是 invalid
            if reason and "invalid" in reason:
                vision_state = VISION_INVALID
            else:
                vision_state = VISION_SOFT
            vision_state_since = time.time()
            update_fusion_state()
            ts = time.strftime("%H:%M:%S", time.localtime())

            reason_str = f" reason={reason}" if reason else ""
            print(
                f"{color}[OCCLUSION] conf={conf:.2f}, depth_vis={depth_vis:.2f}, "
                f"vision_state={vision_state}{reason_str}{COLORS['reset']}"
            )

            # 对于 invalid / soft 遮挡，同样可以尝试用压力估深
            if fusion_state in (FUSION_BLEND, FUSION_PRESSURE_ONLY) and fit_params is not None:
                press_results = pressure_detector.detect_pressure_peaks()
                if not press_results:
                    continue
                a, b, rmse = fit_params
                for local_idx, global_idx, press_val, t_ms in press_results:
                    press_val = float(press_val)
                    pred_depth = a + b * press_val
                    print(
                        f"{COLORS['predict']}[PREDICT] (OCC) 压力#{local_idx}: "
                        f"{press_val:.0f} → 预测深度={pred_depth:.2f} mm{COLORS['reset']}"
                    )
                    maybe_log_first_predict(pred_depth, reason=(reason or "soft"))

        # ============= 遮挡解除 =============
        elif t == "occlusion_clear":
            conf = float(msg.get("conf", 0.0))
            depth_vis = float(msg.get("depth", 0.0))

            occlusion_lost_count = 0
            vision_state = VISION_OK
            vision_state_since = time.time()
            mark_occlusion_end()
            update_fusion_state()
            ts = time.strftime("%H:%M:%S", time.localtime())

            print(
                f"{color}[OCCLUSION_CLEAR] conf={conf:.2f}, "
                f"depth_vis={depth_vis:.2f} at {ts}{COLORS['reset']}"
            )

        else:
            # 可能是未来扩展的事件类型
            print(f"{color}[UNKNOWN] {msg}{COLORS['reset']}")

finally:
    # 最后保存一次压力数据
    try:
        pressure_detector.export_sync_to_csv(
            path="pressure_series.csv",
            max_n=20000
        )
        print("[INFO] Final pressure data saved")
    except Exception as e:
        print(f"[WARN] Final export_sync_to_csv failed: {e}")
    
    # 关闭 CSV 文件
    try:
        csv_file.close()
    except Exception:
        pass
    sock.close()
    print("[INFO] listener_fusion_predict.py terminated")
