# ================== 手部按压深度检测（全面放宽阈值版：快速节奏友好） ==================
# 功能：
#  - 峰值检测：导数拐点 + 不应期(220ms) + 自动武装(400ms) + prominence(3.5mm) + 去重(120ms)
#  - 视觉遮挡检测：低置信度 / 静止 / 丢失 / 他人手（状态机版）
#  - 输出事件到 /tmp/press_event.sock

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv, time, os, json, socket
from camera_calib_loader import load_camera_params

# ------------------ 基本配置 ------------------
WINDOW_NAME = "Hand Depth Monitor"
CALIB_FILE       = "camera_gp23.yml"
EXTRINSIC_FILE   = "extrinsic_result.yml"
CAM_ID           = 2
LOG_PATH         = "./hand_depth_plane_avg.csv"
SOCKET_PATH      = "/tmp/press_event.sock"

# ✅ 改为支持多手检测（从 1 → 2）
MAX_HANDS = 2
MODEL_COMPLEXITY = 1
HAND_BACK_KEYS   = [0, 5, 9, 13, 17]
ZERO_FRAMES      = 100

fs = cv2.FileStorage(EXTRINSIC_FILE, cv2.FILE_STORAGE_READ)
R_oc = fs.getNode("rotation_matrix").mat()
T_oc = fs.getNode("translation_vector").mat()
fs.release()

if R_oc is None or T_oc is None:
    raise RuntimeError("外参文件读取失败")

Z0_MM_DEFAULT = float(T_oc[2,0] * 1000.0)  # ★ 用标定的 Z
print("Z0_MM_DEFAULT =", Z0_MM_DEFAULT)
GAIN_FILE        = "press_gain.json"
RC_FILE          = "residual_correction.json"
ZERO_OFFSET_MM   = 0.0
STABLE_THRESH_MM = 0.5


# ------------------ 判定参数（放宽） ------------------
PEAK_MIN_MM      = 8.0
PEAK_MAX_MM      = 70.0
CONF_THRESH      = 0.45
OCCLUSION_FRAMES = 5

# ✅ 全面放宽峰值检测参数
MIN_INTERVAL_MS  = 220
ARM_THRESH_MM    = 10.0
PROM_MM          = 3.5
MERGE_WINDOW_MS  = 120
AUTO_REARM_MS    = 400
STD_STATIC_WIN   = 10
STD_STATIC_TH    = 1.0

# ✅ 新增：视觉合法深度区间（用于区分“他人手/异常手”）
DEPTH_VALID_MIN_MM = -10.0
DEPTH_VALID_MAX_MM =  70.0
INVALID_FRAMES_TH  = 3

# ✅ 新增：主手选择时的位置约束（像素）
HAND_DIST_NEAR_PX = 40.0
HAND_DIST_FAR_PX  = 120.0


# ------------------ Socket事件广播 ------------------
def send_event(event_type, **kwargs):
    """
    event_type:
      - peak
      - occlusion / occlusion_static / occlusion_lost / occlusion_clear
      - INVALID_HAND 时仍用 occlusion + reason="invalid_*"
    """
    event = {"type": event_type, **kwargs}
    # 不要在这里生成时间戳！！

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(event).encode("utf-8"), SOCKET_PATH)
        sock.close()
    except:
        pass

# ------------------ 参数加载 ------------------
def load_gain():
    if os.path.exists(GAIN_FILE):
        try:
            with open(GAIN_FILE, "r") as f:
                d = json.load(f)
                return float(d.get("gain",1)), float(d.get("z0_mm",Z0_MM_DEFAULT)), float(d.get("offset_mm",ZERO_OFFSET_MM))
        except:
            pass
    return 1, Z0_MM_DEFAULT, ZERO_OFFSET_MM

def load_rc():
    if os.path.exists(RC_FILE):
        try:
            with open(RC_FILE, "r") as f:
                d = json.load(f)
                return float(d.get("a0",0)), float(d.get("a1",1)), float(d.get("a2",0))
        except:
            pass
    return 0.0, 1.0, 0.0

def apply_rc_dynamic(depth_raw, depth_filtered, rc_params, beta, gamma_boost):
    a0,a1,a2 = rc_params
    static_part  = a0 + a1*depth_filtered + a2*(depth_filtered**2)
    dynamic_part = beta*(depth_raw - depth_filtered)
    return gamma_boost*(static_part + dynamic_part)

# ------------------ 滤波类 ------------------
class EMA:
    def __init__(self, alpha=0.9):
        self.a = alpha
        self.v = None
    def update(self,x):
        self.v = x if self.v is None else self.a*x + (1-self.a)*self.v
        return self.v

class OneDimKalman:
    def __init__(self, dt=1/30, q_depth=3, q_vel=15, r_meas=6):
        self.dt=dt
        self.x=np.zeros((2,1)); self.P=np.eye(2)*100
        self.F=np.array([[1,dt],[0,1]]); self.H=np.array([[1,0]])
        self.Q=np.array([[q_depth,0],[0,q_vel]]); self.R=np.array([[r_meas]])
        self.inited=False
    def predict(self):
        self.x=self.F@self.x; self.P=self.F@self.P@self.F.T+self.Q
    def update(self,z):
        z=np.array([[float(z)]])
        if not self.inited:
            self.x[0,0]=z[0,0]; self.inited=True
        self.predict()
        y=z-self.H@self.x; S=self.H@self.P@self.H.T+self.R
        K=self.P@self.H.T@np.linalg.inv(S)
        self.x+=K@y; I=np.eye(2); self.P=(I-K@self.H)@self.P
        return float(self.x[0,0])

# ------------------ 相机参数 + 分辨率适配 ------------------
params = load_camera_params(CALIB_FILE)
K, D = params["K"], params["D"].reshape(-1)

print("原始内参 K:")
print(K)

# 粗略估计当时标定用的分辨率（cx, cy 大约在宽高的一半）
calib_width_est  = int(round(K[0, 2] * 2))
calib_height_est = int(round(K[1, 2] * 2))
print(f"估算标定分辨率: {calib_width_est} x {calib_height_est}")

# 先初始化 MediaPipe（保持你原来的结构）
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# 如有需要，可以强制设置分辨率，例如 640x480：
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

act_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
act_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"当前相机实际分辨率: {act_width} x {act_height}")

# 如果实际分辨率与标定分辨率明显不同，则按比例缩放 K
# if abs(act_width - calib_width_est) > 2 or abs(act_height - calib_height_est) > 2:
#     print("⚠ 当前分辨率与内参不匹配，按比例缩放 K")
#     sx = act_width  / calib_width_est
#     sy = act_height / calib_height_est
#     K[0, 0] *= sx  # fx
#     K[0, 2] *= sx  # cx
#     K[1, 1] *= sy  # fy
#     K[1, 2] *= sy  # cy
#     print("缩放后的 K:")
#     print(K)
# else:
#     print("✅ 当前分辨率与内参匹配，无需缩放 K")
if abs(act_width - calib_width_est) > 2 or abs(act_height - calib_height_est) > 2:
    print("⚠ 当前分辨率与标定分辨率不一致，为保持论文尺度，不对 K 做缩放，仅提示。")
else:
    print("✅ 当前分辨率与标定分辨率近似，可直接使用标定内参 K")

FY_PIX = float(K[1, 1])


# FY_PIX = float(K[1, 1])

# 读取外参
fs = cv2.FileStorage(EXTRINSIC_FILE, cv2.FILE_STORAGE_READ)
R_oc = fs.getNode("rotation_matrix").mat()
T_oc = fs.getNode("translation_vector").mat()
fs.release()
if R_oc is None or T_oc is None:
    raise RuntimeError("外参文件读取失败")

# 法向量 + 倾角余弦
n_c = (R_oc @ np.array([[0.0], [0.0], [1.0]])).reshape(3)
n_c = n_c / (np.linalg.norm(n_c) + 1e-9)
cos_tilt = abs(n_c[2])

# 像素→毫米比例（基于 Z0 和 fy）
px2mm = Z0_MM_DEFAULT / max(FY_PIX, 1e-6)
STABLE_THRESH_PX = STABLE_THRESH_MM / px2mm

# def pixel_to_world_depth_linear(dy_px, fy=FY_PIX, z0=Z0_MM_DEFAULT, cos_t=cos_tilt):
#     return (dy_px * (z0 / max(fy, 1e-6))) * max(cos_t, 1e-3)
def pixel_to_world_depth_linear(dy_px, fy=FY_PIX, z0=Z0_MM_DEFAULT, cos_t=cos_tilt):
    # 论文公式： CCD = (dy_px * Z0 / fy) / cos(theta)
    return (dy_px * z0 / max(fy, 1e-6)) * max(cos_t, 1e-3)

# ------------------ 初始化滤波与状态量 ------------------
ema = EMA(0.6)
kf  = OneDimKalman(dt=1/30, q_depth=20, q_vel=40, r_meas=3)
# kf= OneDimKalman(dt=1/30, q_depth=2.0, q_vel=8.0, r_meas=10.0)
gain, Z0_MM, ZERO_OFFSET_MM = load_gain()
rc_params = load_rc()
zero_ref_y = None
zero_buf = deque(maxlen=ZERO_FRAMES)

# 状态变量（峰值检测相关）
sig_hist   = deque(maxlen=3)
depth_hist = deque(maxlen=60)
frame_idx  = 0
peak_idx   = 0
prev_filtered = None
MOTION_EPS    = 1e-6

armed        = True
last_valley  = np.inf
last_peak_ms = 0
last_peak_depth = 0.0

# ==== 视觉状态机相关变量 ====
VISION_OK      = "ok"
VISION_SOFT    = "soft"
VISION_HARD    = "hard"
VISION_INVALID = "invalid"

vision_state   = VISION_OK
conf_ema       = 0.0
EMA_ALPHA_CONF = 0.6

soft_occ_frames = 0
no_hand_frames  = 0
invalid_frames  = 0

CONF_SOFT_TH     = 0.35
CONF_CLEAR_TH    = 0.55
SOFT_FRAMES_TH   = 5
NO_HAND_TH       = 3
STATIC_DEPTH_MIN = 5.0

# 主手位置，用于多手跟踪
prev_hand_cx = None
prev_hand_cy = None

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logf = open(LOG_PATH, "w", newline="")
writer = csv.writer(logf)

# 统一用 ms 级时间戳，信号和状态都打出来
writer.writerow([
    "frame_idx",
    "timestamp_ms",      
    "depth_raw_mm",      # 像素->mm 直接换算
    "depth_ema_mm",      # EMA 结果
    "depth_kf_mm",       # Kalman 结果（峰值检测用的 sig）
    "depth_corr_mm",     # 残差修正后的深度（现在 ~ 等于 kf）
    "sig_mm",            # 实际用于峰值检测的信号（= depth_kf_mm）
    "conf",              # 当前这一帧的手置信度
    "conf_ema",          # 置信度 EMA
    "vision_state",      # ok / soft / hard / invalid
    "has_hand",          # 这一帧是否有任意手检测到
    "has_main_hand",     # 是否选到了主手
    "velocity_mm_s",     # 卡尔曼估计速度
    "motion_mm",         # 当前帧相对上一帧的运动量
    "gain",
    "offset_mm",
    "cos_tilt"
])
print(f"[INFO] Visual depth log will be saved to: {LOG_PATH}")


print("\n>>> 开始：按压深度检测 + 峰值/遮挡事件广播（含多手 + INVALID_HAND 状态）\n")

# ------------------ 主循环 ------------------
with mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS,
                    model_complexity=MODEL_COMPLEXITY,
                    min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:        
        ok, frame = cap.read()
        timestamp_ms = int(time.time() * 1000)
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        # ================== 多手检测 + 主手选择 ==================
        raw_plane_y = None
        conf = 0.0

        has_any_hand = bool(res.multi_hand_landmarks)
        has_main_hand = False
        max_conf_any = 0.0

        main_lm = None
        main_cx = None
        main_cy = None

        if has_any_hand:
            best_score = -1e9
            best_idx   = -1
            best_raw_y = None
            best_conf  = 0.0
            best_cx    = None
            best_cy    = None

            for i, lm in enumerate(res.multi_hand_landmarks):
                # 手背关键点 y
                y_list = [lm.landmark[k].y * h for k in HAND_BACK_KEYS]
                x_list = [lm.landmark[k].x * w for k in HAND_BACK_KEYS]
                raw_y_i = np.average(y_list, weights=np.array([1.5, 1.2, 1.0, 1.0, 1.0]))
                cx_i = float(np.mean(x_list))
                cy_i = float(np.mean(y_list))

                # 置信度
                if res.multi_handedness and len(res.multi_handedness) > i:
                    conf_i = res.multi_handedness[i].classification[0].score
                else:
                    conf_i = 0.0

                max_conf_any = max(max_conf_any, conf_i)

                # 估算该手的粗略深度（仅用于打分）
                depth_raw_i_mm = 0.0
                depth_penalty  = 0.0
                if zero_ref_y is not None:
                    dy_i = raw_y_i - zero_ref_y
                    depth_raw_i_mm = pixel_to_world_depth_linear(
                        dy_i, FY_PIX, Z0_MM, cos_tilt
                    ) * gain + ZERO_OFFSET_MM
                    # 超出物理合理区间 → 可疑
                    if not (DEPTH_VALID_MIN_MM <= depth_raw_i_mm <= DEPTH_VALID_MAX_MM):
                        depth_penalty = 3.0

                # 打分：置信度 + 与上一帧主手距离 + 深度合理性
                score = 0.0
                score += conf_i * 2.0

                if prev_hand_cx is not None:
                    dist = np.hypot(cx_i - prev_hand_cx, cy_i - prev_hand_cy)
                    if dist < HAND_DIST_NEAR_PX:
                        score += 2.0
                    elif dist < HAND_DIST_FAR_PX:
                        score += 1.0
                    else:
                        score -= 1.0

                score -= depth_penalty

                if score > best_score:
                    best_score = score
                    best_idx   = i
                    best_raw_y = raw_y_i
                    best_conf  = conf_i
                    best_cx    = cx_i
                    best_cy    = cy_i

            # 选择主手：置信度要有一定下限
            if best_idx >= 0 and best_conf >= 0.3:
                has_main_hand = True
                raw_plane_y   = best_raw_y
                conf          = best_conf
                main_lm       = res.multi_hand_landmarks[best_idx]
                main_cx       = best_cx
                main_cy       = best_cy
                prev_hand_cx  = best_cx
                prev_hand_cy  = best_cy
                # 只画主手
                mp_draw.draw_landmarks(frame, main_lm, mp_hands.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF

        # ---------- 零位 ----------
        if zero_ref_y is None:
            if raw_plane_y is not None:
                zero_buf.append(raw_plane_y)
                if len(zero_buf) >= ZERO_FRAMES:
                    y_std = np.std(zero_buf)
                    if y_std < STABLE_THRESH_PX:
                        zero_ref_y = float(np.mean(zero_buf))
                    else:
                        zero_buf.clear()

            cv2.putText(frame, f"Collecting zero: {len(zero_buf)}/{ZERO_FRAMES}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            if key == 27:
                break
            continue

        # ================== 有主手帧：正常深度与遮挡逻辑 ==================
        if raw_plane_y is not None and has_main_hand:
            dy_px = raw_plane_y - zero_ref_y
            depth_mm_raw = pixel_to_world_depth_linear(
                dy_px, FY_PIX, Z0_MM, cos_tilt
            ) #* gain + ZERO_OFFSET_MM

            depth_mm_ema = ema.update(depth_mm_raw)
            depth_mm_kf  = kf.update(depth_mm_ema)
            velocity     = float(kf.x[1, 0])
            v_abs        = abs(velocity)
            motion       = abs(depth_mm_raw - (prev_filtered if prev_filtered is not None else depth_mm_kf))
            prev_filtered = depth_mm_kf

            # ✅ 改进版残差+动态补偿
            beta = np.clip(
                0.12 + 0.0008 * v_abs + 0.08 * (motion / (abs(depth_mm_raw) + MOTION_EPS)),
                0.12, 0.25
            )
            gamma_boost = np.clip(
                1.015 + 0.0002 * v_abs + 0.006 * (motion / (abs(depth_mm_raw) + MOTION_EPS)),
                1.015, 1.05
            )

            RC_GAIN = 0# RC_GAIN = 0.55

            depth_rc = apply_rc_dynamic(depth_mm_raw, depth_mm_kf, rc_params, beta, gamma_boost)
            depth_mm_corr = depth_mm_kf + RC_GAIN * (depth_rc - depth_mm_kf)

            sig = depth_mm_kf
            sig_hist.append(sig)
            depth_hist.append(sig)

            now_ms = int(time.time() * 1000)

            # ==== 更新置信度 EMA ====
            conf_ema = EMA_ALPHA_CONF * conf + (1.0 - EMA_ALPHA_CONF) * conf_ema
            no_hand_frames = 0

            # ==== 运动 / 静止检测 ====
            if len(depth_hist) >= 5:
                recent5 = np.array(list(depth_hist)[-5:])
                depth_std_recent = np.std(recent5)
            else:
                depth_std_recent = 0.0

            moving = (abs(velocity) > 5.0) or (depth_std_recent > 1.5)

            static_flag = False
            static_mean = sig
            if len(depth_hist) >= STD_STATIC_WIN:
                recent = np.array(list(depth_hist)[-STD_STATIC_WIN:])
                static_std  = np.std(recent)
                static_mean = float(np.mean(recent))
                if static_std < STD_STATIC_TH and static_mean > STATIC_DEPTH_MIN:
                    static_flag = True

            # ==== 检测“深度是否在物理合理区间” ====
            depth_invalid = not (DEPTH_VALID_MIN_MM <= depth_mm_corr <= DEPTH_VALID_MAX_MM)
            if depth_invalid:
                invalid_frames += 1
            else:
                invalid_frames = 0

            # ================== 视觉状态机：OK / SOFT / HARD / INVALID ==================
            if vision_state == VISION_OK:
                if invalid_frames >= INVALID_FRAMES_TH:
                    vision_state = VISION_INVALID
                    soft_occ_frames = 0
                    send_event(
                        "occlusion",
                        conf=float(conf_ema),
                        depth=float(depth_mm_corr),
                        reason="invalid_depth"
                    )
                else:
                    if conf_ema < CONF_SOFT_TH:
                        soft_occ_frames += 1
                    else:
                        soft_occ_frames = 0

                    if soft_occ_frames >= SOFT_FRAMES_TH:
                        vision_state = VISION_SOFT
                        send_event("occlusion", conf=float(conf_ema), depth=float(sig))

                    if static_flag:
                        vision_state = VISION_SOFT
                        soft_occ_frames = 0
                        send_event("occlusion_static", conf=float(conf_ema), depth=static_mean)

            elif vision_state == VISION_SOFT:
                if invalid_frames >= INVALID_FRAMES_TH:
                    vision_state = VISION_INVALID
                    soft_occ_frames = 0
                    send_event(
                        "occlusion",
                        conf=float(conf_ema),
                        depth=float(depth_mm_corr),
                        reason="invalid_depth"
                    )
                else:
                    if conf_ema > CONF_CLEAR_TH and not static_flag and not depth_invalid:
                        vision_state   = VISION_OK
                        soft_occ_frames = 0
                        invalid_frames  = 0
                        send_event("occlusion_clear", conf=float(conf_ema), depth=float(sig))

            elif vision_state == VISION_HARD:
                if invalid_frames >= INVALID_FRAMES_TH:
                    vision_state = VISION_INVALID
                    send_event(
                        "occlusion",
                        conf=float(conf_ema),
                        depth=float(depth_mm_corr),
                        reason="invalid_after_hard"
                    )
                elif conf_ema > CONF_CLEAR_TH and not static_flag and not depth_invalid:
                    vision_state   = VISION_OK
                    soft_occ_frames = 0
                    invalid_frames  = 0
                    send_event("occlusion_clear", conf=float(conf_ema), depth=float(sig))

            elif vision_state == VISION_INVALID:
                if not depth_invalid and conf_ema > CONF_CLEAR_TH and not static_flag:
                    invalid_frames  = 0
                    vision_state    = VISION_OK
                    soft_occ_frames = 0
                    send_event("occlusion_clear", conf=float(conf_ema), depth=float(sig))

            # ================== 峰值检测（原逻辑不改） ==================
            if sig <= ARM_THRESH_MM or (not armed and (now_ms - last_peak_ms) > AUTO_REARM_MS):
                armed = True
                last_valley = sig
            elif armed:
                last_valley = min(last_valley, sig)

            if len(sig_hist) == 3:
                d_prev = sig_hist[1] - sig_hist[0]
                d_now  = sig_hist[2] - sig_hist[1]
                is_turning = (d_prev > 0) and (d_now <= 0)
                enough_interval = (now_ms - last_peak_ms) >= MIN_INTERVAL_MS
                peak_candidate = sig_hist[1]

                if (armed and is_turning and enough_interval and
                    (PEAK_MIN_MM < peak_candidate < PEAK_MAX_MM) and
                    (peak_candidate - last_valley >= PROM_MM)):

                    if (now_ms - last_peak_ms) < MERGE_WINDOW_MS:
                        if peak_candidate > last_peak_depth:
                            # send_event("peak", idx=peak_idx, depth=round(peak_candidate, 2))
                            send_event(
                                "peak",
                                idx=peak_idx,
                                depth=round(peak_candidate, 2),
                                vis_time_ms=timestamp_ms  # ★★★ 使用本帧的视觉时间戳
                            )
                            last_peak_depth = peak_candidate
                    else:
                        peak_idx += 1
                        send_event(
                                "peak",
                                idx=peak_idx,
                                depth=round(peak_candidate, 2),
                                vis_time_ms=timestamp_ms  # ★★★ 使用本帧的视觉时间戳
                            )
                        last_peak_ms = now_ms
                        last_peak_depth = peak_candidate
                        armed = False
                        last_valley = np.inf

        # ================== 有手但没有主手：疑似“他人手/异常手” ==================
        elif has_any_hand and not has_main_hand:
            no_hand_frames  = 0
            soft_occ_frames = 0
            invalid_frames += 1

            conf_ema = EMA_ALPHA_CONF * max_conf_any + (1.0 - EMA_ALPHA_CONF) * conf_ema

            if vision_state != VISION_INVALID and invalid_frames >= INVALID_FRAMES_TH:
                vision_state = VISION_INVALID
                send_event(
                    "occlusion",
                    conf=float(conf_ema),
                    depth=float(prev_filtered if prev_filtered is not None else 0.0),
                    reason="invalid_hand"
                )

            cv2.putText(frame, "INVALID HAND (other person?)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, f"vision_state={vision_state} conf_ema={conf_ema:.2f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.imshow(WINDOW_NAME, frame)

            # === 保存当前帧的估深值 & 状态到 CSV ===
            
            sig = depth_mm_kf  # 峰值检测用的信号

            writer.writerow([
                frame_idx,
                timestamp_ms,
                depth_mm_raw,
                depth_mm_ema,
                depth_mm_kf,
                depth_mm_corr,
                sig,
                float(conf),
                float(conf_ema),
                vision_state,
                bool(has_any_hand),
                bool(has_main_hand),
                float(velocity),
                float(motion),
                gain,
                ZERO_OFFSET_MM,
                cos_tilt
            ])
            frame_idx += 1

            if key == 27:
                break
            continue

        # ================== 完全无手帧：HARD 遮挡 ==================
        else:
            conf_ema = (1.0 - EMA_ALPHA_CONF) * conf_ema
            soft_occ_frames = 0
            invalid_frames  = 0
            no_hand_frames += 1

            if no_hand_frames == NO_HAND_TH and vision_state != VISION_HARD:
                vision_state = VISION_HARD
                send_event(
                    "occlusion_lost",
                    conf=0.0,
                    depth=float(prev_filtered if prev_filtered is not None else 0.0)
                )

            if no_hand_frames > NO_HAND_TH and no_hand_frames % 30 == 0:
                send_event(
                    "occlusion_lost",
                    conf=0.0,
                    depth=float(prev_filtered if prev_filtered is not None else 0.0)
                )

            cv2.putText(frame, "NO HAND DETECTED",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            if key == 27:
                break
            continue

        # ---------- 调试显示 ----------
        depth_display = depth_mm_kf

        cv2.putText(frame, f"Depth={depth_display:.2f}mm Conf={conf:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"armed={armed} valley={last_valley:.1f} peak={last_peak_depth:.1f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"vision_state={vision_state} conf_ema={conf_ema:.2f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        writer.writerow([
            frame_idx, timestamp_ms,
            depth_mm_raw, depth_mm_ema, depth_mm_kf, depth_mm_corr,
            beta, gamma_boost, velocity, motion,
            gain, ZERO_OFFSET_MM, cos_tilt
        ])
        frame_idx += 1

        if key == 27:
            break
        cv2.imshow(WINDOW_NAME, frame)

logf.close()
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Data saved to {LOG_PATH}")
