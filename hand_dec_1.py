# hand_dec/hand_dec.py
# ================== 手部按压深度（多点平面近似 + 手动置零多帧平均 + 偏移补偿 + 倾角修正
# + 一键增益校准 + 稳定性检测 + 卡尔曼 + 残差校正RC + 自适应动态补偿） ==================

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv, time, os, json
from camera_calib_loader import load_camera_params

# ------------------ 基本配置 ------------------
CALIB_FILE       = "camera_gp23.yml"
EXTRINSIC_FILE   = "extrinsic_result.yml"
CAM_ID           = 2
LOG_PATH         = "/home/ljy/project/hand_dec/ljy/ljy_1/hand_depth_plane_avg.csv"

MAX_HANDS = 1
MODEL_COMPLEXITY = 1
HAND_BACK_KEYS   = [0, 5, 9, 13, 17]

ZERO_FRAMES      = 100
SHOW_FPS         = True

Z0_MM_DEFAULT    = 600.0
GAIN_FILE        = "press_gain.json"
PRESS_KNOWN_MM   = 100.0
KEY_CALIB        = ord('c')
KEY_ZERO         = ord('z')
ZERO_AVG_FRAMES  = 50
ZERO_OFFSET_MM   = 0.0

USE_BACKPROJ     = False

# ------------------ 小工具 ------------------
def load_gain():
    if os.path.exists(GAIN_FILE):
        try:
            with open(GAIN_FILE, "r") as f:
                data = json.load(f)
                gain = float(data.get("gain", 1.0))
                z0_mm = float(data.get("z0_mm", Z0_MM_DEFAULT))
                offset = float(data.get("offset_mm", ZERO_OFFSET_MM))
                return gain, z0_mm, offset
        except:
            pass
    return 1.0, Z0_MM_DEFAULT, ZERO_OFFSET_MM

def save_gain(gain, z0_mm, offset):
    with open(GAIN_FILE, "w") as f:
        json.dump({"gain": float(gain), "z0_mm": float(z0_mm), "offset_mm": float(offset)}, f, indent=2)

# ------------------ 残差校正 + 动态补偿 ------------------
RC_FILE = "residual_correction.json"

def load_rc():
    if os.path.exists(RC_FILE):
        try:
            with open(RC_FILE, "r") as f:
                data = json.load(f)
                a0 = float(data.get("a0", 0.0))
                a1 = float(data.get("a1", 1.0))
                a2 = float(data.get("a2", 0.0))
                return a0, a1, a2
        except:
            pass
    return 0.0, 1.0, 0.0

def save_rc(a0, a1, a2):
    with open(RC_FILE, "w") as f:
        json.dump({"a0": a0, "a1": a1, "a2": a2}, f, indent=2)

def apply_rc_dynamic(depth_raw, depth_filtered, rc_params, beta, gamma_boost):
    a0, a1, a2 = rc_params
    static_part = a0 + a1 * depth_filtered + a2 * (depth_filtered ** 2)
    dynamic_part = beta * (depth_raw - depth_filtered)
    return gamma_boost * (static_part + dynamic_part)

# ------------------ 平滑与滤波 ------------------
class EMA:
    def __init__(self, alpha=0.9):
        self.a = alpha
        self.v = None
    def update(self, x):
        self.v = x if self.v is None else (self.a * x + (1 - self.a) * self.v)
        return self.v

class OneDimKalman:
    def __init__(self, dt=1/30.0, q_depth=3.0, q_vel=15.0, r_meas=6.0):
        self.dt = dt
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 100.0
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.array([[q_depth, 0.0], [0.0, q_vel]])
        self.R = np.array([[r_meas]])
        self.inited = False
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        z = np.array([[float(z)]])
        if not self.inited:
            self.x[0, 0] = z[0, 0]
            self.inited = True
        self.predict()
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        return float(self.x[0, 0])

# ------------------ 载入相机参数 ------------------
params = load_camera_params(CALIB_FILE)
K, D = params["K"], params["D"].reshape(-1)
FY_PIX = float(K[1, 1])

fs = cv2.FileStorage(EXTRINSIC_FILE, cv2.FILE_STORAGE_READ)
R_oc = fs.getNode("rotation_matrix").mat()
T_oc = fs.getNode("translation_vector").mat()
fs.release()
if R_oc is None or T_oc is None:
    raise RuntimeError("外参文件读取失败")

n_c = (R_oc @ np.array([[0.0],[0.0],[1.0]])).reshape(3)
n_c = n_c / (np.linalg.norm(n_c) + 1e-9)
cos_tilt = abs(n_c[2])
if cos_tilt < 1e-3: cos_tilt = 1e-3

print("=== 相机/外参加载完成 ===")
print(f"FY_PIX={FY_PIX:.2f}, cos_tilt={cos_tilt:.3f}")

px2mm = Z0_MM_DEFAULT / max(FY_PIX, 1e-6)
STABLE_THRESH_MM = 0.5
STABLE_THRESH_PX = STABLE_THRESH_MM / px2mm

def pixel_to_world_depth_linear(dy_px, fy=FY_PIX, z0=Z0_MM_DEFAULT, cos_t=cos_tilt):
    depth_mm = (dy_px * (z0 / max(fy, 1e-6))) / max(cos_t, 1e-3)
    return depth_mm

# ------------------ 初始化 ------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

ema = EMA(0.9)
kf  = OneDimKalman(dt=1/30.0, q_depth=20.0, q_vel=40.0, r_meas=3.0)
gain, Z0_MM, ZERO_OFFSET_MM = load_gain()
rc_params = load_rc()

print(f"[INFO] 初始 Z0={Z0_MM:.1f}mm, 增益 gain={gain:.3f}, offset={ZERO_OFFSET_MM:.1f}mm")
print(f"[INFO] 残差校正参数 a0={rc_params[0]:.3f}, a1={rc_params[1]:.3f}, a2={rc_params[2]:.6f}")
print("按 'c' 校准100mm, 按 'z' 手动置零, ESC退出")

zero_ref_y  = None
zero_buf    = deque(maxlen=ZERO_FRAMES)
frame_times = deque(maxlen=30)
detect_cnt = total_cnt = 0
last_tick = cv2.getTickCount()
frame_idx = 0

prev_filtered = None
MOTION_EPS    = 1e-6

# ✅ 修改部分：日志字段更丰富
log_dir = os.path.dirname(LOG_PATH)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"[INFO] 创建日志目录: {log_dir}")

logf = open(LOG_PATH, "w", newline="")
writer = csv.writer(logf)
writer.writerow([
    "frame_idx", "timestamp",
    "depth_raw_mm", "depth_ema_mm", "depth_kf_mm", "depth_corr_mm",
    "beta", "gamma_boost", "velocity_mm_s", "motion_mm",
    "gain", "offset_mm", "cos_tilt"
])

print("\n>>> 开始：按压深度检测（多点平面平均 + RC自适应动态补偿 + 全量日志）")

# ------------------ 主循环 ------------------
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        total_cnt += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        raw_plane_y = None

        if res.multi_hand_landmarks:
            detect_cnt += 1
            lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            y_list = [lm.landmark[k].y * h for k in HAND_BACK_KEYS]
            weights = np.array([1.5,1.2,1.0,1.0,1.0])
            raw_plane_y = np.average(y_list, weights=weights)

        key = cv2.waitKey(1) & 0xFF

        # 自动零位建立
        if zero_ref_y is None:
            if raw_plane_y is not None:
                zero_buf.append(raw_plane_y)
                if len(zero_buf) >= ZERO_FRAMES:
                    y_std = np.std(zero_buf)
                    if y_std < STABLE_THRESH_PX:
                        zero_ref_y = float(np.mean(zero_buf))
                        print(f"[INFO] 自动零位建立: {zero_ref_y:.2f}")
                    else:
                        zero_buf.clear()
            cv2.putText(frame, f"Collecting zero: {len(zero_buf)}/{ZERO_FRAMES}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        elif raw_plane_y is not None:
            dy_px = raw_plane_y - zero_ref_y
            depth_mm_no_gain = pixel_to_world_depth_linear(dy_px, FY_PIX, Z0_MM, cos_tilt)
            depth_mm_raw = depth_mm_no_gain * gain + ZERO_OFFSET_MM

            depth_mm_ema = ema.update(depth_mm_raw)
            depth_mm_kf  = kf.update(depth_mm_ema)

            velocity = float(kf.x[1,0]) if hasattr(kf, "x") else 0.0
            v_abs    = abs(velocity)
            motion = abs(depth_mm_raw - (prev_filtered if prev_filtered is not None else depth_mm_kf))
            prev_filtered = depth_mm_kf

            beta = 0.20 + 0.0015 * v_abs + 0.20 * (motion / (abs(depth_mm_raw) + MOTION_EPS))
            beta = float(np.clip(beta, 0.20, 0.40))
            gamma_boost = 1.05 + 0.0006 * v_abs + 0.02 * (motion / (abs(depth_mm_raw) + MOTION_EPS))
            gamma_boost = float(np.clip(gamma_boost, 1.03, 1.10))

            depth_mm_corr = apply_rc_dynamic(depth_mm_raw, depth_mm_kf, rc_params, beta, gamma_boost)

            # ✅ 新增部分：深按动态补偿（3.5cm以上线性放大，封顶6cm）
            # ✅ 新增部分：深按动态补偿（30~55mm区间立方骤增补偿，封顶60mm）
            if depth_mm_corr > 30.0:
                if depth_mm_corr < 55.0:
                    # 使用立方函数增强补偿力度（越深补得越多）
                    # 补偿 = 3 + ((depth-30)/25)^3 * 3  -> 从 +3mm 到 +6mm 非线性增长
                    x = (depth_mm_corr - 30.0) / 25.0  # 0~1
                    add_mm = 3.0 + (x ** 3) * 3.0
                    depth_mm_corr = depth_mm_corr + add_mm
                else:
                    # 超过55mm上限封顶，不超过60mm
                    depth_mm_corr = min(depth_mm_corr + 6.0, 60.0)



            # 显示
            cv2.putText(frame, f"Depth={depth_mm_corr:.2f}mm", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # ✅ 写入完整日志
            writer.writerow([
                frame_idx, int(time.time() * 1000),
                depth_mm_raw, depth_mm_ema, depth_mm_kf, depth_mm_corr,
                beta, gamma_boost, velocity, motion,
                gain, ZERO_OFFSET_MM, cos_tilt
            ])
            frame_idx += 1

        if key == 27: break
        cv2.imshow("Hand Depth - Plane Avg (RC + adaptive dynamic boost)", frame)

logf.close()
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Data saved to {LOG_PATH}")
