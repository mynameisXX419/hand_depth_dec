#!/usr/bin/env python3
# =========================================================
# Visual CPR Compression Depth
# Engineering version with Fake Occlusion (SAFE)
# =========================================================

import cv2
import mediapipe as mp  # Ensure mediapipe is imported properly
import numpy as np
from collections import deque
import time, json, socket, csv, os
from camera_calib_loader import load_camera_params
from typing import List

# ======================= Basic config =======================
WINDOW_NAME = "Hand Depth Monitor"

CALIB_FILE     = "camera_gp23.yml"
EXTRINSIC_FILE = "extrinsic_result.yml"
CAM_ID         = 0

SOCKET_PATH = "/tmp/press_event.sock"
LOG_PATH    = "./hand_depth_plane_avg.csv"

MAX_HANDS = 2
MODEL_COMPLEXITY = 1
HAND_BACK_KEYS = [0, 5, 9, 13, 17]
ZERO_FRAMES = 100

# ======================= Fake occlusion =====================
FAKE_OCCLUSION_ENABLE   = True
FAKE_OCCLUSION_AFTER_S  = 25.0
FAKE_OCCLUSION_LEN_S    = 5.0

# ======================= Peak params ========================
PEAK_MIN_MM     = 8.0
PEAK_MAX_MM     = 70.0
PROM_MM         = 3.5
MIN_INTERVAL_MS = 220
AUTO_REARM_MS   = 400
ARM_THRESH_MM   = 10.0

# ======================= Vision states ======================
VISION_OK   = "ok"
VISION_SOFT = "soft"
VISION_HARD = "hard"

CONF_SOFT_TH = 0.35
NO_HAND_TH   = 3

# ======================= Socket =============================
def wait_socket_ready(path):
    print(f"[INFO] waiting for socket ready: {path}")
    while not os.path.exists(path):
        time.sleep(0.05)
    print(f"[INFO] socket ready")

def send_event(event_type, **kwargs):
    evt = {"type": event_type, **kwargs}
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.sendto(json.dumps(evt).encode(), SOCKET_PATH)
        s.close()
    except Exception as e:
        print("[WARN] send_event failed:", e)

# ======================= Filters =============================
class EMA:
    def __init__(self, alpha=0.6):
        self.a = alpha
        self.v = None
    def update(self, x):
        self.v = x if self.v is None else self.a * x + (1 - self.a) * self.v
        return self.v

class OneDimKalman:
    def __init__(self, dt=1/30):
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 100
        self.F = np.array([[1, dt], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.diag([20, 40])
        self.R = np.array([[3]])
        self.inited = False
    def update(self, z):
        z = np.array([[float(z)]])
        if not self.inited:
            self.x[0, 0] = z[0, 0]
            self.inited = True
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0, 0])

# ======================= Camera calib =======================
params = load_camera_params(CALIB_FILE)
K = params["K"]
FY = float(K[1, 1])

fs = cv2.FileStorage(EXTRINSIC_FILE, cv2.FILE_STORAGE_READ)
R_oc = fs.getNode("rotation_matrix").mat()
T_oc = fs.getNode("translation_vector").mat()
fs.release()

Z0_MM = float(T_oc[2, 0] * 1000.0)
n_c = (R_oc @ np.array([[0.0],[0.0],[1.0]])).reshape(3)
n_c /= np.linalg.norm(n_c)
cos_tilt = abs(n_c[2])

def pixel_to_mm(dy_px):
    return (dy_px * Z0_MM / FY) * cos_tilt

# ======================= Init ================================
ema = EMA()
kf  = OneDimKalman()

zero_ref_y = None
zero_buf = deque(maxlen=ZERO_FRAMES)

sig_hist = deque(maxlen=3)
armed = True
last_valley = np.inf
last_peak_ms = 0
peak_idx = 0

vision_state = VISION_OK
conf_ema = 0.0
no_hand_frames = 0

sig = 0.0
frame_idx = 0

start_ts_ms = None
fake_occ_active = False
fake_occ_last = False

# ======================= CSV ================================
logf = open(LOG_PATH, "w", newline="")
writer = csv.writer(logf)
writer.writerow([
    "frame_idx", "timestamp_ms",
    "depth_raw_mm", "depth_ema_mm", "depth_kf_mm",
    "depth_corr_mm", "sig_mm",
    "conf", "conf_ema", "vision_state",
    "has_hand", "has_main_hand",
    "velocity_mm_s", "motion_mm",
    "gain", "offset_mm", "cos_tilt"
])

# ======================= Camera =============================
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Initialize mp_hands properly
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

wait_socket_ready(SOCKET_PATH)

# ======================= Main loop ==========================
with mp_hands.Hands(False, MAX_HANDS, MODEL_COMPLEXITY, 0.6, 0.6) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int(time.time() * 1000)
        if start_ts_ms is None:
            start_ts_ms = timestamp_ms

        elapsed_s = (timestamp_ms - start_ts_ms) / 1000.0
        fake_occ_active = (
            FAKE_OCCLUSION_ENABLE and
            FAKE_OCCLUSION_AFTER_S <= elapsed_s <
            FAKE_OCCLUSION_AFTER_S + FAKE_OCCLUSION_LEN_S
        )

        if fake_occ_active and not fake_occ_last:
            send_event("occlusion", vision_state=VISION_HARD)
        if not fake_occ_active and fake_occ_last:
            send_event("occlusion_clear")
        fake_occ_last = fake_occ_active

        frame = cv2.flip(frame, 1)
        h, _ = frame.shape[:2]

        raw_plane_y = None
        conf = 0.0
        has_any_hand = False
        has_main_hand = False

        if not fake_occ_active:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            has_any_hand = bool(res.multi_hand_landmarks)
            if has_any_hand:
                lm = res.multi_hand_landmarks[0]
                raw_plane_y = np.mean([lm.landmark[k].y * h for k in HAND_BACK_KEYS])
                conf = res.multi_handedness[0].classification[0].score
                has_main_hand = True
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # ---------- ZERO ----------
        if zero_ref_y is None:
            if raw_plane_y is not None:
                zero_buf.append(raw_plane_y)
                if len(zero_buf) == ZERO_FRAMES and np.std(zero_buf) < 1.0:
                    zero_ref_y = float(np.mean(zero_buf))
            cv2.putText(frame, "STATE: ZERO CALIBRATION", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # ---------- DEPTH ----------
        if has_main_hand:
            dy_px = raw_plane_y - zero_ref_y
            depth_raw = pixel_to_mm(dy_px)
            depth_ema = ema.update(depth_raw)
            depth_kf  = kf.update(depth_ema)
            sig = depth_kf
            sig_hist.append(sig)
            conf_ema = 0.6 * conf + 0.4 * conf_ema
            no_hand_frames = 0
        else:
            no_hand_frames += 1
            conf_ema *= 0.4
            depth_raw = 0.0
            depth_ema = 0.0
            depth_kf = 0.0
            sig = 0.0

        # Write the current data to CSV, including the default values when no hand is detected
        writer.writerow([
            frame_idx,
            timestamp_ms,
            depth_raw if has_main_hand else 0.0,
            depth_ema if has_main_hand else 0.0,
            depth_kf if has_main_hand else 0.0,
            depth_kf if has_main_hand else 0.0,  # depth_corr_mm
            sig,
            float(conf),
            float(conf_ema),
            vision_state,
            bool(has_any_hand),
            bool(has_main_hand),
            0.0,  # placeholder for velocity (to be calculated if needed)
            0.0,  # placeholder for motion (to be calculated if needed)
            1.0,  # placeholder for gain
            0.0,  # placeholder for offset_mm
            cos_tilt
        ])

        # ---------- FSM ----------
        prev_state = vision_state
        if no_hand_frames >= NO_HAND_TH:
            vision_state = VISION_HARD
        elif conf_ema < CONF_SOFT_TH:
            vision_state = VISION_SOFT
        else:
            vision_state = VISION_OK

        if vision_state != prev_state and not fake_occ_active:
            if vision_state == VISION_HARD:
                send_event("occlusion", vision_state=vision_state)
            elif vision_state == VISION_OK:
                send_event("occlusion_clear")

        # ---------- PEAK ----------
        if has_main_hand:
            now_ms = timestamp_ms
            if sig <= ARM_THRESH_MM or (not armed and now_ms - last_peak_ms > AUTO_REARM_MS):
                armed = True
                last_valley = sig
            elif armed:
                last_valley = min(last_valley, sig)

            if len(sig_hist) == 3:
                d1 = sig_hist[1] - sig_hist[0]
                d2 = sig_hist[2] - sig_hist[1]
                if (armed and d1 > 0 and d2 <= 0):
                    peak = sig_hist[1]
                    amp = peak - last_valley
                    if PEAK_MIN_MM < amp < PEAK_MAX_MM and amp >= PROM_MM and now_ms - last_peak_ms >= MIN_INTERVAL_MS:
                        peak_idx += 1
                        last_peak_ms = now_ms
                        armed = False
                        send_event("peak", idx=peak_idx,
                                   depth=round(amp,2),
                                   vis_time_ms=timestamp_ms)

        # ---------- Overlay ----------
        cv2.putText(frame, f"Depth={sig:.1f} mm", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"state={vision_state} conf_ema={conf_ema:.2f}",
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == 27:
            break

# ======================= Cleanup =============================
logf.close()
cap.release()
cv2.destroyAllWindows()
print("[INFO] exit")
