#!/usr/bin/env python3
# =========================================================
# Visual CPR Compression Depth
# Save hand_depth_plane_avg.csv (engineering version)
# =========================================================

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time, json, socket, csv, os
from camera_calib_loader import load_camera_params
from typing import List

# =========================================================
# Basic config
# =========================================================
WINDOW_NAME = "Hand Depth Monitor"

CALIB_FILE     = "camera_gp23.yml"
EXTRINSIC_FILE = "extrinsic_result.yml"
CAM_ID         = 2

SOCKET_PATH = "/tmp/press_event.sock"
LOG_PATH    = "./hand_depth_plane_avg.csv"

MAX_HANDS = 2
MODEL_COMPLEXITY = 0
HAND_BACK_KEYS = [0, 5, 9, 13, 17]
ZERO_FRAMES = 100

# =========================================================
# Peak detection parameters (unchanged)
# =========================================================
PEAK_MIN_MM     = 8.0
PEAK_MAX_MM     = 70.0
PROM_MM         = 3.5
MIN_INTERVAL_MS = 220
AUTO_REARM_MS   = 400
ARM_THRESH_MM   = 10.0

# =========================================================
# Vision state thresholds
# =========================================================
CONF_SOFT_TH       = 0.35
NO_HAND_TH         = 3
INVALID_FRAMES_TH  = 3
STATIC_DEPTH_MIN   = 5.0
STD_STATIC_WIN     = 10
STD_STATIC_TH      = 1.0

DEPTH_VALID_MIN_MM = -10.0
DEPTH_VALID_MAX_MM = 70.0

# =========================================================
# Vision states
# =========================================================
VISION_OK      = "ok"
VISION_SOFT    = "soft"
VISION_HARD    = "hard"
VISION_INVALID = "invalid"

# =========================================================
# Socket helper (unchanged)
# =========================================================
def send_event(event_type, **kwargs):
    evt = {"type": event_type, **kwargs}
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        s.sendto(json.dumps(evt).encode(), SOCKET_PATH)
        s.close()
    except Exception:
        pass

# =========================================================
# Filters (unchanged)
# =========================================================
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

# =========================================================
# Camera & calibration
# =========================================================
params = load_camera_params(CALIB_FILE)
K = params["K"]
FY = float(K[1, 1])

fs = cv2.FileStorage(EXTRINSIC_FILE, cv2.FILE_STORAGE_READ)
R_oc = fs.getNode("rotation_matrix").mat()
T_oc = fs.getNode("translation_vector").mat()
fs.release()

Z0_MM = float(T_oc[2, 0] * 1000.0)

n_c = (R_oc @ np.array([[0.0], [0.0], [1.0]])).reshape(3)
n_c /= np.linalg.norm(n_c)
cos_tilt = abs(n_c[2])

def pixel_to_mm(dy_px):
    return (dy_px * Z0_MM / FY) #* cos_tilt

# =========================================================
# Init
# =========================================================
ema = EMA()
kf  = OneDimKalman()

zero_ref_y = None
zero_buf = deque(maxlen=ZERO_FRAMES)

sig_hist   = deque(maxlen=3)
depth_hist = deque(maxlen=60)

armed = True
last_valley = np.inf
last_peak_ms = 0
peak_idx = 0

vision_state = VISION_OK
conf_ema = 0.0
no_hand_frames = 0
invalid_frames = 0

sig = 0.0
frame_idx = 0

# ===== ZERO strict control =====
ZERO_STD_TH = 1.0
last_zero_std = None
zero_fail_count = 0

# =========================================================
# CSV init
# =========================================================
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
logf = open(LOG_PATH, "w", newline="")
writer = csv.writer(logf)

# ===== CSV HEADER (exactly as you required) =====
writer.writerow([
    "frame_idx",
    "timestamp_ms",
    "depth_raw_mm",
    "depth_ema_mm",
    "depth_kf_mm",
    "depth_corr_mm",
    "sig_mm",
    "conf",
    "conf_ema",
    "vision_state",
    "has_hand",
    "has_main_hand",
    "velocity_mm_s",
    "motion_mm",
    "gain",
    "offset_mm",
    "cos_tilt"
])

print(f"[INFO] Logging to {LOG_PATH}")

# =========================================================
# Camera
# =========================================================
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# =========================================================
# Main loop
# =========================================================
with mp_hands.Hands(False, MAX_HANDS, MODEL_COMPLEXITY, 0.6, 0.6) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int(time.time() * 1000)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        raw_plane_y = None
        conf = 0.0
        has_any_hand = bool(res.multi_hand_landmarks)
        has_main_hand = False

        if has_any_hand:
            lm = res.multi_hand_landmarks[0]
            y_list = [lm.landmark[k].y * h for k in HAND_BACK_KEYS]
            raw_plane_y = float(np.mean(y_list))
            if res.multi_handedness:
                conf = res.multi_handedness[0].classification[0].score
            has_main_hand = True
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # =====================================================
        # ZERO STAGE (strict)
        # =====================================================
        if zero_ref_y is None:
            if raw_plane_y is not None:
                zero_buf.append(raw_plane_y)
                if len(zero_buf) == ZERO_FRAMES:
                    last_zero_std = float(np.std(zero_buf))
                    if last_zero_std < ZERO_STD_TH:
                        zero_ref_y = float(np.mean(zero_buf))
                    else:
                        zero_fail_count += 1
                        zero_buf.clear()

            cv2.putText(frame, "STATE: ZERO CALIBRATION",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            cv2.putText(frame, f"Collected: {len(zero_buf)}/{ZERO_FRAMES}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            if last_zero_std is not None:
                cv2.putText(frame, f"std={last_zero_std:.3f}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # =====================================================
        # DEPTH STAGE
        # =====================================================
        depth_invalid = False
        static_flag = False

        if has_main_hand:
            dy_px = raw_plane_y - zero_ref_y
            depth_raw = pixel_to_mm(dy_px)
            depth_ema = ema.update(depth_raw)
            depth_kf  = kf.update(depth_ema)

            sig = depth_kf
            sig_hist.append(sig)
            depth_hist.append(sig)

            if not (DEPTH_VALID_MIN_MM <= depth_kf <= DEPTH_VALID_MAX_MM):
                invalid_frames += 1
                depth_invalid = True
            else:
                invalid_frames = 0

            if len(depth_hist) >= STD_STATIC_WIN:
                recent = np.array(depth_hist)[-STD_STATIC_WIN:]
                if np.std(recent) < STD_STATIC_TH and np.mean(recent) > STATIC_DEPTH_MIN:
                    static_flag = True

            conf_ema = 0.6 * conf + 0.4 * conf_ema
            no_hand_frames = 0
        else:
            no_hand_frames += 1
            conf_ema *= 0.4

        prev_state = vision_state

        if no_hand_frames >= NO_HAND_TH:
            vision_state = VISION_HARD
        elif depth_invalid and invalid_frames >= INVALID_FRAMES_TH:
            vision_state = VISION_INVALID
        elif conf_ema < CONF_SOFT_TH or static_flag:
            vision_state = VISION_SOFT
        else:
            vision_state = VISION_OK

        if vision_state != prev_state:
            if vision_state in (VISION_HARD, VISION_INVALID):
                send_event("occlusion", vision_state=vision_state)
            elif vision_state == VISION_OK:
                send_event("occlusion_clear")

        # =====================================================
        # Peak detection (unchanged)
        # =====================================================
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
                is_peak = (d1 > 0 and d2 <= 0)
                peak_candidate = sig_hist[1]
                peak_amp = peak_candidate #- last_valley

                if (armed and is_peak and
                    PEAK_MIN_MM < peak_amp < PEAK_MAX_MM and
                    peak_amp >= PROM_MM and
                    now_ms - last_peak_ms >= MIN_INTERVAL_MS):

                    peak_idx += 1
                    last_peak_ms = now_ms
                    armed = False
                    last_valley = np.inf

                    send_event("peak",
                               idx=peak_idx,
                               depth=round(peak_amp, 2),
                               vis_time_ms=timestamp_ms)

        # =====================================================
        # CSV WRITE (THIS IS WHAT YOU ASKED FOR)
        # =====================================================
        velocity = float(kf.x[1, 0]) if kf.inited else 0.0
        motion   = abs(depth_raw - depth_ema) if has_main_hand else 0.0

        writer.writerow([
            frame_idx,
            timestamp_ms,
            depth_raw if has_main_hand else 0.0,
            depth_ema if has_main_hand else 0.0,
            depth_kf  if has_main_hand else 0.0,
            depth_kf  if has_main_hand else 0.0,   # depth_corr_mm
            sig,
            float(conf),
            float(conf_ema),
            vision_state,
            bool(has_any_hand),
            bool(has_main_hand),
            velocity,
            motion,
            1.0,          # gain (placeholder)
            0.0,          # offset_mm
            cos_tilt
        ])
        frame_idx += 1

        # =====================================================
        # Overlay
        # =====================================================
        cv2.putText(frame, f"Depth={sig:.2f}mm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"vision_state={vision_state} conf_ema={conf_ema:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) == 27:
            break

# =========================================================
# Cleanup
# =========================================================
logf.close()
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Data saved to {LOG_PATH}")
