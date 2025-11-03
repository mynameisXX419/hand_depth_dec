# hand_dec/hand_dec.py
# ================== 手部按压深度（多点平面近似 + 手动置零多帧平均 + 光流补偿(物理域, 动态权重) + 卡尔曼 + 残差校正RC + 自适应动态补偿） ==================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 静音TFLite日志

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv, time, json
from camera_calib_loader import load_camera_params

# ------------------ 基本配置 ------------------
CALIB_FILE       = "camera_gp23.yml"
EXTRINSIC_FILE   = "extrinsic_result.yml"
CAM_ID           = 2
LOG_PATH         = "/home/ljy/project/hand_dec/ljy/ljy_1/ljyhand_depth_plane_avg1014_of_dyn.csv"

MAX_HANDS = 1
MODEL_COMPLEXITY = 1
HAND_BACK_KEYS   = [0, 5, 9, 13, 17]

ZERO_FRAMES      = 100
Z0_MM_DEFAULT    = 600.0
GAIN_FILE        = "press_gain.json"
RC_FILE          = "residual_correction.json"

# ------------------ 小工具 ------------------
def load_gain():
    if os.path.exists(GAIN_FILE):
        try:
            with open(GAIN_FILE, "r") as f:
                d = json.load(f)
                return float(d.get("gain", 1.0)), float(d.get("z0_mm", Z0_MM_DEFAULT)), float(d.get("offset_mm", 0.0))
        except: pass
    return 1.0, Z0_MM_DEFAULT, 0.0

def load_rc():
    if os.path.exists(RC_FILE):
        try:
            with open(RC_FILE, "r") as f:
                d = json.load(f)
                return float(d.get("a0", 0.0)), float(d.get("a1", 1.0)), float(d.get("a2", 0.0))
        except: pass
    return 0.0, 1.0, 0.0

def apply_rc_dynamic(depth_raw, depth_filtered, rc_params, beta, gamma_boost):
    a0, a1, a2 = rc_params
    return gamma_boost * (a0 + a1 * depth_filtered + a2 * depth_filtered ** 2 + beta * (depth_raw - depth_filtered))

# ------------------ EMA & KF ------------------
class EMA:
    def __init__(self, alpha=0.9): 
        self.a, self.v = alpha, None
    def update(self, x):
        self.v = x if self.v is None else self.a*x+(1-self.a)*self.v
        return self.v

class OneDimKalman:
    def __init__(self, dt=1/30.0,q_depth=20.0,q_vel=40.0,r_meas=3.0):
        self.dt=dt
        self.x=np.zeros((2,1)); self.P=np.eye(2)*100
        self.F=np.array([[1,dt],[0,1]]); self.H=np.array([[1,0]])
        self.Q=np.diag([q_depth,q_vel]); self.R=np.array([[r_meas]])
        self.init=False
    def predict(self):
        self.x=self.F@self.x
        self.P=self.F@self.P@self.F.T+self.Q
    def update(self,z):
        z=np.array([[float(z)]])
        if not self.init:
            self.x[0,0]=z[0,0]; self.init=True
        self.predict()
        y=z-self.H@self.x; S=self.H@self.P@self.H.T+self.R
        K=self.P@self.H.T@np.linalg.inv(S)
        self.x+=K@y
        self.P=(np.eye(2)-K@self.H)@self.P
        return float(self.x[0,0])

# ------------------ 相机参数 ------------------
params = load_camera_params(CALIB_FILE)
K, D = params["K"], params["D"].reshape(-1)
FY_PIX = float(K[1,1])

fs = cv2.FileStorage(EXTRINSIC_FILE, cv2.FILE_STORAGE_READ)
R = fs.getNode("rotation_matrix").mat(); T = fs.getNode("translation_vector").mat(); fs.release()
if R is None or T is None: raise RuntimeError("外参文件读取失败")

n_c = (R @ np.array([[0.0],[0.0],[1.0]])).reshape(3)
n_c /= (np.linalg.norm(n_c)+1e-9)
cos_tilt = abs(n_c[2]) if abs(n_c[2])>1e-3 else 1e-3

print(f"=== 外参加载完成 | FY={FY_PIX:.2f} | cosθ={cos_tilt:.3f} ===")

def pixel_to_world_depth_linear(dy_px, fy=FY_PIX, z0=Z0_MM_DEFAULT, cos_t=cos_tilt):
    return (dy_px * (z0/max(fy,1e-6))) / max(cos_t,1e-3)

# ------------------ 初始化 ------------------
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened(): raise RuntimeError("❌ 无法打开摄像头")

ema = EMA(0.9)
kf  = OneDimKalman()
gain, Z0_MM, ZERO_OFFSET_MM = load_gain()
rc_params = load_rc()

print(f"[INFO] Gain={gain:.3f}, Z0={Z0_MM:.1f}, Offset={ZERO_OFFSET_MM:.1f}")
print(f"[INFO] RC参数: a0={rc_params[0]:.3f}, a1={rc_params[1]:.3f}, a2={rc_params[2]:.5f}")

zero_ref_y=None; zero_buf=deque(maxlen=ZERO_FRAMES)
prev_gray=None; last_y=None; last_depth=None; frame_idx=0
MOTION_EPS=1e-6; prev_filtered=None

# ------------------ 日志 ------------------
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
writer = csv.writer(open(LOG_PATH,"w",newline=""))
writer.writerow([
    "frame_idx","timestamp","depth_raw_mm","depth_of_mm","depth_ema_mm","depth_kf_mm","depth_corr_mm",
    "beta","gamma_boost","velocity_mm_s","motion_mm","gain","offset_mm","cos_tilt"
])

print("\n>>> 开始：按压深度检测（多点平面平均 + 光流补偿(动态权重, 物理域) + RC动态补偿）")

# ------------------ 主循环 ------------------
with mp_hands.Hands(max_num_hands=MAX_HANDS,model_complexity=MODEL_COMPLEXITY,
                    min_detection_confidence=0.6,min_tracking_confidence=0.6) as hands:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        h,w=frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        raw_plane_y=None

        # ---------- 手部检测 ----------
        if res.multi_hand_landmarks:
            lm=res.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS)
            y_list=[lm.landmark[k].y*h for k in HAND_BACK_KEYS]
            raw_plane_y=np.average(y_list,weights=[1.5,1.2,1,1,1])

        key=cv2.waitKey(1)&0xFF

        # ---------- 自动零位 ----------
        if zero_ref_y is None:
            if raw_plane_y is not None:
                zero_buf.append(raw_plane_y)
                if len(zero_buf)>=ZERO_FRAMES and np.std(zero_buf)<0.5:
                    zero_ref_y=float(np.mean(zero_buf))
                    print(f"[INFO] 自动零位建立: {zero_ref_y:.2f}")
            cv2.putText(frame,f"Collecting zero: {len(zero_buf)}/{ZERO_FRAMES}",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.imshow("Hand Depth",frame); continue

        # ---------- 深度计算 ----------
        if raw_plane_y is not None:
            dy_px=raw_plane_y-zero_ref_y
            depth_mm_raw=pixel_to_world_depth_linear(dy_px,FY_PIX,Z0_MM,cos_tilt)*gain+ZERO_OFFSET_MM

            # ---------- 光流法补偿(物理域, 动态权重) ----------
            if prev_gray is not None and last_y is not None and last_depth is not None:
                flow=cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
                dy_flow=np.mean(flow[int(last_y)-10:int(last_y)+10, w//3:2*w//3,1])
                dy_mm=pixel_to_world_depth_linear(dy_flow,FY_PIX,Z0_MM,cos_tilt)
                motion = abs(depth_mm_raw - last_depth)
                w_dyn = np.clip(0.3 + 0.4 * (motion / 30.0), 0.3, 0.9)  # 动态权重: 压深越大, 原始占比越高
                depth_of_mm = (1 - w_dyn) * (last_depth + dy_mm) + w_dyn * depth_mm_raw
            else:
                depth_of_mm = depth_mm_raw

            # ---------- EMA / KF / RC ----------
            depth_mm_ema=ema.update(depth_of_mm)
            depth_mm_kf=kf.update(depth_mm_ema)
            velocity=float(kf.x[1,0])
            motion=abs(depth_mm_raw-(prev_filtered if prev_filtered is not None else depth_mm_kf))
            prev_filtered=depth_mm_kf

            beta=0.20+0.0015*abs(velocity)+0.20*(motion/(abs(depth_mm_raw)+MOTION_EPS))
            gamma_boost=1.05+0.0006*abs(velocity)+0.02*(motion/(abs(depth_mm_raw)+MOTION_EPS))
            beta=float(np.clip(beta,0.20,0.40))
            gamma_boost=float(np.clip(gamma_boost,1.03,1.10))

            depth_mm_corr=apply_rc_dynamic(depth_mm_raw,depth_mm_kf,rc_params,beta,gamma_boost)

            cv2.putText(frame,f"Depth={depth_mm_corr:.2f}mm",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            writer.writerow([frame_idx,time.time(),depth_mm_raw,depth_of_mm,
                             depth_mm_ema,depth_mm_kf,depth_mm_corr,
                             beta,gamma_boost,velocity,motion,
                             gain,ZERO_OFFSET_MM,cos_tilt])

            frame_idx+=1; last_y=raw_plane_y; last_depth=depth_of_mm
            prev_gray=gray.copy()

        if key==27: break
        cv2.imshow("Hand Depth - OpticalFlow(Dynamic)+RC",frame)

cap.release(); cv2.destroyAllWindows()
print(f"[INFO] Data saved to {LOG_PATH}")
