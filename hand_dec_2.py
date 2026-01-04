# ================== 手部按压深度检测（全面放宽阈值版：快速节奏友好） ==================
# 功能：
#  - 峰值检测：导数拐点 + 不应期(220ms) + 自动武装(400ms) + prominence(3.5mm) + 去重(120ms)
#  - 遮挡检测：低置信度 / 静止 / 丢失
#  - 输出事件到 /tmp/press_event.sock

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv, time, os, json, socket
from camera_calib_loader import load_camera_params

# ------------------ 基本配置 ------------------
CALIB_FILE       = "camera_gp23.yml"
EXTRINSIC_FILE   = "extrinsic_result.yml"
CAM_ID           = 2
LOG_PATH         = "/home/ljy/project/hand_dec/ljy/ljy_1/hand_depth_plane_avg.csv"
SOCKET_PATH      = "/tmp/press_event.sock"

MAX_HANDS = 1
MODEL_COMPLEXITY = 1
HAND_BACK_KEYS   = [0, 5, 9, 13, 17]
ZERO_FRAMES      = 100

Z0_MM_DEFAULT    = 523.1
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
MIN_INTERVAL_MS  = 220      # 不应期缩短（更快节奏）
ARM_THRESH_MM    = 10.0     # 武装阈值
PROM_MM          = 3.5      # 峰谷差阈值（更灵敏）
MERGE_WINDOW_MS  = 120      # 合并窗口
AUTO_REARM_MS    = 400      # 自动重新武装时间
STD_STATIC_WIN   = 10
STD_STATIC_TH    = 1.0

# ------------------ Socket事件广播 ------------------
def send_event(event_type, **kwargs):
    event = {"type": event_type, "timestamp": time.time(), **kwargs}
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
        except: pass
    return 1.0, Z0_MM_DEFAULT, ZERO_OFFSET_MM

def load_rc():
    if os.path.exists(RC_FILE):
        try:
            with open(RC_FILE, "r") as f:
                d = json.load(f)
                return float(d.get("a0",0)), float(d.get("a1",1)), float(d.get("a2",0))
        except: pass
    return 0.0, 1.0, 0.0

def apply_rc_dynamic(depth_raw, depth_filtered, rc_params, beta, gamma_boost):
    a0,a1,a2 = rc_params
    static_part  = a0 + a1*depth_filtered + a2*(depth_filtered**2)
    dynamic_part = beta*(depth_raw - depth_filtered)
    return gamma_boost*(static_part + dynamic_part)

# ------------------ 滤波类 ------------------
class EMA:
    def __init__(self, alpha=0.9): self.a,self.v = alpha,None
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
    def predict(self): self.x=self.F@self.x; self.P=self.F@self.P@self.F.T+self.Q
    def update(self,z):
        z=np.array([[float(z)]])
        if not self.inited: self.x[0,0]=z[0,0]; self.inited=True
        self.predict(); y=z-self.H@self.x; S=self.H@self.P@self.H.T+self.R
        K=self.P@self.H.T@np.linalg.inv(S)
        self.x+=K@y; I=np.eye(2); self.P=(I-K@self.H)@self.P
        return float(self.x[0,0])

# ------------------ 相机参数 ------------------
params = load_camera_params(CALIB_FILE)
K,D=params["K"],params["D"].reshape(-1)
FY_PIX=float(K[1,1])
fs=cv2.FileStorage(EXTRINSIC_FILE,cv2.FILE_STORAGE_READ)
R_oc=fs.getNode("rotation_matrix").mat(); T_oc=fs.getNode("translation_vector").mat(); fs.release()
if R_oc is None or T_oc is None: raise RuntimeError("外参文件读取失败")
n_c=(R_oc@np.array([[0.0],[0.0],[1.0]])).reshape(3)
n_c=n_c/(np.linalg.norm(n_c)+1e-9)
cos_tilt=abs(n_c[2])
px2mm=Z0_MM_DEFAULT/max(FY_PIX,1e-6)
STABLE_THRESH_PX=STABLE_THRESH_MM/px2mm
def pixel_to_world_depth_linear(dy_px,fy=FY_PIX,z0=Z0_MM_DEFAULT,cos_t=cos_tilt):
    return (dy_px*(z0/max(fy,1e-6)))/max(cos_t,1e-3)

# ------------------ 初始化 ------------------
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(CAM_ID)
if not cap.isOpened(): raise RuntimeError("无法打开摄像头")

ema=EMA(0.9); kf=OneDimKalman(dt=1/30,q_depth=20,q_vel=40,r_meas=3)
gain,Z0_MM,ZERO_OFFSET_MM=load_gain(); rc_params=load_rc()
zero_ref_y=None; zero_buf=deque(maxlen=ZERO_FRAMES)

# 状态变量
sig_hist = deque(maxlen=3)
depth_hist = deque(maxlen=60)
frame_idx = 0
peak_idx = 0
prev_filtered=None
MOTION_EPS=1e-6
conf_low_count=0
last_occluded=False

armed=True
last_valley=np.inf
last_peak_ms=0
last_peak_depth=0.0

os.makedirs(os.path.dirname(LOG_PATH),exist_ok=True)
logf=open(LOG_PATH,"w",newline=""); writer=csv.writer(logf)
writer.writerow(["frame_idx","timestamp","depth_raw_mm","depth_ema_mm","depth_kf_mm","depth_corr_mm",
                 "beta","gamma_boost","velocity_mm_s","motion_mm","gain","offset_mm","cos_tilt"])

print("\n>>> 开始：按压深度检测 + 峰值/遮挡事件广播\n")

# ------------------ 主循环 ------------------
with mp_hands.Hands(static_image_mode=False,max_num_hands=MAX_HANDS,
                    model_complexity=MODEL_COMPLEXITY,
                    min_detection_confidence=0.6,min_tracking_confidence=0.6) as hands:
    while True:
        ok,frame=cap.read()
        if not ok: break
        frame=cv2.flip(frame,1); h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=hands.process(rgb)
        raw_plane_y=None; conf=0.0; has_hand=bool(res.multi_hand_landmarks)

        if has_hand:
            lm=res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame,lm,mp_hands.HAND_CONNECTIONS)
            y_list=[lm.landmark[k].y*h for k in HAND_BACK_KEYS]
            weights=np.array([1.5,1.2,1.0,1.0,1.0])
            raw_plane_y=np.average(y_list,weights=weights)
            conf=res.multi_handedness[0].classification[0].score if res.multi_handedness else 0.0

        key=cv2.waitKey(1)&0xFF

        # ---------- 零位 ----------
        if zero_ref_y is None:
            if raw_plane_y is not None:
                zero_buf.append(raw_plane_y)
                if len(zero_buf)>=ZERO_FRAMES:
                    y_std=np.std(zero_buf)
                    if y_std<STABLE_THRESH_PX: zero_ref_y=float(np.mean(zero_buf))
                    else: zero_buf.clear()
            cv2.putText(frame,f"Collecting zero: {len(zero_buf)}/{ZERO_FRAMES}",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.imshow("Hand Depth - Plane Avg",frame)
            if key==27: break
            continue

        if raw_plane_y is not None:
            # ================= 有手帧 =================
            dy_px=raw_plane_y-zero_ref_y
            depth_mm_raw=pixel_to_world_depth_linear(dy_px,FY_PIX,Z0_MM,cos_tilt)*gain+ZERO_OFFSET_MM
            depth_mm_ema=ema.update(depth_mm_raw)
            depth_mm_kf=kf.update(depth_mm_ema)
            velocity=float(kf.x[1,0]); v_abs=abs(velocity)
            motion=abs(depth_mm_raw-(prev_filtered if prev_filtered is not None else depth_mm_kf))
            prev_filtered=depth_mm_kf
            # ✅ 改进版残差+动态补偿（与新版 hand_dec.py 一致）
            beta = np.clip(
                0.12 + 0.0008 * v_abs + 0.08 * (motion / (abs(depth_mm_raw) + MOTION_EPS)),
                0.12, 0.25
            )
            gamma_boost = np.clip(
                1.015 + 0.0002 * v_abs + 0.006 * (motion / (abs(depth_mm_raw) + MOTION_EPS)),
                1.015, 1.05
            )

            RC_GAIN = 0.55  # 混合权重，防止过度补偿（0.4~0.6 之间可调）

            depth_rc = apply_rc_dynamic(depth_mm_raw, depth_mm_kf, rc_params, beta, gamma_boost)
            depth_mm_corr = depth_mm_kf + RC_GAIN * (depth_rc - depth_mm_kf)


            sig = depth_mm_corr
            sig_hist.append(sig)
            depth_hist.append(sig)

            now_ms = int(time.time()*1000)

            # ✅ 施密特触发 + 自动武装（放宽）
            if sig <= ARM_THRESH_MM or (not armed and (now_ms - last_peak_ms) > AUTO_REARM_MS):
                armed = True
                last_valley = sig
            elif armed:
                last_valley = min(last_valley, sig)

            # ✅ 峰值检测
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
                            send_event("peak", idx=peak_idx, depth=round(peak_candidate,2))
                            last_peak_depth = peak_candidate
                    else:
                        peak_idx += 1
                        send_event("peak", idx=peak_idx, depth=round(peak_candidate,2))
                        last_peak_ms = now_ms
                        last_peak_depth = peak_candidate
                        armed = False
                        last_valley = np.inf

            # ---- 遮挡检测 ----
            if conf<CONF_THRESH:
                conf_low_count+=1
            else:
                conf_low_count=0
                if last_occluded:
                    send_event("occlusion_clear",conf=conf,depth=sig)
                    last_occluded=False
            if conf_low_count>=OCCLUSION_FRAMES and not last_occluded:
                send_event("occlusion",conf=conf,depth=sig)
                last_occluded=True

            # ✅ 静止检测
            if len(depth_hist)>=STD_STATIC_WIN:
                recent=np.array(list(depth_hist)[-STD_STATIC_WIN:])
                if np.std(recent)<STD_STATIC_TH and not last_occluded:
                    send_event("occlusion_static",conf=conf,depth=float(np.mean(recent)))
                    last_occluded=True

        else:
            # ================= 无手帧 =================
            conf_low_count+=1
            if conf_low_count>=3 and not last_occluded:
                send_event("occlusion_lost",conf=0.0,depth=prev_filtered if prev_filtered else 0.0)
                last_occluded=True
            if conf_low_count%30==0:
                send_event("occlusion_lost",conf=0.0,depth=prev_filtered if prev_filtered else 0.0)
            cv2.putText(frame,"NO HAND DETECTED",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
            cv2.imshow("Hand Depth - Plane Avg (RC + Peak/Occlusion)",frame)
            if key==27: break
            continue

        # ---------- 调试显示 ----------
        depth_display = depth_mm_corr

        # ✅ 新增部分：仅用于显示的深按动态补偿（非逻辑修正）
        #   - 30~55mm区间非线性放大显示（立方增强）
        #   - 封顶显示不超过60mm
        if depth_display > 30.0:
            if depth_display < 55.0:
                x = (depth_display - 30.0) / 25.0  # 0~1
                add_mm = 3.0 + (x ** 3) * 3.0      # +3 ~ +6 mm
                depth_display = depth_display + add_mm
            else:
                depth_display = min(depth_display + 6.0, 60.0)

        # ✅ 仅显示修正值，不写入日志、不影响峰值
        cv2.putText(frame, f"Depth={depth_display:.2f}mm Conf={conf:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"armed={armed} valley={last_valley:.1f} peak={last_peak_depth:.1f}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


        writer.writerow([frame_idx,int(time.time()*1000),depth_mm_raw,depth_mm_ema,depth_mm_kf,
                         depth_mm_corr,beta,gamma_boost,velocity,motion,
                         gain,ZERO_OFFSET_MM,cos_tilt])
        frame_idx+=1

        if key==27: break
        cv2.imshow("Hand Depth - Plane Avg (RC + Peak/Occlusion)",frame)

logf.close(); cap.release(); cv2.destroyAllWindows()
print(f"[INFO] Data saved to {LOG_PATH}")
