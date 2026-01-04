# hand_dec/hand_dec.py
# ================== 手部关键点采集脚本（保存带关键点的图像） ==================
# 功能：
#   - 打开指定相机实时捕获视频流
#   - 检测手部关键点并绘制在图像上
#   - 保存“带关键点的图像”到指定文件夹
# 保存目录：
#   ./raw_frames/hand_000000.jpg, hand_000010.jpg, ...

import cv2
import mediapipe as mp
import os
import time

# ------------------ 基本配置 ------------------
CAM_ID = 2
SAVE_DIR = "./raw_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_INTERVAL = 10   # 每隔多少帧保存一次
MAX_HANDS = 1
MODEL_COMPLEXITY = 1

# ------------------ 初始化 MediaPipe Hands ------------------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ------------------ 打开摄像头 ------------------
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开摄像头，请检查CAM_ID是否正确")

print(f"✅ 摄像头已打开，保存目录: {SAVE_DIR}")
print("按 ESC 键退出")

frame_idx = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=MODEL_COMPLEXITY,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 读取帧失败，已退出")
            break

        # 翻转图像（镜像显示）
        frame = cv2.flip(frame, 1)

        # 检测手部关键点
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # ✅ 每隔固定帧保存一次“带关键点”的图像
        if frame_idx % SAVE_INTERVAL == 0:
            img_name = f"hand_{frame_idx:06d}.jpg"
            img_path = os.path.join(SAVE_DIR, img_name)
            cv2.imwrite(img_path, frame)

        cv2.imshow("Hand Keypoint Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 退出
            break

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ 共保存 {frame_idx // SAVE_INTERVAL} 张带关键点图像至 {SAVE_DIR}")
