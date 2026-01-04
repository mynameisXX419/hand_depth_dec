#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import time

# ================== 基本配置 ==================
CAM_ID = 0
IMG_WIDTH  = 1280
IMG_HEIGHT = 960
NUM_IMAGES = 50

# 👉 改成你的“内参标定图片文件夹”
CALIB_IMG_DIR = "./camera_calib_images"

# 文件名前缀
IMG_PREFIX = "calib_"

# ================== 准备保存目录 ==================
os.makedirs(CALIB_IMG_DIR, exist_ok=True)
print(f"[INFO] Images will be saved to: {os.path.abspath(CALIB_IMG_DIR)}")

# ================== 打开摄像头 ==================
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("❌ 无法打开摄像头")

# 强制 YUYV
cap.set(cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'YUYV'))

# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT)

# 设置帧率（可选，防止驱动乱跳）
cap.set(cv2.CAP_PROP_FPS, 30)

# 再次读取，确认真实分辨率
ret, frame = cap.read()
if not ret:
    raise RuntimeError("❌ 无法读取相机帧")

h, w = frame.shape[:2]
print(f"[INFO] Camera resolution: {w} x {h}")

if w != IMG_WIDTH or h != IMG_HEIGHT:
    print("⚠ 警告：实际分辨率与期望不一致，但仍继续采集")

# ================== 采集循环 ==================
print("\n=== 相机标定图片采集 ===")
print("操作说明：")
print("  - 空格：保存一张")
print("  - ESC ：提前退出")
print(f"  - 目标数量：{NUM_IMAGES} 张\n")

count = 0
window_name = "Calibration Image Capture"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ 读取帧失败，跳过")
        continue

    # 显示提示文字
    disp = frame.copy()
    cv2.putText(disp,
                f"Saved: {count}/{NUM_IMAGES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2)

    cv2.putText(disp,
                "Press SPACE to capture, ESC to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    cv2.imshow(window_name, disp)
    key = cv2.waitKey(1) & 0xFF

    # ESC 退出
    if key == 27:
        print("\n[INFO] Capture aborted by user.")
        break

    # 空格保存
    if key == 32:
        img_name = f"{IMG_PREFIX}{count:02d}.png"
        img_path = os.path.join(CALIB_IMG_DIR, img_name)

        cv2.imwrite(img_path, frame)
        print(f"[SAVE] {img_path}")

        count += 1
        time.sleep(0.3)  # 防抖，避免连拍

        if count >= NUM_IMAGES:
            print("\n[INFO] Image capture completed.")
            break

# ================== 清理 ==================
cap.release()
cv2.destroyAllWindows()

print(f"\n✅ 共保存 {count} 张标定图片")
