#dec_depth_cap_main.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from camera_calib_loader import load_camera_params

# ================== 参数设置 ==================
CALIB_FILE = "camera_gp23.yml"
CAM_ID = 2
NUM_FRAMES = 40
SAVE_PATH = "extrinsic_result.yml"

# Charuco参数
SQUARES_X = 12
SQUARES_Y = 9
SQUARE_LENGTH = 0.015
MARKER_LENGTH = 0.01125
DICT_ID = cv2.aruco.DICT_5X5_100

# ================== 初始化 ==================
params = load_camera_params(CALIB_FILE)
K, D = params["K"], params["D"]
print(f"FY_PIX = {K[1,1]:.2f}")

aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)

# 兼容不同版本的OpenCV
try:
    # OpenCV 4.9+ 新版本API
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
except TypeError:
    try:
        # OpenCV 4.7-4.8 中间版本API
        board = cv2.aruco.CharucoBoard.create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    except AttributeError:
        # OpenCV 4.6 及以下版本API
        board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

# 创建检测器
try:
    # 新版本需要DetectorParameters
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, detector_params)
except:
    # 旧版本API
    detector = cv2.aruco.CharucoDetector(board)

cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

print("\n=== Charuco 外参标定开始 ===")
print(f"内参文件: {CALIB_FILE}")
print("请将标靶平放于目标平面，按空格拍照采集，目标累计 40 张。\n")

rvecs, tvecs = [], []
retval = False

# ================== 主循环 ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    vis = frame.copy()
    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    if charuco_ids is not None and len(charuco_ids) > 3:
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, K, D, None, None
        )
        if retval:
            cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.05)
            cv2.putText(vis, f"Pose OK ({len(rvecs)}/{NUM_FRAMES})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Charuco Extrinsic Calibration", vis)
    key = cv2.waitKey(1) & 0xFF

    if key == 32 and retval:  # 空格保存
        rvecs.append(rvec)
        tvecs.append(tvec)
        
        # 计算当前帧的欧拉角和位置信息
        R_current, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R_current[0,0] * R_current[0,0] + R_current[1,0] * R_current[1,0])
        if sy > 1e-6:
            x = np.arctan2(R_current[2,1], R_current[2,2])
            y = np.arctan2(-R_current[2,0], sy)
            z = np.arctan2(R_current[1,0], R_current[0,0])
        else:
            x = np.arctan2(-R_current[1,2], R_current[1,1])
            y = np.arctan2(-R_current[2,0], sy)
            z = 0
        
        euler_deg = [np.degrees(x), np.degrees(y), np.degrees(z)]
        position_mm = tvec.ravel() * 1000  # 转换为mm
        
        print(f"[{len(rvecs)}/{NUM_FRAMES}] 帧已采集 - 位置: [{position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}]mm, "
              f"旋转: [{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°]")
        
        if len(rvecs) >= NUM_FRAMES:
            break
    elif key == 27:  # ESC退出
        break

cap.release()
cv2.destroyAllWindows()

# ================== 求平均外参 ==================
if len(rvecs) > 0:
    print("\n=== 开始计算外参标定指标 ===")
    
    R_all, T_all = [], []
    rvecs_array = np.array(rvecs).squeeze()
    tvecs_array = np.array(tvecs).squeeze()
    
    for rv, tv in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rv)
        R_all.append(R)
        T_all.append(tv)
    
    R_mean = np.mean(np.stack(R_all), axis=0)
    T_mean = np.mean(np.stack(T_all), axis=0)

    # 正交化旋转矩阵
    u, _, vt = np.linalg.svd(R_mean)
    R_mean = np.dot(u, vt)

    # ================== 计算标定指标 ==================
    
    # 1. 旋转向量和平移向量的标准差
    rvec_std = np.std(rvecs_array, axis=0)
    tvec_std = np.std(tvecs_array, axis=0)
    
    # 2. 欧拉角标准差 (更直观)
    euler_angles = []
    for rv in rvecs:
        R, _ = cv2.Rodrigues(rv)
        # 计算欧拉角 (ZYX顺序)
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        euler_angles.append([np.degrees(x), np.degrees(y), np.degrees(z)])
    
    euler_angles = np.array(euler_angles)
    euler_std = np.std(euler_angles, axis=0)
    
    # 3. 位置和方向的变化范围
    position_range = np.ptp(tvecs_array, axis=0)  # peak-to-peak (max-min)
    rotation_range = np.ptp(euler_angles, axis=0)
    
    # 4. 计算重投影误差（如果有板子的角点信息）
    reprojection_errors = []
    all_corners_data = []  # 存储每帧的角点数据用于重投影
    
    print("\n✅ 外参标定完成！")
    print("=" * 60)
    
    # 打印基本结果
    print("📍 标定结果:")
    print("平均旋转矩阵 R =")
    print(R_mean)
    print("平均平移向量 T =", T_mean.ravel())
    
    # 打印标定指标
    print("\n📊 标定质量指标:")
    print(f"📏 采集帧数: {len(rvecs)} 帧")
    
    print("\n🔄 旋转稳定性:")
    print(f"   旋转向量标准差: [{rvec_std[0]:.6f}, {rvec_std[1]:.6f}, {rvec_std[2]:.6f}] (rad)")
    print(f"   欧拉角标准差:   [{euler_std[0]:.3f}°, {euler_std[1]:.3f}°, {euler_std[2]:.3f}°]")
    print(f"   旋转角度变化范围: [{rotation_range[0]:.3f}°, {rotation_range[1]:.3f}°, {rotation_range[2]:.3f}°]")
    
    print("\n📍 平移稳定性:")
    print(f"   平移向量标准差: [{tvec_std[0]:.6f}, {tvec_std[1]:.6f}, {tvec_std[2]:.6f}] (m)")
    print(f"   位置变化范围:   [{position_range[0]*1000:.2f}, {position_range[1]*1000:.2f}, {position_range[2]*1000:.2f}] (mm)")
    
    # 质量评估
    print("\n🎯 标定质量评估:")
    # 判断旋转稳定性 (欧拉角标准差)
    rotation_quality = "优秀" if max(euler_std) < 0.5 else "良好" if max(euler_std) < 1.0 else "一般" if max(euler_std) < 2.0 else "较差"
    print(f"   旋转稳定性: {rotation_quality} (最大角度标准差: {max(euler_std):.3f}°)")
    
    # 判断平移稳定性 (mm为单位)
    translation_quality = "优秀" if max(position_range)*1000 < 1.0 else "良好" if max(position_range)*1000 < 2.0 else "一般" if max(position_range)*1000 < 5.0 else "较差"
    print(f"   平移稳定性: {translation_quality} (最大位置变化: {max(position_range)*1000:.2f}mm)")
    
    # 综合评估
    overall_quality = "优秀" if rotation_quality in ["优秀"] and translation_quality in ["优秀", "良好"] else \
                     "良好" if rotation_quality in ["优秀", "良好"] and translation_quality in ["优秀", "良好", "一般"] else \
                     "一般" if rotation_quality in ["优秀", "良好", "一般"] else "较差"
    print(f"   📋 综合质量: {overall_quality}")
    
    print("=" * 60)

    fs = cv2.FileStorage(SAVE_PATH, cv2.FILE_STORAGE_WRITE)
    fs.write("rotation_matrix", R_mean)
    fs.write("translation_vector", T_mean)
    fs.release()
    print(f"已保存到 {SAVE_PATH}")
else:
    print("❌ 未采集到有效帧，外参标定失败。")
