# dec_depth_cap_main.py
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from camera_calib_loader import load_camera_params

# ================== 参数设置 ==================
CALIB_FILE = "camera_gp23.yml"      # 张正友法内参文件（Matlab导出的）
CAM_ID = 0
NUM_FRAMES = 40
SAVE_PATH = "extrinsic_result.yml"

# Charuco 参数（单位：米）
SQUARES_X     = 12
SQUARES_Y     = 9
SQUARE_LENGTH = 0.015     # 相邻 Charuco 交点间距（棋盘格边长）
MARKER_LENGTH = 0.01125
DICT_ID       = cv2.aruco.DICT_5X5_100

# ================== 加载内参 ==================
params = load_camera_params(CALIB_FILE)
K = params["K"].copy()
D = params["D"].copy()
print("=== 相机张正友法标定参数读取完成 ===")
print("内参矩阵 K:\n", K)
print("畸变系数 D:\n", D)

# 保证 D 形状为 (5,) 一维
D = D.reshape(-1)

# 估算“标定时图像分辨率”（因为 cx, cy ≈ 图像中心）
calib_width_est  = int(round(K[0, 2] * 2))   # ≈ 2 * cx
calib_height_est = int(round(K[1, 2] * 2))   # ≈ 2 * cy
print(f"\n估算标定使用的图像分辨率约为: {calib_width_est} x {calib_height_est}")

# ================== 打开相机并设置分辨率 ==================
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

# 优先尝试把相机设为与标定时一致的分辨率
if calib_width_est > 0 and calib_height_est > 0:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  calib_width_est)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calib_height_est)

# 读取实际分辨率
act_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
act_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n当前相机实际分辨率: {act_width} x {act_height}")

# ================== 如果实际分辨率和标定分辨率不一致 -> 缩放内参 ==================
if abs(act_width - calib_width_est) > 2 or abs(act_height - calib_height_est) > 2:
    print("\n⚠ 内参与当前分辨率不匹配，按比例缩放内参 K ...")
    sx = act_width  / calib_width_est
    sy = act_height / calib_height_est

    K[0, 0] *= sx      # fx
    K[0, 2] *= sx      # cx
    K[1, 1] *= sy      # fy
    K[1, 2] *= sy      # cy

    print("缩放后的内参矩阵 K:\n", K)
else:
    print("✅ 当前相机分辨率与内参匹配，无需缩放 K")

FY_PIX = K[1, 1]
print(f"\nFY_PIX = {FY_PIX:.2f}")

# ================== Charuco Board 初始化 ==================
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_ID)

# 兼容不同版本 OpenCV 的 CharucoBoard 创建
try:
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                                   SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
except Exception:
    board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y,
                                          SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)

# 创建检测器 (兼容新旧 API)
try:
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.CharucoDetector(board, detector_params)
except Exception:
    detector = cv2.aruco.CharucoDetector(board)

print("\n=== Charuco 外参标定开始 ===")
print("请将标靶平放于胸板平面，保持大致与胸板共面、距离 ~0.9m 左右")
print("按空格拍照采集，共 40 张；ESC 退出。\n")

rvecs, tvecs = [], []
all_charuco_corners, all_charuco_ids = [], []

# ================== 主循环采集 ==================
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
            cv2.putText(vis,
                        f"Pose OK ({len(rvecs)}/{NUM_FRAMES})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    cv2.imshow("Charuco Extrinsic Calibration", vis)
    key = cv2.waitKey(1) & 0xFF

    # 空格保存当前帧
    if key == 32 and 'retval' in locals() and retval and charuco_ids is not None and len(charuco_ids) > 3:
        rvecs.append(rvec)
        tvecs.append(tvec)
        all_charuco_corners.append(charuco_corners.copy())
        all_charuco_ids.append(charuco_ids.copy())

        pos = tvec.ravel() * 1000.0
        print(f"[{len(rvecs)}/{NUM_FRAMES}] 位置: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]mm")

        if len(rvecs) >= NUM_FRAMES:
            break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ================== 求平均外参 ==================
if len(rvecs) == 0:
    print("❌ 未采集到有效帧，标定失败")
    exit()

print("\n=== 开始计算外参标定指标 ===")

R_all, T_all = [], []
for rv, tv in zip(rvecs, tvecs):
    R, _ = cv2.Rodrigues(rv)
    R_all.append(R)
    T_all.append(tv)

R_all = np.stack(R_all, axis=0)   # (N,3,3)
T_all = np.stack(T_all, axis=0)   # (N,1,3) 或 (N,3,1)

R_mean = np.mean(R_all, axis=0)
# 正交化旋转矩阵
u, _, vt = np.linalg.svd(R_mean)
R_mean = u @ vt
T_mean = np.mean(T_all, axis=0)

# ================== 重投影误差 ==================
reproj_errs = []

for rv, tv, ch_pts, ch_ids in zip(rvecs, tvecs,
                                  all_charuco_corners,
                                  all_charuco_ids):
    ids = ch_ids.flatten().astype(int)
    img_pts = ch_pts.reshape(-1, 2)

    # Charuco 角点在棋盘平面上的 3D 坐标（Z=0）
    obj_pts = []
    for cid in ids:
        row = cid // SQUARES_X
        col = cid % SQUARES_X
        obj_pts.append([col * SQUARE_LENGTH, row * SQUARE_LENGTH, 0.0])
    obj_pts = np.array(obj_pts, np.float32)

    proj, _ = cv2.projectPoints(obj_pts, rv, tv, K, D)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(img_pts - proj, axis=1)
    reproj_errs.extend(err.tolist())

reproj_errs = np.array(reproj_errs)
print(f"平均重投影误差: {np.mean(reproj_errs):.3f}px")
print(f"最大误差:     {np.max(reproj_errs):.3f}px\n")

print("R =\n", R_mean)
print("T =", T_mean.ravel())

# ================== 保存到外参文件 ==================
fs = cv2.FileStorage(SAVE_PATH, cv2.FILE_STORAGE_WRITE)
fs.write("rotation_matrix", R_mean)
fs.write("translation_vector", T_mean)
fs.release()

print(f"\n📌 外参已保存: {SAVE_PATH}")
print("🎯 标定完成！")
