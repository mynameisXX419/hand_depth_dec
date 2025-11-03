import cv2
import numpy as np

def calibrate_charuco_extrinsics(
    camera_matrix, dist_coeffs,
    squares_x=12, squares_y=9,
    square_length=0.015, marker_length=0.01125,
    dictionary_id=cv2.aruco.DICT_5X5_100,
    cam_id=0
):
    """
    在线标定外参（R, T） —— Charuco 棋盘检测
    输出评价指标：
        1️⃣ 重投影误差 RMS (px)
        2️⃣ 旋转矩阵正交误差
        3️⃣ 棋盘平面法向与Z轴夹角 (°)
    """

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    # 旧API（OpenCV<=4.8）
    charuco_board = cv2.aruco.CharucoBoard_create(
        squares_x, squares_y, square_length, marker_length, aruco_dict
    )
    detector = cv2.aruco.CharucoDetector(charuco_board)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    print("\n=== Charuco 外参标定开始 ===")
    print("请将标靶平放于世界坐标系原点平面，按 ENTER / 空格 确认，ESC 退出。\n")

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
                charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, None, None
            )

            if retval:
                cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                cv2.putText(vis, "Pose detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
                cv2.imshow("Charuco Extrinsic Calibration", vis)
                key = cv2.waitKey(1) & 0xFF

                if key in [13, 32]:  # Enter / Space
                    R, _ = cv2.Rodrigues(rvec)

                    # ✅ === 计算标定质量指标 ===
                    # 1️⃣ 重投影误差
                    obj_points = charuco_board.chessboardCorners[charuco_ids.flatten()]
                    img_points_proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
                    reproj_err = cv2.norm(charuco_corners, img_points_proj, cv2.NORM_L2) / len(img_points_proj)

                    # 2️⃣ 旋转矩阵正交误差
                    orth_err = np.linalg.norm(R @ R.T - np.eye(3))

                    # 3️⃣ 法向偏差角（世界Z轴夹角）
                    normal = R[:, 2]
                    z_axis = np.array([0, 0, 1])
                    angle_deg = np.degrees(np.arccos(np.clip(np.dot(normal, z_axis) / np.linalg.norm(normal), -1, 1)))

                    print("\n✅ 标定成功：")
                    print("Rotation Matrix R =\n", R)
                    print("Translation Vector T =\n", tvec.ravel())
                    print(f"\n📊 评价指标：")
                    print(f"  - 重投影误差 (RMS): {reproj_err:.3f} px")
                    print(f"  - 旋转矩阵正交误差: {orth_err:.2e}")
                    print(f"  - 平面法向与Z轴夹角: {angle_deg:.2f}°")

                    # ✅ 判断质量等级
                    if reproj_err < 0.5 and orth_err < 1e-3 and angle_deg < 2:
                        print("  ✅ 质量等级：优秀")
                    elif reproj_err < 0.8:
                        print("  ⚠️ 质量等级：一般，可接受")
                    else:
                        print("  ❌ 质量等级：较差，请重试")

                    cap.release()
                    cv2.destroyAllWindows()
                    return R, tvec

            else:
                cv2.putText(vis, "Detecting...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)
        cv2.imshow("Charuco Extrinsic Calibration", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[WARN] 未成功获取姿态，返回 None。")
    return None, None
