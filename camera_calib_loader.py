import cv2
import numpy as np

def load_camera_params(yaml_path: str):
    """
    读取张正友法标定文件（OpenCV YAML 格式）
    返回：camera_matrix, dist_coeffs, R, T
    """
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"无法打开标定文件: {yaml_path}")

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    rotation_matrix = fs.getNode("rotation_matrix").mat()
    translation_vector = fs.getNode("translation_vector").mat()
    fs.release()

    params = {
        "K": camera_matrix,
        "D": dist_coeffs,
        "R": rotation_matrix,
        "T": translation_vector
    }

    print("=== 相机张正友法标定参数读取完成 ===")
    print("内参矩阵 K:\n", camera_matrix)
    print("畸变系数 D:\n", dist_coeffs.ravel())
    # print("旋转矩阵 R:\n", rotation_matrix)
    # print("平移向量 T:\n", translation_vector.ravel())

    return params
