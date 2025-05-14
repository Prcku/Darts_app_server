import cv2
import numpy as np
import yaml
import os

def load_calibration(file_path, image_size=(1000, 1000)):
    """
    Načíta kalibračný YAML súbor a pripraví všetky potrebné parametre vrátane remapovacích máp.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
    distortion_coeffs = np.array(data['distortion_coefficients'], dtype=np.float32)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, image_size, 0, image_size)

    if new_camera_matrix is None:
        print("[WARNING] Použitá pôvodná camera_matrix")
        new_camera_matrix = camera_matrix

    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, new_camera_matrix, image_size, cv2.CV_32FC1)

    return [camera_matrix, distortion_coeffs, new_camera_matrix, mapx, mapy]

def load_cameras_and_data():
    """
    Načíta všetky kalibračné parametre a stereo extrinsiky pre tri kamery.
    Vracia:
        calibration_data - zoznam [K, D, K_opt, mapx, mapy] pre každú kameru
        extrinsics_data - zoznam [R, T] pre každú kameru (0 = left, 1 = middle, 2 = right)
        cams - indexy zariadení pre VideoCapture
    """
    calibration_files = [
        "./Calibration/calibration1000/calib_data/left_calibration.yaml",
        "./Calibration/calibration1000/calib_data/middle_calibration.yaml",
        "./Calibration/calibration1000/calib_data/right_calibration.yaml"
    ]

    stereo_files = [
        "./Calibration/calibration1000/calib_data/stereo_middle_to_left_calibration.yaml",
        "./Calibration/calibration1000/calib_data/stereo_middle_to_right_calibration.yaml"
    ]

    image_size = (1000, 1000)

    # Načítaj vnútornú kalibráciu a mapy
    calibration_data = [load_calibration(path, image_size) for path in calibration_files]

    # Načítaj stereo transformácie
    with open(stereo_files[0], 'r') as f:
        left_stereo = yaml.safe_load(f)
    R_left = np.array(left_stereo['rotation_matrix'], dtype=np.float32)
    T_left = np.array(left_stereo['translation_vector'], dtype=np.float32).reshape(3, 1)

    with open(stereo_files[1], 'r') as f:
        right_stereo = yaml.safe_load(f)
    R_right = np.array(right_stereo['rotation_matrix'], dtype=np.float32)
    T_right = np.array(right_stereo['translation_vector'], dtype=np.float32).reshape(3, 1)

    # Usporiadaj extrinsiky tak, aby middle bola referenčná (0,0,0)
    extrinsics_data = [
        [R_left, T_left],
        [np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)],
        [R_right, T_right]
    ]

    cams = [0, 1, 2]  # indexy video zariadení

    return calibration_data, extrinsics_data, cams