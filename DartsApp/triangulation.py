def triangulate_shots(detected_darts):
    import cv2
    import numpy as np
    import yaml

    calibration_files = [
        "./Calibration/calibration1000/calib_data/left_calibration.yaml",
        "./Calibration/calibration1000/calib_data/middle_calibration.yaml",
        "./Calibration/calibration1000/calib_data/right_calibration.yaml"
    ]
    stereo_files = [
        "./Calibration/calibration1000/calib_data/stereo_middle_to_left_calibration.yaml",
        "./Calibration/calibration1000/calib_data/stereo_middle_to_right_calibration.yaml"
    ]

    cameras = []
    for idx, path in enumerate(calibration_files):
        with open(path, 'r') as f:
            calib = yaml.safe_load(f)
            cam = {
                'id': idx,
                'K': np.array(calib['camera_matrix']),
                'D': np.array(calib['distortion_coefficients'])
            }
            cameras.append(cam)

    with open(stereo_files[0]) as f:
        left_stereo = yaml.safe_load(f)
    R_left = np.array(left_stereo['rotation_matrix'])
    T_left = np.array(left_stereo['translation_vector'])

    with open(stereo_files[1]) as f:
        right_stereo = yaml.safe_load(f)
    R_right = np.array(right_stereo['rotation_matrix'])
    T_right = np.array(right_stereo['translation_vector'])

    extrinsics = [
        {'pos': (-R_left.T @ T_left).flatten(), 'R': R_left, 'T': T_left.reshape(-1, 1)},
        {'pos': np.zeros((3,)), 'R': np.eye(3), 'T': np.zeros((3, 1))},
        {'pos': (-R_right.T @ T_right).flatten(), 'R': R_right, 'T': T_right.reshape(-1, 1)}
    ]

    def projection_matrix(K, R, T):
        RT = np.hstack((R, T.reshape(-1, 1)))
        return K @ RT

    cam_points = detected_darts
    triangulated_points = []

    stereo_pairs = [(0, 1), (1, 2)]

    for id1, id2 in stereo_pairs:
        if id1 not in cam_points or id2 not in cam_points:
            continue

        pt1 = np.array([[cam_points[id1][0], cam_points[id1][1]]], dtype=np.float32)
        pt2 = np.array([[cam_points[id2][0], cam_points[id2][1]]], dtype=np.float32)

        pt1_ud = cv2.undistortPoints(pt1, cameras[id1]['K'], cameras[id1]['D'], P=cameras[id1]['K']).reshape(-1)
        pt2_ud = cv2.undistortPoints(pt2, cameras[id2]['K'], cameras[id2]['D'], P=cameras[id2]['K']).reshape(-1)

        P1 = projection_matrix(cameras[id1]['K'], extrinsics[id1]['R'], extrinsics[id1]['T'])
        P2 = projection_matrix(cameras[id2]['K'], extrinsics[id2]['R'], extrinsics[id2]['T'])

        pt1_h = np.array([[pt1_ud[0]], [pt1_ud[1]]])
        pt2_h = np.array([[pt2_ud[0]], [pt2_ud[1]]])

        point_4d = cv2.triangulatePoints(P1, P2, pt1_h, pt2_h)
        point_3d = point_4d[:3] / point_4d[3]
        triangulated_points.append(point_3d.flatten())

    if len(triangulated_points) != 2:
        return None

    x_estimate = np.mean([p[0] for p in triangulated_points])
    if x_estimate < 0:
        weights = [0.6, 0.4]
    elif x_estimate > 0:
        weights = [0.4, 0.6]
    else:
        weights = [0.5, 0.5]

    triangulated_points = np.array(triangulated_points)
    averaged_point = np.average(triangulated_points, axis=0, weights=weights)

    return averaged_point.tolist()