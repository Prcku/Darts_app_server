from calibration import MultiCameraCalibrator

calibrator = MultiCameraCalibrator(
    camera_indices=[3, 1, 2],
    frame_size=(1000, 1000),
    checkerboard_size=(8, 6),
    square_size=27.0  # mm
)

calibrator.run_calibration_pipeline()