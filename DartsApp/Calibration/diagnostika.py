import cv2
import numpy as np
import yaml
import glob
import os
import matplotlib.pyplot as plt

# === CESTY ===
left_img_folder = "./calibration1000/images/right/"
middle_img_folder = "./calibration1000/images/middle/"
stereo_yaml_path = "./calibration1000/calib_data/stereo_middle_to_right_calibration.yaml"

# === CESTY K SÚBOROM KALIBRÁCIE ===
calibration_files = [
    "./calibration1000/calib_data/right_calibration.yaml",
    "./calibration1000/calib_data/middle_calibration.yaml",
]

# === NAHRANIE PARAMETROV ===
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

pattern_size = (8, 6)  # počet vnútorných rohov šachovnice

# === NAHRANIE STEREO KALIBRÁCIE ===
with open(stereo_yaml_path, "r") as f:
    stereo_data = yaml.safe_load(f)

with open(calibration_files[0], 'r') as f:
    calib = yaml.safe_load(f)

with open(calibration_files[1], 'r') as f:
    calib1 = yaml.safe_load(f)

K1 = np.array(calib["camera_matrix"])
D1 = np.array(calib["distortion_coefficients"][0])
K2 = np.array(calib1["camera_matrix"])
D2 = np.array(calib1["distortion_coefficients"][0])
R = np.array(stereo_data["rotation_matrix"])
T = np.array(stereo_data["translation_vector"]).reshape((3, 1))

print("ahoj")
# === ZÍSKANIE OBRÁZKOV ===
left_images = sorted(glob.glob(os.path.join(left_img_folder, "stereo_right*.png")))
middle_images = sorted(glob.glob(os.path.join(middle_img_folder, "stereo_right*.png")))
print(left_images)
assert len(left_images) == len(middle_images), "Počet obrázkov vľavo a v strede sa nezhoduje!"

# === PREJDI VŠETKY PÁRY ===
for i, (left_path, middle_path) in enumerate(zip(left_images, middle_images)):
    img1 = cv2.imread(left_path)
    img2 = cv2.imread(middle_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size)
    ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size)

    if not (ret1 and ret2):
        print(f"[WARN] Nenašli sa rohy v páre {i}")
        continue

    # Korekcia rohov (presnosť)
    corners1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    corners2 = cv2.cornerSubPix(gray2, corners2, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # === VÝPOČET epipolárnych čiar pre img1 (ľavý) podľa bodov z img2 ===
    F, _ = cv2.findFundamentalMat(corners1, corners2, cv2.FM_8POINT)

    lines1 = cv2.computeCorrespondEpilines(corners2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    img1_copy = img1.copy()
    for r, pt1 in zip(lines1, corners1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0]*img1.shape[1]) / r[1]])
        pt = tuple(pt1.ravel().astype(int))
        cv2.line(img1_copy, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1_copy, pt, 4, color, -1)

    # === ZOBRAZENIE ===
    plt.figure(figsize=(10, 5))
    plt.title(f"Epipolárne čiary – pár {i}")
    plt.imshow(cv2.cvtColor(img1_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
