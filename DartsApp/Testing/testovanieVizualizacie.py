# import json
# import numpy as np
# import cv2
# 
# def load_json_config(json_path):
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     cameras = []
#     for cam_data, ext in zip(data["cameras"], data["extrinsics"]):
#         K = np.array(cam_data["K"], dtype=np.float32)
#         D = np.array(cam_data["D"][0], dtype=np.float32)
#         R = np.array(ext["R"], dtype=np.float32)
#         T = np.array(ext["T"], dtype=np.float32)
# 
#         rvec, _ = cv2.Rodrigues(R)
#         tvec = T.reshape((3, 1))
# 
#         cameras.append({
#             "id": cam_data["id"],
#             "K": K,
#             "D": D,
#             "rvec": rvec,
#             "tvec": tvec
#         })
#     points = [np.array(p["position"], dtype=np.float32) for p in data["triangulated_points"]]
#     return cameras, points
# 
# def project_point_cv(point_3d, camera):
#     point_3d = np.array(point_3d, dtype=np.float32).reshape((1, 1, 3))
#     image_points, _ = cv2.projectPoints(
#         point_3d,
#         camera["rvec"],
#         camera["tvec"],
#         camera["K"],
#         camera["D"]
#     )
#     return tuple(int(round(c)) for c in image_points[0][0])
# 
# def draw_points_on_frame(frame, points, camera, color=(0, 255, 0)):
#     for pt in points:
#         try:
#             x, y = project_point_cv(pt, camera)
#             cv2.circle(frame, (x, y), 5, color, -1)
#         except:
#             continue
#     return frame
# 
# def main():
#     # === CONFIG ===
#     json_path = "darts_3d_data.json"  # zmeň podľa potreby
#     camera_indices = [3, 1, 2]  # zmeniť ak treba
#     frame_size = (1000, 1000)
# 
#     # === LOAD CONFIG ===
#     cameras, triangulated_points = load_json_config(json_path)
# 
#     # === OPEN CAMERAS ===
#     video_captures = []
#     for idx in camera_indices:
#         cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
#         if not cap.isOpened():
#             print(f"[ERROR] Kamera {idx} sa nepodarilo otvoriť.")
#             return
#         video_captures.append(cap)
# 
#     # === READ FRAMES AND DRAW ===
#     for i, cap in enumerate(video_captures):
#         ret, frame = cap.read()
#         if not ret:
#             print(f"[ERROR] Kamera {i} nevytvorila snímku.")
#             continue
# 
#         frame_with_points = draw_points_on_frame(frame, triangulated_points, cameras[i])
#         cv2.imshow(f"Camera {i} - projected points", frame_with_points)
# 
#     # === CLEANUP ===
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     for cap in video_captures:
#         cap.release()
# 
# if __name__ == "__main__":
#     main()
# 
import cv2
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Wedge
from matplotlib.path import Path
import io
from PIL import Image
import copy
import math

def project_point(point_3d, center_3d, center_px, scale):
    
    relative = np.array(point_3d) - center_3d
    print(center_px[0], relative[0], relative[2] , scale)
    x_img = center_px[0] + relative[0] * scale
    y_img = center_px[1] - relative[2] * scale * 1.2
    return int(round(x_img)), int(round(y_img))

def draw_shot_positions(image, positions_list, center_3d, center_px, scale):
    # Vytvorenie kópie obrázku, aby sme nemodifikovali originál
    result_image = copy.deepcopy(image)
    
    # Farby pre jednotlivé šípky
    colors = [(255, 0, 0),(25, 0, 255),(255, 255,255),]
    
    # Vykreslenie každej šípky
    for i, position in enumerate(positions_list):
        if position is not None:
            x, y = project_point(position, center_3d, center_px, scale)
            cv2.circle(result_image, (x, y), 3, colors[i], -1)
            cv2.circle(result_image, (x, y), 5, (255, 255, 255), 2)
    
    return result_image

center_3d = np.array([19.10, -24.91, 414.20])  # Centrum terča v 3D priestore

center_px = np.array([502, 500])
edge_px = np.array([990, 501])
pixel_radius = np.linalg.norm(edge_px - center_px)
scale = pixel_radius / 225  # px per mm
dart_positions = [
    [30.78093032836914, -73.5782501220703, 497.5089111328125],
    [26.84093704223633, -90.3562240600586, 495.986181640625],
    [27.973385620117188, -77.06609954833985, 501.79078369140626]
]

image = cv2.imread("../../Dartboard.png")  # zmeň cestu ak treba

if image is None:
    print("[ERROR] Obrázok sa nepodarilo načítať. Skontroluj cestu.")
else:
    output_img = draw_shot_positions(image, dart_positions, center_3d, center_px, scale)

    # === ZOBRAZENIE ===
    cv2.imshow("Dartboard with projected darts", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()