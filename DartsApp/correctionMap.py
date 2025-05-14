import numpy as np
import cv2
from calibration_tool import run_calibration_test

# ========== PARAMETRE ==========

# Cesta k obrázku terča (kalibračný terč v top pohľade)
image_path = "../Dartboard.png"  # Uprav podľa potreby

# Stred cieľa v 3D priestore (v milimetroch)
center_3d = np.array([19.10, -24.91, 414.20])

# Stred cieľa v 2D obrázku (v pixeloch)
center_px = np.array([502, 500])

# Okraj terča pre výpočet mierky (225 mm je polomer reálneho terča)
edge_px = np.array([990, 501])
pixel_radius = np.linalg.norm(edge_px - center_px)
scale = pixel_radius / 225  # px / mm

# ========== NAČÍTANIE OBRÁZKA ==========

image = cv2.imread(image_path)
if image is None:
    print(f"[ERROR] Obrázok '{image_path}' sa nepodarilo načítať.")
    exit()

# ========== NAČÍTANIE KAMIER A KALIBRAČNÝCH ÚDAJOV ==========

from calibration_data_loader import load_cameras_and_data

try:
    cameras, calibration_data, cams = load_cameras_and_data()
except Exception as e:
    print("[ERROR] Nepodarilo sa načítať kamery alebo kalibračné údaje.")
    print(e)
    exit()

# ========== SPUSTENIE KALIBRÁCIE ==========

run_calibration_test(
    cameras=cameras,
    calibration_data=calibration_data,
    cams=cams,
    original_target_image=image,
    center_3d=center_3d,
    center_px=center_px,
    scale=scale,
    output_dir="calibration_results"
)
