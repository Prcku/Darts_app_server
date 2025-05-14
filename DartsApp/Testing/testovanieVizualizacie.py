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

image = cv2.imread("../Dartboard1000.png")  # zmeň cestu ak treba

if image is None:
    print("[ERROR] Obrázok sa nepodarilo načítať. Skontroluj cestu.")
else:
    output_img = draw_shot_positions(image, dart_positions, center_3d, center_px, scale)

    # === ZOBRAZENIE ===
    cv2.imshow("Dartboard with projected darts", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()