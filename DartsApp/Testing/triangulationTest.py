import cv2
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# === CESTY K SÚBOROM KALIBRÁCIE ===
calibration_files = [
    "../Calibration/calibration1000/calib_data/left_calibration.yaml",
    "../Calibration/calibration1000/calib_data/middle_calibration.yaml",
    "../Calibration/calibration1000/calib_data/right_calibration.yaml"
]

stereo_files = [
    "../Calibration/calibration1000/calib_data/stereo_middle_to_left_calibration.yaml",
    "../Calibration/calibration1000/calib_data/stereo_middle_to_right_calibration.yaml"
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

# === STEREO TRANSFORMÁCIE ===
with open(stereo_files[0]) as f:
    left_stereo = yaml.safe_load(f)
R_left = np.array(left_stereo['rotation_matrix'])
T_left = np.array(left_stereo['translation_vector'])

with open(stereo_files[1]) as f:
    right_stereo = yaml.safe_load(f)
R_right = np.array(right_stereo['rotation_matrix'])
T_right = np.array(right_stereo['translation_vector'])

# === POZÍCIE A ROTÁCIE KAMIER ===
extrinsics = [
    {'pos': (-R_left.T @ T_left).flatten(), 'R': R_left, 'T': T_left.reshape(-1, 1)},
    {'pos': np.zeros((3,)), 'R': np.eye(3), 'T': np.zeros((3, 1))},
    {'pos': (-R_right.T @ T_right).flatten(), 'R': R_right, 'T': T_right.reshape(-1, 1)}
]

# === NAČÍTANIE 2D BODOV ===
with open("./detected_darts.json") as f:
    darts_data = json.load(f)

# === POMOCNÉ FUNKCIE ===
def projection_matrix(K, R, T):
    RT = np.hstack((R, T.reshape(-1, 1)))
    return K @ RT

def triangulate_three_cameras(shot, cameras, extrinsics):
    cam_points = {c['camera_id']: c for c in shot['cameras']}
    required_ids = [0, 1, 2]  # left, middle, right

    if not all(cid in cam_points for cid in required_ids):
        return None

    triangulated_points = []
    stereo_pairs = [(0, 1), (1, 2)]  # left-middle and middle-right

    for id1, id2 in stereo_pairs:
        c1 = cam_points[id1]
        c2 = cam_points[id2]

        pt1 = np.array([[c1['x'], c1['y']]], dtype=np.float32)
        pt2 = np.array([[c2['x'], c2['y']]], dtype=np.float32)

        pt1_ud = cv2.undistortPoints(pt1, cameras[id1]['K'], cameras[id1]['D'], P=cameras[id1]['K']).reshape(-1)
        pt2_ud = cv2.undistortPoints(pt2, cameras[id2]['K'], cameras[id2]['D'], P=cameras[id2]['K']).reshape(-1)

        P1 = projection_matrix(cameras[id1]['K'], extrinsics[id1]['R'], extrinsics[id1]['T'])
        P2 = projection_matrix(cameras[id2]['K'], extrinsics[id2]['R'], extrinsics[id2]['T'])

        pt1_h = np.array([[pt1_ud[0]], [pt1_ud[1]]])
        pt2_h = np.array([[pt2_ud[0]], [pt2_ud[1]]])

        point_4d = cv2.triangulatePoints(P1, P2, pt1_h, pt2_h)
        point_3d = point_4d[:3] / point_4d[3]
        triangulated_points.append(point_3d.flatten())

   # --- Adaptívny vážený priemer podľa X pozície: viac vľavo = viac dôvera LM, vpravo = MR
    x_estimate = np.mean([p[0] for p in triangulated_points])
    if x_estimate < 0:
        weights = [0.6, 0.4]
    elif x_estimate > 0:
        weights = [0.4, 0.6]
    else:
        weights = [0.5, 0.5]

    triangulated_points = np.array(triangulated_points)
    averaged_point = np.average(triangulated_points, axis=0, weights=weights)
    print(f"Váhy pre trianguláciu (LM, MR): {weights}")

    # Korekcia do roviny terča sa vypustí, pretože narúšala orientáciu
    return averaged_point

# === TRIANGULÁCIA ===
points_3d = []
points_3d_json = []
view_lines = []
#[7.85, -74.5, 351.77] DEFAULT STRED
#[1, -32.81, 405] new one
##################################################################################################################
##################################################################################################################
target_center = np.array([19.10, -24.91, 414.20])
##################################################################################################################
##################################################################################################################

for shot in darts_data['shots']:
    point_3d = triangulate_three_cameras(shot, cameras, extrinsics)
    if point_3d is None:
        continue

    distance = np.linalg.norm(point_3d - target_center)
    points_3d.append((shot['shot_id'], point_3d))
    points_3d_json.append({"shot_id": shot['shot_id'], "position": point_3d.tolist()})
    view_lines.append([])  # prázdne view_lines pre kompatibilitu s vykreslením



# === VIZUALIZÁCIA ===
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Kamery
colors = ['blue', 'green', 'orange']
for i, cam in enumerate(extrinsics):
    pos = cam['pos'].flatten()
    R = cam['R']
    ax.scatter(*pos, color=colors[i], marker='^', s=100, label=f"Camera {i}")

    # Orientácia kamery (osi)
    scale = 300
    x_axis = pos + R.T @ np.array([scale, 0, 0])
    y_axis = pos + R.T @ np.array([0, scale, 0])
    z_axis = pos + R.T @ np.array([0, 0, scale])
    ax.plot([pos[0], x_axis[0]], [pos[1], x_axis[1]], [pos[2], x_axis[2]], color='r')
    ax.plot([pos[0], y_axis[0]], [pos[1], y_axis[1]], [pos[2], y_axis[2]], color='g')
    ax.plot([pos[0], z_axis[0]], [pos[1], z_axis[1]], [pos[2], z_axis[2]], color='b')
    print(f"Camera {i}:")
    print(f"  Position: {cam['pos'].flatten()}")
    print(f"  Rotation Matrix:\n{cam['R']}")
    print(f"  Intrinsic Matrix (K):\n{cameras[i]['K']}\n")

def correct_distance_by_y(distance, y_position):
    print(y_position)
    if distance > 50:
        if y_position > 0:
            # Horná polovica (nad stredom)
        #     return (
        #     -0.0000005992 * distance**3
        #     - 0.0004997 * distance**2
        #     + 0.9656 * distance
        #     - 0.4132
        # )
            return distance
        else:
            # Dolná polovica (pod stredom)
            return distance
    else:
        return distance
    
# === Bodky, čiary od stredu a vzdialenosť ===
for (shot_id, point), lines in zip(points_3d, view_lines):
     # === Výpočet vzdialenosti a uhla od stredu ===
    relative_vector = point - target_center
    distance2 = np.linalg.norm(relative_vector)

    # Rovinu berieme ako X-Z, kde Z ide "do hĺbky"
    angle_rad = np.arctan2(-relative_vector[2], relative_vector[0])
    angle_deg = (np.degrees(angle_rad) + 360) % 360

    # === Určenie segmentu na základe uhla ===

    segment_values = [6, 10, 15, 2, 17, 3, 19, 7, 16, 8,
                     11, 14, 9, 12, 5, 20, 1, 18, 4, 13]
    segment_size = 18  # 360 / 20
    angle_offset = 9  # ak je segment 6 hore
    segment_index = int((angle_deg + angle_offset) % 360 // segment_size)
    segment_value = segment_values[segment_index % 20]
    # === Výpočet skóre na základe vzdialenosti ===
    BULLSEYE_RADIUS = 7
    OUTER_BULL_RADIUS = 17
    TRIPLE_INNER_RADIUS = 97
    TRIPLE_OUTER_RADIUS = 107
    DOUBLE_INNER_RADIUS = 160
    DOUBLE_OUTER_RADIUS = 170

    print(f"vzdialenost pred {distance2}")

    distance = correct_distance_by_y(distance2, point[2] - target_center[2])

    print(f"vzdialenost {distance}")

    if distance <= BULLSEYE_RADIUS:
        score = 50
    elif distance <= OUTER_BULL_RADIUS:
        score = 25
    elif distance <= TRIPLE_INNER_RADIUS:
        score = segment_value
    elif distance <= TRIPLE_OUTER_RADIUS:
        score = segment_value * 3
    elif distance <= DOUBLE_INNER_RADIUS:
        score = segment_value
    elif distance <= DOUBLE_OUTER_RADIUS:
        score = segment_value * 2
    else:
        score = 0

    print(score)

    # === Zobrazenie v grafe ===
    ax.plot([target_center[0], point[0]],
            [target_center[1], point[1]],
            [target_center[2], point[2]], color='purple', linestyle='dashed')

    text_pos = (point + target_center) / 2
    ax.text(*text_pos, f"{distance:.1f} mm\n{angle_deg:.1f}°", color='black', fontsize=8)

    


# Rotácia okolo osi X
def Rx(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

# Rotácia okolo osi Y
def Ry(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

# Rotácia okolo osi Z
def Rz(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])

# === FUNKCIA: vykreslenie terča ===
def draw_target(ax, center, radius=225, rotation_angles=(0,-30,90), resolution=100):
    # rotation_angles = (x_angle, y_angle, z_angle)
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle_pts = np.vstack((np.zeros_like(theta), radius * np.cos(theta), radius * np.sin(theta)))
    
    # Vytvorenie rotačnej matice
    x_angle, y_angle, z_angle = rotation_angles
    rotation_matrix = Rz(z_angle) @ Ry(y_angle) @ Rx(x_angle)
    
    rotated_pts = rotation_matrix @ circle_pts
    rotated_pts = rotated_pts + center[:, np.newaxis]
    
    ax.plot(rotated_pts[0], rotated_pts[1], rotated_pts[2], color='black', label='Target')
    
    # Vykreslenie osí
    #axes_len = 100
    #ax.quiver(*center, *(rotation_matrix @ [axes_len, 0, 0]), color='r')
    #ax.quiver(*center, *(rotation_matrix @ [0, axes_len, 0]), color='g')
    #ax.quiver(*center, *(rotation_matrix @ [0, 0, axes_len]), color='b')


# Triangulované body a priamky
for (shot_id, point), lines in zip(points_3d, view_lines):
    ax.scatter(*point, c='red', s=60, label=f"Shot {shot_id}")
    for C, dir in lines:
        end = C + dir * 200
        ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]], linestyle='dashed', alpha=0.3)

print("--- TRIANGULOVANÉ BODY ---")
for shot_id, point in points_3d:
    print(f"Shot {shot_id}: X={point[0]:.2f}, Y={point[1]:.2f}, Z={point[2]:.2f}")

draw_target(ax, center=target_center)


target_center = target_center.tolist()
target_radius = 225
target_rotation = (180, -35, 90)  # (x_angle, y_angle, z_angle)

output_data = {
    "cameras": cameras,
    "extrinsics": extrinsics, 
    "triangulated_points": points_3d_json,
    "target": {
        "center": target_center,
        "radius": target_radius,
        "rotation": target_rotation
    }
}

# Pomocná funkcia na konverziu NumPy arrayov na zoznamy
def convert_np_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_np_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_list(item) for item in obj]
    else:
        return obj

output_data = convert_np_to_list(output_data)

# Uloženie do JSON súboru
with open("darts_3d_data.json", "w") as outfile:
    json.dump(output_data, outfile, indent=4)

print("Údaje boli úspešne exportované do darts_3d_data.json")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vizualizácia kamier, triangulovaných bodov a pohladu')
plt.legend()
plt.tight_layout()
plt.show()