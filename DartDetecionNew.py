import cv2
import numpy as np
import yaml
import json
import time

def load_calibration(file):
    """Načíta kalibračné údaje z YAML súboru."""
    with open(file, "r") as f:
        data = yaml.safe_load(f)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float32)

    # Vytvorenie optimalizovanej kamery a remapovacích máp
    h, w = 720, 1280
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0.4, (w, h))

    if new_camera_mtx is None or not isinstance(new_camera_mtx, np.ndarray):
        print("⚠️ new_camera_mtx nie je validná! Používam pôvodnú camera_matrix.")
        new_camera_mtx = camera_matrix

    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_mtx, (w, h), cv2.CV_32FC1)

    return camera_matrix, dist_coeffs, new_camera_mtx, mapx, mapy

# Kalibrácia a kamery
cam_files = [
    "./calibration/calib_data/left_calibration.yaml",
    "./calibration/calib_data/middle_calibration.yaml",
    "./calibration/calib_data/right_calibration.yaml"
]

cams = [3, 1, 2]  # Používame tri kamery
frame_size = (1280, 720)
cameras = []

for i in cams:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera {i}")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    cameras.append(cap)

calibration_data = [load_calibration(file) for file in cam_files]

# Funkcia na zachytenie referenčných obrázkov
def capture_reference_images():
    ref_images = []
    for i, cap in enumerate(cameras):
        ret, frame = cap.read()
        if ret:
            frame = cv2.remap(frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
            ref_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            ref_images.append(None)
    return ref_images

def threshing(frame, ref_image):
    """Vylepšená verzia prahovania s menšou citlivosťou na šum."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplikácia gaussovho filtra na referenčný aj aktuálny snímok (redukcia šumu)
    ref_blur = cv2.GaussianBlur(ref_image, (3, 3), 0)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Výpočet rozdielu medzi referenčným a aktuálnym snímkom
    diff = cv2.absdiff(ref_blur, gray_blur)
    
    # Použitie FIXNÉHO prahu, ale s vyššou hodnotou (50 namiesto 35)
    # Toto zníži falošné detekcie spôsobené malými zmenami v obraze
    _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    
    # Voliteľne: canny hrany s konzervatívnymi hodnotami
    edges = cv2.Canny(diff, 30, 100)  # Zvýšenie hodnôt pre menšiu citlivosť
    
    # Kombinácia thresholdu a hrán
    combined = cv2.bitwise_or(thresh, edges)
    
    # Morfologické operácie na odstránenie malých artefaktov
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Dodatočné odstránenie malých objektov
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Vytvorenie prázdneho obrazu a kreslenie len kontúr s dostatočnou veľkosťou
    filtered = np.zeros_like(combined)
    min_contour_area = 500  # Minimálna plocha kontúry, ktorú budeme uvažovať
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(filtered, [contour], -1, 255, -1)
    
    # Nájdenie finálnych kontúr na filtrovanom obraze
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, filtered

def reset_system():
    """Reset systému a zachytenie nových referenčných obrázkov."""
    print("⚠️ Reset detekcie...")
    time.sleep(1)  # Krátka pauza pre stabilizáciu
    return capture_reference_images()

# Inicializácia
print("Inicializácia kamier a referenčných snímok...")
ref_images = capture_reference_images()
shots = 0
shot_positions = {i: [] for i in range(len(cameras))}
stable_darts = {i: {"count": 0, "last_pos": None} for i in range(len(cameras))}
saved_images = {i: [] for i in range(len(cameras))}
detected_darts = {i: None for i in range(len(cameras))}
json_data = {"shots": []}

# Parametre detekcie
min_area = 800         # Minimálna plocha kontúry pre uvažovanie
stability_frames = 3   # Počet snímok, kedy musí byť hrot stabilný
reset_counter = 0      # Počítadlo pre periodický reset

print("Pripravené na detekciu. Čakám na šípky...")

while shots < 3:
    all_frames_empty = True  # Flag na kontrolu, či sú všetky thresholdy prázdne
    any_detection = False    # Flag na kontrolu, či bola nejaká detekcia
    
    for i in range(len(cameras)):
        ret, frame = cameras[i].read()
        if not ret or ref_images[i] is None:
            continue

        # Remap podľa kalibrácie
        frame = cv2.remap(frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
        
        # Detekcia kontúr s vylepšeným prahovaním
        contours, thresh = threshing(frame, ref_images[i])
        
        # Kontrola, či threshold nie je úplne prázdny
        if np.sum(thresh) > 0:
            all_frames_empty = False
        
        best_dart = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                any_detection = True
                x, y, w, h = cv2.boundingRect(contour)
                
                # Hľadanie najnižšieho bodu kontúry (hrot)
                bottom_threshold = y + int(0.9 * h)
                bottom_points = [pt[0] for pt in contour if pt[0][1] >= bottom_threshold]
                
                if bottom_points:
                    # Zoradenie bodov podľa y-súradnice (najnižší bod)
                    bottom_points = sorted(bottom_points, key=lambda point: point[1], reverse=True)
                    lowest_point = tuple(bottom_points[0])
                    
                    if stable_darts[i]["last_pos"] is None:
                        stable_darts[i]["count"] = 1
                        stable_darts[i]["last_pos"] = lowest_point
                    elif np.linalg.norm(np.array(stable_darts[i]["last_pos"]) - np.array(lowest_point)) < 10:
                        stable_darts[i]["count"] += 1
                        # Aktualizovať pozíciu pre plynulejšie sledovanie
                        alpha = 0.7  # Váha pre starú pozíciu
                        new_pos_x = int(alpha * stable_darts[i]["last_pos"][0] + (1-alpha) * lowest_point[0])
                        new_pos_y = int(alpha * stable_darts[i]["last_pos"][1] + (1-alpha) * lowest_point[1])
                        stable_darts[i]["last_pos"] = (new_pos_x, new_pos_y)
                    else:
                        stable_darts[i]["count"] = 1
                        stable_darts[i]["last_pos"] = lowest_point
                    
                    if stable_darts[i]["count"] >= stability_frames:
                        best_dart = stable_darts[i]["last_pos"]
                        # Vykresliť obdĺžnik a bod
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, best_dart, 5, (0, 0, 255), -1)
        
        # Aktualizovať detekciu len ak sme našli dart
        if best_dart is not None:
            detected_darts[i] = best_dart
        
        # Zobraziť upravené výstupy
        cv2.putText(frame, f"Camera {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if detected_darts[i] is not None:
            cv2.putText(frame, "DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(f"Camera {i}", frame)
        cv2.imshow(f"Threshold {i}", thresh)
    
    # Kontrola, či sú všetky thresholdy prázdne a resetovanie systému, ak áno
    if all_frames_empty and reset_counter % 50 == 0:
        print("Všetky thresholdy sú prázdne, resetovanie referencií...")
        ref_images = reset_system()
    
    reset_counter += 1
    
    # Periodický reset referencií, ak nie je žiadna detekcia
    if not any_detection and reset_counter % 100 == 0:
        print("Žiadna detekcia dlhšiu dobu, resetovanie referencií...")
        ref_images = reset_system()
    
    # Ak všetky kamery detekovali hrot
    if all(detected_darts[i] is not None for i in detected_darts):
        print("Detekované hroty na všetkých kamerách!")
        shots += 1
        shot_entry = {"shot_id": shots, "cameras": []}
        
        for i in range(len(cameras)):
            shot_positions[i].append(detected_darts[i])
            shot_entry["cameras"].append({
                "camera_id": i,
                "x": int(detected_darts[i][0]),
                "y": int(detected_darts[i][1])
            })
            print(f"🎯 Kamera {i}: Stabilný hrot šípky na X={detected_darts[i][0]}, Y={detected_darts[i][1]}")
            
            # Uloženie finálneho snímku
            ret, final_frame = cameras[i].read()
            if ret:
                final_frame = cv2.remap(final_frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
                cv2.circle(final_frame, detected_darts[i], 5, (0, 0, 255), -1)
                saved_images[i].append(final_frame)
                cv2.imwrite(f"DetectedDartsFoto/detected_dart_camera_{i}_shot_{shots}.png", final_frame)
        
        json_data["shots"].append(shot_entry)
        
        # Reset po detekcii
        time.sleep(1)
        ref_images = reset_system()
        detected_darts = {i: None for i in range(len(cameras))}
        stable_darts = {i: {"count": 0, "last_pos": None} for i in range(len(cameras))}
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Manuálny reset
        print("Manuálny reset...")
        ref_images = reset_system()
        detected_darts = {i: None for i in range(len(cameras))}
        stable_darts = {i: {"count": 0, "last_pos": None} for i in range(len(cameras))}

# Uloženie detekovaných pozícií
with open("detected_darts.json", "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print("\n🎯 FINAL DETECTED DART POSITIONS:")
for i in shot_positions:
    for idx, tip in enumerate(shot_positions[i]):
        print(f"📌 Kamera {i}, Hod {idx+1}: X={tip[0]}, Y={tip[1]}")
        if ref_images[i] is not None:
            result_img = cv2.cvtColor(ref_images[i], cv2.COLOR_GRAY2BGR)
            cv2.circle(result_img, tip, 5, (0, 0, 255), -1)
            cv2.putText(result_img, f"Shot {idx+1}", (tip[0]+15, tip[1]), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(f"Final Results Camera {i}", result_img)

for cap in cameras:
    cap.release()
cv2.destroyAllWindows()

print("Program ukončený. Stlačte ENTER pre ukončenie.")
input()