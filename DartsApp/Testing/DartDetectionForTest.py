import cv2
import numpy as np
import yaml
import json
import time

class KalmanTracker:
    """Trieda pre sledovanie objektov pomocou Kalmanovho filtra."""
    def __init__(self):
        # Inicializácia Kalmanovho filtra pre sledovanie v 2D priestore (x, y)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        
        self.last_prediction = None
        self.initialized = False
        
    def update(self, x, y):
        """Aktualizuje filter s novým meraním."""
        measurement = np.array([[x], [y]], np.float32)
        
        if not self.initialized:
            # Inicializácia stavu filtra
            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
        
        # Predikcia
        prediction = self.kalman.predict()
        
        # Korekcia s novým meraním
        corrected = self.kalman.correct(measurement)
        
        # Uloženie výsledku
        self.last_prediction = corrected[:2].reshape(-1)
        return self.last_prediction
    
    def get_position(self):
        """Vráti aktuálnu pozíciu."""
        if self.last_prediction is not None:
            return (int(self.last_prediction[0]), int(self.last_prediction[1]))
        return None

def load_calibration(file):
    """Načíta kalibračné údaje z YAML súboru."""
    with open(file, "r") as f:
        data = yaml.safe_load(f)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float32)

    # Vytvorenie optimalizovanej kamery a remapovacích máp
    h, w = 1000, 1000
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))

    if new_camera_mtx is None or not isinstance(new_camera_mtx, np.ndarray):
        print("⚠️ new_camera_mtx nie je validná! Používam pôvodnú camera_matrix.")
        new_camera_mtx = camera_matrix

    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_mtx, (w, h), cv2.CV_32FC1)

    return camera_matrix, dist_coeffs, new_camera_mtx, mapx, mapy

def capture_reference_images(cameras, calibration_data):
    """Zachytí referenčné obrázky s viacerými snímkami pre stabilnejší referenčný obraz."""
    ref_images = []
    
    for i, cap in enumerate(cameras):
        # Zachytíme viac snímok a spriemerujeme ich pre lepšiu referenčnú snímku
        frames_count = 5
        frames = []
        
        for _ in range(frames_count):
            ret, frame = cap.read()
            if ret:
                frame = cv2.remap(frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            time.sleep(0.05)  # Krátka pauza medzi snímkami
        
        if frames:
            # Vytvoríme priemer zo všetkých snímok
            avg_frame = np.mean(frames, axis=0).astype(np.uint8)
            # Aplikujeme Gaussovský filter pre odstránenie zvyškového šumu
            avg_frame = cv2.GaussianBlur(avg_frame, (3, 3), 0.5)
            ref_images.append(avg_frame)
        else:
            ref_images.append(None)
            
    return ref_images

def adaptive_threshing(frame, ref_image):
    """Vylepšená prahovacia funkcia s adaptívnym prahovaním a viacerými metódami."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_image, gray)
    
    # Aplikácia filtrov pre redukciu šumu
    blurred = cv2.GaussianBlur(diff, (3, 3), 2)
    
    # Základné prahovanie
    # _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2
    )
    combined = cv2.bitwise_or(thresh, adaptive)

    # Canny detektor hrán s optimalizovanými parametrami
    edges = cv2.Canny(blurred, 30, 90)
    
    # === MORFOLOGIA ===
    kernel_erode = np.ones((2, 2), np.uint8)
    kernel_dilate = np.ones((3, 3), np.uint8)
    morph = cv2.erode(thresh, kernel_erode, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel_dilate, iterations=2)
    
    # Filtrovanie malých kontúr
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, thresh, morph

def detect_arrowV1(frame, ref_image):
    """Detekcia šípky pomocou kombinácie najlepších techník podľa vizuálnej analýzy."""
    import cv2
    import numpy as np

    # Prevod na odtieň šedej a výpočet rozdielu od referenčného obrazu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 1.0)
    ref_blur = cv2.GaussianBlur(ref_image, (3, 3), 1.0)
    diff = cv2.absdiff(ref_blur, gray_blur)

    # Gaussov filter (7x7, sigma=2.0)
    # blurred = cv2.GaussianBlur(diff, (3, 3), 2.0)

    # Thresholdovanie
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # Adaptívne thresholdovanie (block size = 11, C = 3)
    adaptive = cv2.adaptiveThreshold(
        diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 7, 3
    )

    # Kombinácia thresholdov
    combined = cv2.bitwise_or(thresh, adaptive)

    # Canny detekcia hrán
    # edges = cv2.Canny(blurred, 30, 90)

    # Morfologické operácie – dilatácia + closing
    kernel_dilate = np.ones((4, 4), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_dilate, iterations=2)

    # Nájsť kontúry
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Môžeš si pridať ďalšie filtrovanie kontúr tu (napr. podľa veľkosti, pomeru strán...)

    return contours, thresh, morph

def detect_dart_tip(contours, frame, min_area=1000 ):
    """Vylepšená detekcia hrotu šípky s použitím viacerých kritérií."""
    dart_candidates = []
    
    

    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # Ďalšie kritériá pre filtrovanie
        x, y, w, h = cv2.boundingRect(contour)
        
        # Kontrola pomeru strán
        aspect_ratio = w / float(h) if h > 0 else 0
        print(f"[DEBUG] Detekovaných kontúr: {len(contours)} a area = {area} a aspect_ration = {aspect_ratio}")      
        if aspect_ratio > 2.5 or aspect_ratio < 0.08:  # Filtrovanie príliš širokých alebo úzkych objektov
            continue
            
        # Aproximácia kontúry pre analýzu tvaru
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)  
        
        # Výpočet najnižšieho bodu kontúry (hrot šípky)
        bottom_threshold = y + int(0.8 * h)
        bottom_points = [pt[0] for pt in contour if pt[0][1] >= bottom_threshold]
        
        if not bottom_points:
            continue
            
        lowest_point = max(bottom_points, key=lambda point: point[1])
        
        # Výpočet skóre pre túto kontúru na základe rôznych metrík
        # - väčšia plocha je lepšia, ale nie príliš veľká
        # - nižšia pozícia je lepšia (šípka smeruje nadol)
        # - vhodný pomer strán
        score = min(area / 1000, 3) + (lowest_point[1] / 100) - abs(aspect_ratio - 1.0)
        # score = 500

        # === ZOBRAZENIE DETEKCIE ===
        # cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, tuple(lowest_point), 1, (0, 0, 255), -1)  # 💥 hrot šípky
        cv2.putText(frame, "Dart Tip", (lowest_point[0] + 5, lowest_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        #cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)  # žltá farba
        #cv2.putText(frame, f"Pts: {len(approx)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        dart_candidates.append({
            'contour': contour,
            'lowest_point': lowest_point,
            'score': score,
            'area': area,
            'rect': (x, y, w, h)
        })
    
    # Zoradenie kandidátov podľa skóre
    dart_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # if dart_candidates:
    #     best_candidate = dart_candidates[0]
    #     return best_candidate
    
    return None

def detect_dart_tipNEW(contours, frame, min_area=700):
    dart_candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        print(f"[DEBUG] Kontúra: area = {area}, aspect_ratio = {aspect_ratio}")

        if aspect_ratio > 2.5 or aspect_ratio < 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        # === Nová detekcia cez PCA ===
        tip_point = find_tip_by_orientation(contour, frame, draw_axis=True)

        # Skóre: stále kombinujeme viaceré faktory
        score = min(area / 1000, 3) + (tip_point[1] / 100) - abs(aspect_ratio - 1.0)

        # === Vizualizácia kandidáta ===
        cv2.putText(frame, "Dart Tip", (tip_point[0] + 5, tip_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        #cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)

        dart_candidates.append({
            'contour': contour,
            'lowest_point': tip_point,
            'score': score,
            'area': area,
            'rect': (x, y, w, h)
        })
    dart_candidates.sort(key=lambda x: x['score'], reverse=True)
    return dart_candidates[0] if dart_candidates else None

def find_tip_by_orientation(contour, frame=None, draw_axis=False):
    """Použije PCA na určenie hlavného smeru šípky a nájde bod najvzdialenejší v tomto smere."""
    data_pts = np.array(contour, dtype=np.float32).reshape(-1, 2)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)
    center = tuple(mean[0])
    direction = eigenvectors[0]

    # Nájdeme bod najďalej v smere direction (projekcia)
    projections = np.dot(data_pts - mean, direction.reshape(2, 1))
    max_idx = np.argmax(projections)
    tip_point = tuple(data_pts[max_idx].astype(int))

    if frame is not None and draw_axis:
        # Nakresli hlavnú os šípky (modrá čiara)
        p1 = (int(center[0] - direction[0]*100), int(center[1] - direction[1]*100))
        p2 = (int(center[0] + direction[0]*100), int(center[1] + direction[1]*100))
        cv2.line(frame, p1, p2, (255, 0, 0), 1)  # Modrá = PCA os

        # Označ tip
        cv2.circle(frame, tip_point, 4, (0, 0, 255), -1)  # Červený = nový hrot

    return tip_point

# Hlavný program
def main():
    # Kalibrácia a kamery
    cam_files = [
        "../Calibration/calibration1000/calib_data/left_calibration.yaml",
        "../Calibration/calibration1000/calib_data/middle_calibration.yaml",
        "../Calibration/calibration1000/calib_data/right_calibration.yaml"
    ]

    cams = [3,1,2]  # Používame tri kamery
    frame_size = (1000, 1000)
    cameras = []

    # Inicializácia kalmanových filtrov pre každú kameru
    kalman_filters = [KalmanTracker() for _ in range(len(cams))]

    # Otvorenie kamier s optimalizovaným nastavením
    for i in cams:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Cannot open camera {i}")
            exit()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        # Nastavenia pre zlepšenie kvality obrazu
        # cap.set(cv2.CAP_PROP_FPS, 30)  # Vyššia frekvencia snímok
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Vypnutie automatického ostrenia (ak je to možné)
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Automatická expozícia pre prispôsobenie sa osvetleniu
        cameras.append(cap)

    calibration_data = [load_calibration(file) for file in cam_files]

    # Inicializácia
    print("🔄 Inicializácia referenčných snímok...")
    time.sleep(1)  # Dáme kamerám čas na inicializáciu
    ref_images = capture_reference_images(cameras, calibration_data)
    shots = 0  
    shot_positions = {i: [] for i in range(len(cameras))} 
    stable_darts = {i: {"count": 0, "last_pos": None, "positions": []} for i in range(len(cameras))}  
    saved_images = {i: [] for i in range(len(cameras))}  
    detected_darts = {i: None for i in range(len(cameras))}  
    json_data = {"shots": []}

    print("📸 Čakám na hody šípkami...")

    while shots < 3:
        for i in range(len(cameras)):
            ret, frame = cameras[i].read()
            if not ret or ref_images[i] is None:
                continue

            # Aplikácia kalibrácie
            frame = cv2.remap(frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
            
            # Použijeme vylepšenú adaptívnu threshing funkciu
            contours, thresh, morph = detect_arrowV1(frame, ref_images[i])

            # Detekcia hrotu šípky
            # dart_tip = detect_dart_tip(contours, frame)
            dart_tip = detect_dart_tipNEW(contours, frame)

            if dart_tip:
                lowest_point = dart_tip['lowest_point']
                rect = dart_tip['rect']
                
                # Aplikácia Kalmanovho filtra na pozíciu hrotu
                filtered_pos = kalman_filters[i].update(lowest_point[0], lowest_point[1])
                filtered_point = (int(filtered_pos[0]), int(filtered_pos[1]))
                
                # Kontrola stability - sledujeme posledných niekoľko pozícií
                max_positions = 5
                stable_darts[i]["positions"].append(filtered_point)
                if len(stable_darts[i]["positions"]) > max_positions:
                    stable_darts[i]["positions"].pop(0)
                
                # Kontrola stability na základe rozptylu pozícií
                if len(stable_darts[i]["positions"]) >= 3:
                    positions = np.array(stable_darts[i]["positions"])
                    std_dev = np.std(positions, axis=0)
                    
                    # Ak je rozptyl malý (stabilná pozícia)
                    if np.mean(std_dev) < 4.0:
                        stable_darts[i]["count"] += 1
                    else:
                        stable_darts[i]["count"] = max(0, stable_darts[i]["count"] - 1)
                
                # Vizualizácia detekcie
                x, y, w, h = rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Zobrazenie hrotu a filtrovanej pozície
                cv2.circle(frame, tuple(lowest_point), 2, (0, 0, 255), -1)
                cv2.circle(frame, filtered_point, 2, (255, 0, 0), -1)
                
                # Ak je detekcia stabilná
                if stable_darts[i]["count"] >= 3:
                    detected_darts[i] = filtered_point
                    cv2.putText(frame, "STABLE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Zobraziť informácie o stabilite
                    stability_info = f"Stability: {stable_darts[i]['count']}/3"
                    cv2.putText(frame, stability_info, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Žiadna detekcia
                stable_darts[i]["count"] = max(0, stable_darts[i]["count"] - 1)
                if stable_darts[i]["count"] == 0:
                    detected_darts[i] = None

            # Zobrazenie informácií na obraze
            cv2.putText(frame, f"Kamera {i} - Hody: {shots}/3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(f"Kamera {i}", frame)
            # cv2.imshow(f"Threshold {i}", thresh)
            cv2.imshow(f"Morphology {i}", morph)
            

        # Kontrola, či máme detekovanú šípku na všetkých kamerách
        if all(detected_darts[i] is not None for i in detected_darts):
            shots += 1
            shot_entry = {"shot_id": shots, "cameras": []}
            
            print(f"\n🎯 DETEKOVANÁ ŠÍPKA {shots}/3:")
            for i in range(len(cameras)):
                shot_positions[i].append(detected_darts[i])
                shot_entry["cameras"].append({
                    "camera_id": i,
                    "x": int(detected_darts[i][0]),
                    "y": int(detected_darts[i][1])
                })
                print(f"📌 Kamera {i}: Hrot šípky na X={detected_darts[i][0]}, Y={detected_darts[i][1]}")
                
                # Uloženie snímky s detekciou
                ret, final_frame = cameras[i].read()
                if ret:
                    final_frame = cv2.remap(final_frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
                    cv2.circle(final_frame, detected_darts[i], 2, (0, 0, 255), -1)
                    cv2.putText(final_frame, f"Shot {shots}", (detected_darts[i][0] - 30, detected_darts[i][1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    saved_images[i].append(final_frame)
                    cv2.imwrite(f"./DetectedDartsFoto/detected_dart_camera_{i}_shot_{shots}.png", final_frame)
                    cv2.imwrite(f"./DetectedDartsFoto/detected_dart_cameraMOPRHaa_{i}_shot_{shots}.png", morph)

            
            json_data["shots"].append(shot_entry)
            
            # Reset detekcií a obnovenie referenčných snímok
            print("⏳ Čakám na ďalší hod...")
            time.sleep(0.5)  # Krátke čakanie
            ref_images = capture_reference_images(cameras, calibration_data)
            detected_darts = {i: None for i in range(len(cameras))}
            stable_darts = {i: {"count": 0, "last_pos": None, "positions": []} for i in range(len(cameras))}
            # Reset Kalmanových filtrov pre nový hod
            kalman_filters = [KalmanTracker() for _ in range(len(cams))]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Uloženie výsledkov
    with open("detected_darts.json", "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    print("\n🎯 FINÁLNE POZÍCIE DETEKOVANÝCH ŠÍPOK:")
    for i in shot_positions:
        for idx, tip in enumerate(shot_positions[i]):
            print(f"📌 Kamera {i}, Hod {idx+1}: X={tip[0]}, Y={tip[1]}")
    
    # Vytvorenie súhrnného obrázka
    for i in range(len(cameras)):
        if saved_images[i]:
            # Vytvorenie gridového zobrazenia všetkých hodov
            grid_height = len(saved_images[i]) * frame_size[1]
            grid_img = np.zeros((grid_height, frame_size[0], 3), dtype=np.uint8)
            
            for j, img in enumerate(saved_images[i]):
                start_y = j * frame_size[1]
                grid_img[start_y:start_y + frame_size[1], :] = img
                
            cv2.imshow(f"Výsledky kamery {i}", grid_img)
            cv2.imwrite(f"./DetectedDartsFoto/camera_{i}_all_shots.png", grid_img)

    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()