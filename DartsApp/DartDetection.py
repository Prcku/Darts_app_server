import cv2
import numpy as np
import yaml
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
        for _ in range(5):  # napr. 5 snímok, prípadne viac ak máš vysoký FPS
            cap.read()
            time.sleep(0.05)  # trochu počkaj, aby boli nové snímky

        ret, frame = cap.read()
        if ret:
            frame = cv2.remap(frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
    
        avg_frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0.5)
        ref_images.append(avg_frame)
            
    return ref_images

def adaptive_threshing(frame, ref_image):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 1.0)
    ref_blur = cv2.GaussianBlur(ref_image, (3, 3), 1.0)
    diff = cv2.absdiff(ref_blur, gray_blur)

    # Thresholdovanie
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

    # Adaptívne thresholdovanie (block size = 11, C = 3)
    adaptive = cv2.adaptiveThreshold(
        diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 7, 3
    )

    # Kombinácia thresholdov
    combined = cv2.bitwise_or(thresh, adaptive)

    # Morfologické operácie – dilatácia + closing
    kernel_dilate = np.ones((4, 4), np.uint8)
    morph = cv2.dilate(thresh, kernel_dilate, iterations=1)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_dilate, iterations=2)

    # Nájsť kontúry
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, thresh, morph

def detect_dart_tipNEW(contours, min_area=700):
    dart_candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0

        if aspect_ratio > 2.5 or aspect_ratio < 0.08:
            continue

        # === Nová detekcia cez PCA ===
        tip_point = find_tip_by_orientation(contour)

        # Skóre: stále kombinujeme viaceré faktory
        score = min(area / 1000, 3) + (tip_point[1] / 100) - abs(aspect_ratio - 1.0)

        dart_candidates.append({
            'contour': contour,
            'lowest_point': tip_point,
            'score': score,
            'area': area,
            'rect': (x, y, w, h)
        })
    dart_candidates.sort(key=lambda x: x['score'], reverse=True)
    return dart_candidates[0] if dart_candidates else None

def find_tip_by_orientation(contour):
    """Použije PCA na určenie hlavného smeru šípky a nájde bod najvzdialenejší v tomto smere."""
    data_pts = np.array(contour, dtype=np.float32).reshape(-1, 2)
    mean, eigenvectors = cv2.PCACompute(data_pts, mean=None)
    center = tuple(mean[0])
    direction = eigenvectors[0]

    # Nájdeme bod najďalej v smere direction (projekcia)
    projections = np.dot(data_pts - mean, direction.reshape(2, 1))
    max_idx = np.argmax(projections)
    tip_point = tuple(data_pts[max_idx].astype(int))

    return tip_point

def dartDetection(cameras,calibration_data, camsID, ref_images, shots=0):

    detected_darts = {i: None for i in range(len(cameras))}
    stable_darts = {i: {"count": 0, "last_pos": None, "positions": []} for i in range(len(cameras))}
    kalman_filters = [KalmanTracker() for _ in range(len(camsID))]

    print("presiel cas")

    while shots < 1:
        for i in range(len(cameras)):
            ret, frame = cameras[i].read()
            if not ret or ref_images[i] is None:
                continue

            # Aplikácia kalibrácie
            frame = cv2.remap(frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
            
            # Použijeme vylepšenú adaptívnu threshing funkciu
            contours, thresh, morph = adaptive_threshing(frame, ref_images[i])

            # Detekcia hrotu šípky
            dart_tip = detect_dart_tipNEW(contours)

            if dart_tip:
                lowest_point = dart_tip['lowest_point']
                
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
                
                # Zobrazenie hrotu a filtrovanej pozície
                cv2.circle(frame, filtered_point, 2, (255, 0, 0), -1)
                
                # Ak je detekcia stabilná
                if stable_darts[i]["count"] >= 3:
                    detected_darts[i] = filtered_point

            else:
                # Žiadna detekcia
                stable_darts[i]["count"] = max(0, stable_darts[i]["count"] - 1)
                if stable_darts[i]["count"] == 0:
                    detected_darts[i] = None

        # Kontrola, či máme detekovanú šípku na všetkých kamerách
        if all(detected_darts[i] is not None for i in detected_darts):
            shots += 1
            
            print(f"\n🎯 DETEKOVANÁ ŠÍPKA")
            timestamp = int(time.time())
            ret, final_frame = cameras[0].read()
            final_frame = cv2.remap(final_frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
            cv2.imwrite(f"../DetectedDartsFoto/detection/detectedNEW_dart_camera_{0}_shot_{timestamp}.png", final_frame)
            for i in range(len(cameras)):
                print(f"📌 Kamera {i}: Hrot šípky na X={detected_darts[i][0]}, Y={detected_darts[i][1]}")
                
                # Uloženie snímky s detekciou
                ret, final_frame = cameras[i].read()
                if ret:
                    final_frame = cv2.remap(final_frame, calibration_data[i][3], calibration_data[i][4], cv2.INTER_LINEAR)
                    # cv2.circle(final_frame, detected_darts[i], 2, (0, 0, 255), -1)
                    # cv2.putText(final_frame, f"Shot {shots}", (detected_darts[i][0] - 30, detected_darts[i][1] - 20), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    #cv2.imwrite(f"../DetectedDartsFoto/ref/ref_dart_camera_{i}_shot_{timestamp}.png", ref_images[i])
                    #cv2.imwrite(f"../DetectedDartsFoto/moprh/morphNEW_dart_camera_{i}_shot_{timestamp}.png", morph)
                    
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return detected_darts

    


     
def releaseCameras(cameras):
    for cap in cameras:
        cap.release()
    cv2.destroyAllWindows()