import cv2
import numpy as np
import os
import glob
import yaml
import time

class MultiCameraCalibrator:
    def __init__(self, camera_indices, frame_size, checkerboard_size, square_size, 
                 calibration_dir="calibration1000"):
        """
        Inicializácia kalibrátora pre viacero kamier
        
        Parametre:
        - camera_indices: zoznam indexov kamier (napr. [3, 1, 2])
        - frame_size: rozmer snímky (šírka, výška)
        - checkerboard_size: počet vnútorných rohov šachovnice (šírka, výška)
        - square_size: veľkosť štvorca na šachovnici v mm
        - calibration_dir: základný priečinok pre kalibračné dáta
        """
        self.camera_indices = camera_indices
        self.camera_names = ["left", "middle", "right"]
        self.frame_size = frame_size
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Vytvorenie adresárovej štruktúry
        self.calibration_dir = calibration_dir
        self.images_dir = os.path.join(calibration_dir, "images")
        
        for cam_name in self.camera_names:
            os.makedirs(os.path.join(self.images_dir, cam_name), exist_ok=True)
        
        # Priečinok pre kalibračné údaje
        self.calib_data_dir = os.path.join(calibration_dir, "calib_data")
        os.makedirs(self.calib_data_dir, exist_ok=True)
        
        print(f"Inicializovaný kalibračný systém pre {len(camera_indices)} kamery")
        print(f"Rozmer šachovnice: {checkerboard_size}, veľkosť štvorca: {square_size}mm")
        print(f"Priečinky pripravené v: {calibration_dir}")

    def load_calibration_data(self, file_paths):
        return [
            {
                'id': idx,
                'camera_matrix': np.array(yaml.safe_load(open(path))['camera_matrix']),
                'distortion_coefficients': np.array(yaml.safe_load(open(path))['distortion_coefficients'])
            }
            for idx, path in enumerate(file_paths)
        ]

    def capture_single_camera_images(self):
        """
        Postupné zachytávanie obrázkov z každej kamery
        """
        for idx, (cam_idx, cam_name) in enumerate(zip(self.camera_indices, self.camera_names)):
            print(f"\nSpúšťam zachytávanie pre kameru {idx+1}/{len(self.camera_indices)} (index {cam_idx}, {cam_name})")
            
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"CHYBA: Nemôžem otvoriť kameru s indexom {cam_idx}")
                continue
            
            # Nastavenie rozlíšenia kamery
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            
            output_dir = os.path.join(self.images_dir, cam_name)
            img_counter = 0
            
            print(f"Zachytávanie obrázkov pre kameru {cam_name} (index {cam_idx})")
            print("Stlač 's' pre zachytenie obrázka, 'q' pre ukončenie")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Nepodarilo sa zachytiť snímku!")
                    break
                
                cv2.imshow(f"Kamera {cam_name}", frame)
                
                k = cv2.waitKey(1)
                if k == ord('s'):  # Stlač 's' pre uloženie obrázka
                    
                    
                    
                    # Skontroluj detekciu šachovnice
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret_check, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
                    
                    if ret_check:

                        img_path = os.path.join(output_dir, f"{cam_name}_{img_counter}.png")
                        cv2.imwrite(img_path, frame)
                        print(f"Uložený obrázok: {img_path}")
                        img_counter += 1

                        # Spresnenie detekcie rohov
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        
                        # Vykreslenie rohov
                        vis_frame = frame.copy()
                        cv2.drawChessboardCorners(vis_frame, self.checkerboard_size, corners2, ret_check)
                        cv2.imshow(f"Detekcia šachovnice {cam_name}", vis_frame)
                        cv2.waitKey(100)  # Zobraz výsledok na 1 sekundu
                    else:
                        print("UPOZORNENIE: Šachovnica nebola detekovaná na tomto obrázku! Obrazok sa neulozil")
                        
                elif k == ord('q'):  # Stlač 'q' pre ukončenie
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print(f"Celkovo zachytených {img_counter} obrázkov pre kameru {cam_name}")

    def capture_stereo_images(self, referenceCamera, camera, whichCamera,calibration_results):
        """
        Súčasné zachytávanie obrázkov zo všetkých kamier pre stereokalibráciu
        """
        # Otvorenie všetkých kamier
        caps = []

        if whichCamera == 'left':
            
            camera_pairs = [(camera, "left",calibration_results[0]), (referenceCamera, "middle",calibration_results[1])]
        else:
            camera_pairs = [(camera, "right",calibration_results[2]), (referenceCamera, "middle",calibration_results[1])]
        
        for cam_idx, cam_name, _ in camera_pairs:
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"CHYBA: Nemôžem otvoriť kameru {cam_name} s indexom {cam_idx}")
                # Zavrieť už otvorené kamery
                for c in caps:
                    c.release()
                return False
            
            # Nastavenie rozlíšenia kamery
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            caps.append(cap)
        
        img_counter = 0
        print(f"\nZachytávanie stereo obrázkov zo strednej a {whichCamera} kamery")
        print("Stlač 's' pre zachytenie obrázkov, 'q' pre ukončenie")
        
        objp = np.zeros((self.checkerboard_size[0]*self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        axis = np.float32([
            [0, 0, 0],
            [self.square_size, 0, 0],
            [0, self.square_size, 0],
            [0, 0, -self.square_size]
        ]).reshape(-1, 3)
        
        while True:
            frames = []
            frame_names = []
            for i, (cap, (_, cam_name,calibrationData)) in enumerate(zip(caps, camera_pairs)):
                ret, frame = cap.read()
                if not ret:
                    print(f"Nepodarilo sa zachytiť snímku z kamery {cam_name}!")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_cb, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

                if ret_cb:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    ret_pnp, rvecs, tvecs = cv2.solvePnP(objp, corners2, calibrationData['camera_matrix'], calibrationData['distortion_coefficients'])

                    if ret_pnp:
                        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, calibrationData['camera_matrix'], calibrationData['distortion_coefficients'])
                        corner = tuple(corners2[0].ravel().astype(int))
                        frame = cv2.line(frame, corner, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3)
                        frame = cv2.line(frame, corner, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 3)
                        frame = cv2.line(frame, corner, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 3)
                        # cv2.drawChessboardCorners(frame, self.checkerboard_size, corners2, ret_cb)

                frames.append(frame)
                frame_names.append(cam_name)
                cv2.imshow(f"Kamera {cam_name}", frame)
            
            if len(frames) != len(caps):
                print("CHYBA: Nepodarilo sa zachytiť snímky zo všetkých kamier")
                time.sleep(0.5)
                continue
            
            k = cv2.waitKey(1)
            if k == ord('s'):  # Stlač 's' pre uloženie obrázkov
                valid_capture = True
                
                # Skontroluj detekciu šachovnice na všetkých obrázkoch
                chessboard_frames = []
                chess_corners = []
                
                for i, frame in enumerate(frames):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret_check, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
                    
                    if ret_check:
                        # Spresnenie detekcie rohov
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        
                        # Vykreslenie rohov
                        vis_frame = frame.copy()
                        cv2.drawChessboardCorners(vis_frame, self.checkerboard_size, corners2, ret_check)
                        chessboard_frames.append(vis_frame)
                        chess_corners.append(corners2)
                    else:
                        print(f"UPOZORNENIE: Šachovnica nebola detekovaná na kamere {frame_names[i]}!")
                        valid_capture = False
                
                # Ak je šachovnica detekovaná na všetkých kamerách, ulož obrázky
                if valid_capture:
                    for i, (frame, chess_frame) in enumerate(zip(frames, chessboard_frames)):
                        cam_name = frame_names[i]
                        img_path = os.path.join(self.images_dir, cam_name, f"stereo_{whichCamera}_middle_{img_counter}.png")
                        cv2.imwrite(img_path, frame)
                        
                        # Zobraz detekciu šachovnice
                        # cv2.imshow(f"Detekcia šachovnice {cam_name}", chess_frame)
                    
                    # Zobraziť prepojenie medzi sachovnicami
                    if len(chess_corners) == 2:
                        # Vytvor kombinovaný obrázok zobrazujúci prepojenia
                        combined_img = np.hstack((chessboard_frames[0], chessboard_frames[1]))
                        h, w = frames[0].shape[:2]
                        
                        # Vykresli prepojenia medzi zodpovedajúcimi rohmi
                        num_corners = chess_corners[0].shape[0]
                        for j in range(0,num_corners, 7):
                            pt1 = (int(chess_corners[0][j][0][0]), int(chess_corners[0][j][0][1]))
                            pt2 = (int(chess_corners[1][j][0][0]) + w, int(chess_corners[1][j][0][1]))
                            color = (0, 255, 0)  # Zelená farba pre prepojenia
                            cv2.line(combined_img, pt1, pt2, color, 1)
                        
                        # cv2.imshow(f"Prepojenie šachovníc {whichCamera}-middle", combined_img)
                        # Uložiť aj kombinovaný obrázok
                        combined_path = os.path.join(self.images_dir, "Combined", f"combined_{whichCamera}_middle_{img_counter}BADBADBAD.png")
                        os.makedirs(os.path.join(self.images_dir, "Combined"), exist_ok=True)
                        cv2.imwrite(combined_path, combined_img)
                    
                    print(f"Uložená stereo sada {whichCamera}-middle #{img_counter}")
                    img_counter += 1
                    cv2.waitKey(1000)  # Zobraz výsledok na 1 sekundu
                
            elif k == ord('q'):  # Stlač 'q' pre ukončenie
                break
        
        # Zatvoriť všetky kamery
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print(f"Celkovo zachytených {img_counter} stereo sád obrázkov")
        return True

    def calibrate_single_camera(self, camera_index):
        """
        Kalibrácia jednej kamery a uloženie výsledkov
        
        Parametre:
        - camera_index: index kamery v zozname camera_indices
        
        Vracia:
        - slovník s kalibračnými parametrami
        """
        cam_name = self.camera_names[camera_index]
        print(f"\nKalibrácia kamery: {cam_name}")
        
        # Príprava objektových bodov (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp = objp * self.square_size  # Aplikácia veľkosti štvorca
        
        # Polia pre uloženie objektových a obrazových bodov
        objpoints = []  # 3D body v reálnom svete
        imgpoints = []  # 2D body v rovine snímky
        
        # Načítanie všetkých kalibračných obrázkov pre danú kameru
        images_path = os.path.join(self.images_dir, cam_name, "*.png")
        images = glob.glob(images_path)
        
        if len(images) == 0:
            print(f"CHYBA: Neboli nájdené žiadne obrázky v {images_path}")
            return None
        
        print(f"Spracovávam {len(images)} obrázkov pre kameru {cam_name}")
        
        successful_images = 0
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Nájdenie rohov šachovnice
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                # Spresnenie detekcie rohov
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Vykreslenie a zobrazenie rohov
                cv2.drawChessboardCorners(img, self.checkerboard_size, corners2, ret)
                cv2.imshow(f'Detekcia šachovnice - {cam_name}', img)
                cv2.waitKey(500)
                successful_images += 1
            else:
                print(f"Šachovnica nebola detekovaná v obrázku: {os.path.basename(fname)}")
        
        cv2.destroyAllWindows()
        print(f"Úspešne spracovaných {successful_images}/{len(images)} obrázkov")
        
        if successful_images == 0:
            print("CHYBA: Nebola úspešne spracovaná žiadna snímka pre kalibráciu!")
            return None
        
        print(f"Výpočet kalibračných parametrov pre kameru {cam_name}...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        # Výpočet chyby reprojekcie
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        mean_error /= len(objpoints)
        print(f"Celková chyba reprojekcie: {mean_error}")
        
        # Uloženie kalibračných parametrov do YAML súboru
        calibration_data = {
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'avg_reprojection_error': float(mean_error),
            'image_width': int(self.frame_size[0]),
            'image_height': int(self.frame_size[1])
        }
        
        # Uloženie kalibračných údajov
        calib_file = os.path.join(self.calib_data_dir, f"{cam_name}_calibration.yaml")
        with open(calib_file, 'w') as f:
            yaml.dump(calibration_data, f)
        
        print(f"Kalibračné údaje uložené do {calib_file}")
        
        return {
            'camera_matrix': mtx,
            'distortion_coefficients': dist,
            'rotation_vectors': rvecs,
            'translation_vectors': tvecs,
            'object_points': objpoints,
            'image_points': imgpoints
        }

    def calibrate_all_cameras(self):
        """
        Kalibrácia všetkých kamier a uloženie výsledkov
        
        Vracia:
        - zoznam slovníkov s kalibračnými parametrami pre každú kameru
        """
        calibration_results = []
        
        for i in range(len(self.camera_indices)):
            result = self.calibrate_single_camera(i)
            calibration_results.append(result)
        
        return calibration_results

    def perform_stereo_calibration(self, ref_camera_idx, target_camera_idx, calibration_results):
        """
        Vykonanie stereokalibrácie medzi dvoma kamerami
        
        Parametre:
        - ref_camera_idx: index referenčnej kamery (ktorá má pozíciu 0,0,0)
        - target_camera_idx: index cieľovej kamery
        - calibration_results: výsledky kalibrácie jednotlivých kamier
        
        Vracia:
        - slovník s výsledkami stereokalibrácie
        """
        ref_cam_name = self.camera_names[ref_camera_idx]
        target_cam_name = self.camera_names[target_camera_idx]
        
        print(f"\nStereokalibrácia: {ref_cam_name} (referencia) -> {target_cam_name}")
        
        # Získanie kalibračných údajov pre obe kamery
        ref_calib = calibration_results[ref_camera_idx]
        target_calib = calibration_results[target_camera_idx]
        
        if ref_calib is None or target_calib is None:
            print("CHYBA: Chýbajú kalibračné údaje pre jednu z kamier!")
            return None
        
        # Načítanie stereo obrázkov
        ref_images_pattern = os.path.join(self.images_dir, ref_cam_name, f"stereo_{target_cam_name}*.png")
        target_images_pattern = os.path.join(self.images_dir, target_cam_name, "stereo_*.png")
        
        ref_images = sorted(glob.glob(ref_images_pattern))
        target_images = sorted(glob.glob(target_images_pattern))
        
        if len(ref_images) == 0 or len(target_images) == 0:
            print("CHYBA: Neboli nájdené stereo obrázky!")
            return None
        
        # Zabezpečenie, že máme rovnaký počet obrázkov pre obe kamery
        min_images = min(len(ref_images), len(target_images))
        ref_images = ref_images[:min_images]
        target_images = target_images[:min_images]
        
        print(f"Spracovávam {min_images} párov stereo obrázkov")
        
        # Príprava objektových bodov (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp = objp * self.square_size  # Aplikácia veľkosti štvorca
        
        # Polia pre uloženie bodov
        objpoints = []  # 3D body v reálnom svete
        ref_imgpoints = []  # 2D body v rovine referenčnej kamery
        target_imgpoints = []  # 2D body v rovine cieľovej kamery
        
        for ref_img_path, target_img_path in zip(ref_images, target_images):
            # Načítanie páru obrázkov
            ref_img = cv2.imread(ref_img_path)
            target_img = cv2.imread(target_img_path)
            
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            
            # Nájdenie rohov šachovnice na oboch obrázkoch
            ret1, corners1 = cv2.findChessboardCorners(ref_gray, self.checkerboard_size, None)
            ret2, corners2 = cv2.findChessboardCorners(target_gray, self.checkerboard_size, None)
            
            if ret1 and ret2:  # Ak bola detekovaná šachovnica na oboch obrázkoch
                objpoints.append(objp)
                
                # Spresnenie detekcie rohov
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners1 = cv2.cornerSubPix(ref_gray, corners1, (11, 11), (-1, -1), criteria)
                corners2 = cv2.cornerSubPix(target_gray, corners2, (11, 11), (-1, -1), criteria)
                
                ref_imgpoints.append(corners1)
                target_imgpoints.append(corners2)
                
                # Vykreslenie a zobrazenie rohov
                vis_ref = ref_img.copy()
                vis_target = target_img.copy()
                
                cv2.drawChessboardCorners(vis_ref, self.checkerboard_size, corners1, ret1)
                cv2.drawChessboardCorners(vis_target, self.checkerboard_size, corners2, ret2)
                
                # Vykreslenie prepojenia medzi zodpovedajúcimi bodmi
                combined = np.hstack((vis_ref, vis_target))
                
                # Zobrazenie niektorých prepojení bodov (každý piaty bod pre prehľadnosť)
                for i in range(0, len(corners1), 7):
                    pt1 = (int(corners1[i][0][0]), int(corners1[i][0][1]))
                    pt2 = (int(corners2[i][0][0]) + vis_ref.shape[1], int(corners2[i][0][1]))
                    cv2.line(combined, pt1, pt2, (0, 255, 0), 1)
                
                cv2.imshow('Stereo detekcia šachovnice a prepojenia', combined)
                cv2.waitKey(100)
            else:
                print(f"Šachovnica nebola detekovaná v jednom z párov obrázkov: {os.path.basename(ref_img_path)}")
        
        cv2.destroyAllWindows()
        
        if len(objpoints) == 0:
            print("CHYBA: Nebola úspešne spracovaná žiadna stereo snímka!")
            return None
        
        print(f"Úspešne spracovaných {len(objpoints)} stereo párov")
        
        # Získanie kalibračných údajov pre jednotlivé kamery
        ref_mtx = ref_calib['camera_matrix']
        ref_dist = ref_calib['distortion_coefficients']
        target_mtx = target_calib['camera_matrix']
        target_dist = target_calib['distortion_coefficients']
        
        # Vykonanie stereokalibrácie
        print(f"Výpočet stereokalibračných parametrov...")
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret, ref_mtx, ref_dist, target_mtx, target_dist, R, T, E, F = cv2.stereoCalibrate(
            objpoints, ref_imgpoints, target_imgpoints,
            ref_mtx, ref_dist,
            target_mtx, target_dist,
            ref_gray.shape[::-1],
            flags=flags
        )
        
        # Výpočet rektifikačných transformácií
        R1, R2, P1, P2, Q, roi_ref, roi_target = cv2.stereoRectify(
            ref_mtx, ref_dist,
            target_mtx, target_dist,
            ref_gray.shape[::-1], R, T
        )
        
        # Uloženie stereokalibračných údajov do YAML
        stereo_data = {
            'rotation_matrix': R.tolist(),
            'translation_vector': T.tolist(),
            'essential_matrix': E.tolist(),
            'fundamental_matrix': F.tolist(),
            'rectification_transform_1': R1.tolist(),
            'rectification_transform_2': R2.tolist(),
            'projection_matrix_1': P1.tolist(),
            'projection_matrix_2': P2.tolist(),
            'disparity_to_depth_mapping': Q.tolist(),
            'stereo_calibration_error': float(ret)
        }
        
        stereo_file = os.path.join(
            self.calib_data_dir, 
            f"stereo_{ref_cam_name}_to_{target_cam_name}_calibration.yaml"
        )
        with open(stereo_file, 'w') as f:
            yaml.dump(stereo_data, f)
        
        print(f"Stereokalibračné údaje uložené do {stereo_file}")
        print(f"Chyba stereokalibrácie (RMS): {ret}")
        
        return stereo_data

    def test_stereo_rectification(self, ref_camera_idx, target_camera_idx, stereo_data):
        """
        Test rektifikácie stereo páru obrázkov
        
        Parametre:
        - ref_camera_idx: index referenčnej kamery
        - target_camera_idx: index cieľovej kamery
        - stereo_data: výsledky stereokalibrácie
        """
        ref_cam_name = self.camera_names[ref_camera_idx]
        target_cam_name = self.camera_names[target_camera_idx]
        
        print(f"\nTest stereo rektifikácie: {ref_cam_name} -> {target_cam_name}")
        
        # Načítanie kalibračných údajov
        ref_calib_file = os.path.join(self.calib_data_dir, f"{ref_cam_name}_calibration.yaml")
        target_calib_file = os.path.join(self.calib_data_dir, f"{target_cam_name}_calibration.yaml")
        
        with open(ref_calib_file, 'r') as f:
            ref_calib = yaml.safe_load(f)
        
        with open(target_calib_file, 'r') as f:
            target_calib = yaml.safe_load(f)
        
        # Konverzia na numpy polia
        ref_mtx = np.array(ref_calib['camera_matrix'])
        ref_dist = np.array(ref_calib['distortion_coefficients'])
        target_mtx = np.array(target_calib['camera_matrix'])
        target_dist = np.array(target_calib['distortion_coefficients'])
        
        # Získanie rektifikačných matíc
        R = np.array(stereo_data['rotation_matrix'])
        T = np.array(stereo_data['translation_vector'])
        R1 = np.array(stereo_data['rectification_transform_1'])
        R2 = np.array(stereo_data['rectification_transform_2'])
        P1 = np.array(stereo_data['projection_matrix_1'])
        P2 = np.array(stereo_data['projection_matrix_2'])
        
        # Načítanie stereo obrázkov pre test
        ref_images_pattern = os.path.join(self.images_dir, ref_cam_name, "stereo_*.png")
        target_images_pattern = os.path.join(self.images_dir, target_cam_name, "stereo_*.png")
        
        ref_images = sorted(glob.glob(ref_images_pattern))
        target_images = sorted(glob.glob(target_images_pattern))
        
        if len(ref_images) == 0 or len(target_images) == 0:
            print("CHYBA: Neboli nájdené stereo obrázky pre test!")
            return
        
        # Použijeme prvý pár obrázkov pre test
        ref_img = cv2.imread(ref_images[0])
        target_img = cv2.imread(target_images[0])
        
        # Výpočet rektifikačných máp
        h, w = ref_img.shape[:2]
        
        mapL1, mapL2 = cv2.initUndistortRectifyMap(ref_mtx, ref_dist, R1, P1, (w, h), cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(target_mtx, target_dist, R2, P2, (w, h), cv2.CV_32FC1)
        
        # Aplikácia rektifikácie
        rectL = cv2.remap(ref_img, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(target_img, mapR1, mapR2, cv2.INTER_LINEAR)
        
        # Pridanie horizontálnych čiar pre vizuálnu kontrolu rektifikácie
        for i in range(0, h, 30):
            cv2.line(rectL, (0, i), (w, i), (0, 0, 255), 1)
            cv2.line(rectR, (0, i), (w, i), (0, 0, 255), 1)
        
        # Zobrazenie výsledkov
        orig_pair = np.hstack((ref_img, target_img))
        rect_pair = np.hstack((rectL, rectR))
        
        cv2.imshow('Pôvodný stereo pár', orig_pair)
        cv2.imshow('Rektifikovaný stereo pár', rect_pair)
        
        print("Stlač ľubovoľný kláves pre ukončenie zobrazenia...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_calibration_pipeline(self):

        print("\n=== KALIBRAČNÁ PIPELINE ===")
        
        # 1. Zachytávanie obrázkov pre jednotlivé kamery
        print("\n=== KROK 1: Zachytávanie obrázkov pre kalibráciu jednotlivých kamier ===")
        # self.capture_single_camera_images()
        
        # 2. Kalibrácia jednotlivých kamier
        print("\n=== KROK 2: Kalibrácia jednotlivých kamier ===")
        # calibration_results = self.calibrate_all_cameras()

        # ked uz kalibraciu mas zrobenu tak zavolas uz vytvorene yaml subory

        calibration_files = [
            "./calibration1000/calib_data/left_calibration.yaml",
            "./calibration1000/calib_data/middle_calibration.yaml",
            "./calibration1000/calib_data/right_calibration.yaml"
        ]
        calibration_results = self.load_calibration_data(calibration_files)

         # 3. Zachytenie stereo obrázkov
        print("\n=== KROK 3: Zachytávanie stereo obrázkov === left-middle")
        if not self.capture_stereo_images(self.camera_indices[1],self.camera_indices[0],"left",calibration_results):
            print("CHYBA: Nepodarilo sa zachytiť stereo obrázky!")
            return
        
        #print("\n=== KROK 3: Zachytávanie stereo obrázkov === right-middle")
        if not self.capture_stereo_images(self.camera_indices[1],self.camera_indices[2],"right",calibration_results):
            print("CHYBA: Nepodarilo sa zachytiť stereo obrázky!")
            return
        
        # 4. Stereokalibrácia
        print("\n=== KROK 4: Stereokalibrácia medzi kamerami ===")
        # Vykonanie stereokalibrácie pre všetky páry kamier s referenčnou kamerou (strednou)
        ref_camera_idx = 1  # stredná kamera ako referenčná
        stereo_results = {}
        
        for i in range(len(self.camera_indices)):
            if i != ref_camera_idx:
                stereo_data = self.perform_stereo_calibration(ref_camera_idx, i, calibration_results)
                if stereo_data is not None:
                    pair_key = f"{self.camera_names[ref_camera_idx]}_to_{self.camera_names[i]}"
                    stereo_results[pair_key] = stereo_data
        
        print("\n=== KALIBRÁCIA DOKONČENÁ ===")
        print(f"Všetky kalibračné údaje boli uložené do: {self.calib_data_dir}")

    def test_depth_map(self, ref_camera_idx, target_camera_idx, stereo_data):
        """
        Test vytvorenia hĺbkovej mapy z rektifikovaného stereo páru
        
        Parametre:
        - ref_camera_idx: index referenčnej kamery
        - target_camera_idx: index cieľovej kamery
        - stereo_data: výsledky stereokalibrácie
        """
        ref_cam_name = self.camera_names[ref_camera_idx]
        target_cam_name = self.camera_names[target_camera_idx]
        
        print(f"\nTest hĺbkovej mapy: {ref_cam_name} -> {target_cam_name}")
        
        # Načítanie kalibračných údajov
        ref_calib_file = os.path.join(self.calib_data_dir, f"{ref_cam_name}_calibration.yaml")
        target_calib_file = os.path.join(self.calib_data_dir, f"{target_cam_name}_calibration.yaml")
        
        with open(ref_calib_file, 'r') as f:
            ref_calib = yaml.safe_load(f)
        
        with open(target_calib_file, 'r') as f:
            target_calib = yaml.safe_load(f)
        
        # Konverzia na numpy polia
        ref_mtx = np.array(ref_calib['camera_matrix'])
        ref_dist = np.array(ref_calib['distortion_coefficients'])
        target_mtx = np.array(target_calib['camera_matrix'])
        target_dist = np.array(target_calib['distortion_coefficients'])
        
        # Získanie rektifikačných matíc
        R = np.array(stereo_data['rotation_matrix'])
        T = np.array(stereo_data['translation_vector'])
        R1 = np.array(stereo_data['rectification_transform_1'])
        R2 = np.array(stereo_data['rectification_transform_2'])
        P1 = np.array(stereo_data['projection_matrix_1'])
        P2 = np.array(stereo_data['projection_matrix_2'])
        Q = np.array(stereo_data['disparity_to_depth_mapping'])
        
        # Načítanie stereo obrázkov pre test
        ref_images_pattern = os.path.join(self.images_dir, ref_cam_name, "stereo_*.png")
        target_images_pattern = os.path.join(self.images_dir, target_cam_name, "stereo_*.png")
        
        ref_images = sorted(glob.glob(ref_images_pattern))
        target_images = sorted(glob.glob(target_images_pattern))
        
        if len(ref_images) == 0 or len(target_images) == 0:
            print("CHYBA: Neboli nájdené stereo obrázky pre test!")
            return
        
        # Použijeme prvý pár obrázkov pre test
        ref_img = cv2.imread(ref_images[0])
        target_img = cv2.imread(target_images[0])
        
        # Výpočet rektifikačných máp
        h, w = ref_img.shape[:2]
        
        mapL1, mapL2 = cv2.initUndistortRectifyMap(ref_mtx, ref_dist, R1, P1, (w, h), cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(target_mtx, target_dist, R2, P2, (w, h), cv2.CV_32FC1)
        
        # Aplikácia rektifikácie
        rectL = cv2.remap(ref_img, mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(target_img, mapR1, mapR2, cv2.INTER_LINEAR)
        
        # Konverzia na grayscale pre výpočet disparity
        grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        
        # Vytvorenie disparity mapy
        stereo = cv2.StereoBM_create(numDisparities=16*16, blockSize=21)
        disparity = stereo.compute(grayL, grayR)
        
        # Normalizácia disparity pre zobrazenie
        norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Vytvorenie 3D point cloud
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        
        # Zobrazenie výsledkov
        cv2.imshow('Ľavý rektifikovaný obrázok', rectL)
        cv2.imshow('Pravý rektifikovaný obrázok', rectR)
        cv2.imshow('Disparity mapa', norm_disparity)
        
        # Uloženie disparity mapy a 3D bodov
        disparity_file = os.path.join(
            self.calib_data_dir, 
            f"disparity_{ref_cam_name}_to_{target_cam_name}.png"
        )
        cv2.imwrite(disparity_file, norm_disparity)
        
        # Vytvorenie farebnej disparity mapy
        disparity_color = cv2.applyColorMap(norm_disparity, cv2.COLORMAP_JET)
        disparity_color_file = os.path.join(
            self.calib_data_dir, 
            f"disparity_color_{ref_cam_name}_to_{target_cam_name}.png"
        )
        cv2.imwrite(disparity_color_file, disparity_color)
        
        print(f"Disparity mapa uložená do {disparity_file} a {disparity_color_file}")
        print("Stlač ľubovoľný kláves pre ukončenie zobrazenia...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calculate_extrinsic_parameters(self):
        """
        Výpočet extrinsických parametrov pre všetky kamery
        
        Extrinsické parametre popisujú pozíciu a orientáciu kamier vo vzťahu k referenčnej kamere.
        """
        print("\nVýpočet extrinsických parametrov kamier")
        
        # Použitie strednej kamery ako referenčnej
        ref_camera_idx = 1
        ref_cam_name = self.camera_names[ref_camera_idx]
        
        extrinsic_params = {
            ref_cam_name: {
                'rotation': np.eye(3).tolist(),  # Identitná matica pre referenčnú kameru
                'translation': np.zeros(3).tolist()  # Nulový vektor pre referenčnú kameru
            }
        }
        
        # Načítanie stereokalibračných dát pre ostatné kamery
        for i in range(len(self.camera_indices)):
            if i != ref_camera_idx:
                target_cam_name = self.camera_names[i]
                stereo_file = os.path.join(
                    self.calib_data_dir, 
                    f"stereo_{ref_cam_name}_to_{target_cam_name}_calibration.yaml"
                )
                
                if not os.path.exists(stereo_file):
                    print(f"UPOZORNENIE: Chýba stereokalibračný súbor pre {target_cam_name}")
                    continue
                
                with open(stereo_file, 'r') as f:
                    stereo_data = yaml.safe_load(f)
                
                R = np.array(stereo_data['rotation_matrix'])
                T = np.array(stereo_data['translation_vector'])
                
                extrinsic_params[target_cam_name] = {
                    'rotation': R.tolist(),
                    'translation': T.tolist()
                }
        
        # Uloženie extrinsických parametrov do YAML súboru
        extrinsic_file = os.path.join(self.calib_data_dir, "extrinsic_parameters.yaml")
        with open(extrinsic_file, 'w') as f:
            yaml.dump(extrinsic_params, f)
        
        print(f"Extrinsické parametre kamier uložené do {extrinsic_file}")
        return extrinsic_params

    def generate_calibration_report(self):
        """
        Vytvorenie súhrnnej správy o kalibrácii
        """
        print("\nGenerovanie kalibračnej správy")
        
        report = {
            'system_info': {
                'num_cameras': len(self.camera_indices),
                'camera_indices': self.camera_indices,
                'camera_names': self.camera_names,
                'frame_size': self.frame_size,
                'checkerboard_size': self.checkerboard_size,
                'square_size': self.square_size,
                'calibration_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'intrinsic_calibration': {},
            'stereo_calibration': {}
        }
        
        # Načítanie kalibračných údajov pre jednotlivé kamery
        for cam_name in self.camera_names:
            calib_file = os.path.join(self.calib_data_dir, f"{cam_name}_calibration.yaml")
            if os.path.exists(calib_file):
                with open(calib_file, 'r') as f:
                    calib_data = yaml.safe_load(f)
                report['intrinsic_calibration'][cam_name] = calib_data
        
        # Načítanie stereokalibračných údajov
        ref_camera_idx = 1  # stredná kamera ako referenčná
        ref_cam_name = self.camera_names[ref_camera_idx]
        
        for i in range(len(self.camera_indices)):
            if i != ref_camera_idx:
                target_cam_name = self.camera_names[i]
                stereo_file = os.path.join(
                    self.calib_data_dir, 
                    f"stereo_{ref_cam_name}_to_{target_cam_name}_calibration.yaml"
                )
                
                if os.path.exists(stereo_file):
                    with open(stereo_file, 'r') as f:
                        stereo_data = yaml.safe_load(f)
                    
                    pair_key = f"{ref_cam_name}_to_{target_cam_name}"
                    report['stereo_calibration'][pair_key] = {
                        'stereo_calibration_error': stereo_data['stereo_calibration_error'],
                        'baseline': np.linalg.norm(np.array(stereo_data['translation_vector'])).item()
                    }
        
        # Uloženie správy do YAML súboru
        report_file = os.path.join(self.calib_data_dir, "calibration_report.yaml")
        with open(report_file, 'w') as f:
            yaml.dump(report, f)
        
        # Vytvorenie HTML správy pre lepšiu čitateľnosť
        html_report = os.path.join(self.calib_data_dir, "calibration_report.html")
    
        print(f"Kalibračná správa vygenerovaná v {report_file} a {html_report}")
        return report