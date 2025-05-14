import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from scipy.interpolate import griddata

def create_calibration_pattern(num_points=20):
    """
    Vytvorí kalibračný vzor pre testovanie detekcie.
    Vytvára zoznam bodov rozmiestnených po celom terči v rôznych vzdialenostiach.
    """
    # Vytvorte zoznam bodov v polárnych súradniciach
    calibration_points = []
    
    # Rôzne vzdialenosti od stredu
    distances = [20, 40, 70, 100, 130, 160]  # v milimetroch
    
    # Počet bodov na každom kruhu
    points_per_circle = num_points
    
    # Vygenerovať body pre každú vzdialenosť
    for dist in distances:
        for i in range(points_per_circle):
            angle = i * (360 / points_per_circle)
            # Konverzia z polárnych na kartézske súradnice
            x = dist * np.cos(np.radians(angle))
            y = dist * np.sin(np.radians(angle))
            calibration_points.append((x, y, angle, dist))
    
    return calibration_points

def display_calibration_pattern(image, center_px, points, scale):
    """
    Zobrazí kalibračný vzor na terči pre vizuálnu kontrolu.
    """
    display_img = image.copy()
    
    # Vykreslite stred terča
    cv2.circle(display_img, center_px, 5, (0, 0, 255), -1)
    
    # Zobrazte všetky kalibračné body
    for x, y, angle, dist in points:
        # Prepočítajte na súradnice v pixeloch
        px = int(center_px[0] + x * scale)
        py = int(center_px[1] - y * scale)
        
        # Vykreslite bod
        cv2.circle(display_img, (px, py), 3, (255, 0, 0), -1)
        
        # Voliteľne: Pridajte číslo bodu
        # cv2.putText(display_img, f"{int(angle)}°", (px+5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    return display_img

def capture_reference_images(video_captures, cameras):
    images = []
    for i, cap in enumerate(video_captures):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Kamera {i} nezachytila obraz.")
            images.append(None)
            continue

        mapx = cameras[i]["mapx"]
        mapy = cameras[i]["mapy"]
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        images.append(undistorted)
    return images

def run_calibration_test(cameras, calibration_data, cams, original_target_image, 
                         center_3d, center_px, scale, output_dir="calibration_results"):
    """
    Spustí kalibračný test a zbiera dáta.
    """
    # Vytvorte adresár pre výstup, ak neexistuje
    os.makedirs(output_dir, exist_ok=True)
    
    # Vytvorte kalibračný vzor
    calibration_points = create_calibration_pattern()
    
    # Zobrazte kalibračný vzor na terči
    pattern_img = display_calibration_pattern(original_target_image, center_px, calibration_points, scale)
    cv2.imwrite(os.path.join(output_dir, "calibration_pattern.png"), pattern_img)
    
    # Dátové štruktúry pre ukladanie výsledkov
    results = []
    
    # Zobrazte kalibračný vzor s inštrukciami
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB))
    plt.title("Kalibračný vzor: Hoďte šípky do označených bodov")
    plt.show()
    
    print("Začína kalibračný test:")
    print("1. Postupne hádžte šípky do označených bodov na terči")
    print("2. Po každom hode stlačte klávesu 'ENTER' pre zachytenie pozície")
    print("3. Ak šípka nezasiahla žiadny z bodov, stlačte 'S' na preskočenie")
    print("4. Na ukončenie kalibračného testu stlačte 'Q'")
    
    from DartDetection import dartDetection
    from triangulation import triangulate_shots
    
    # Hlavný kalibračný cyklus
    point_index = 0
    while point_index < len(calibration_points):
        # Zobrazte aktuálny cieľový bod
        curr_point = calibration_points[point_index]
        x, y, angle, dist = curr_point
        
        print(f"\nCieľový bod #{point_index+1}: Uhol {angle}°, Vzdialenosť {dist}mm")
        
        # Čakajte na vstup od používateľa
        key = input("Stlačte ENTER po hodení šípky, 'S' na preskočenie, 'Q' na ukončenie: ")
        
        if key.upper() == 'Q':
            break
        
        if key.upper() == 'S':
            point_index += 1
            continue
        
        video_captures = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in cams]

        ref_images = capture_reference_images(video_captures, cameras)
        # Zachytite pozíciu šípky
        time.sleep(0.5)
        dart_position = dartDetection(video_captures, cameras, cams, ref_images)
        if dart_position is None or not any(dart_position):
            print("Šípka nebola detekovaná, skúste znova.")
            continue
        
        # Triangulujte pozíciu šípky
        triangulated_position = triangulate_shots(dart_position)
        if triangulated_position is None:
            print("Nepodarilo sa triangulovať pozíciu šípky, skúste znova.")
            continue
        
        # Vypočítajte skutočnú vzdialenosť a uhol
        relative = np.array(triangulated_position) - center_3d
        
        # Vypočítajte vzdialenosť a uhol (v rovine X-Z)
        actual_dist = np.linalg.norm([relative[0], relative[2]])
        actual_angle = (np.degrees(np.arctan2(-relative[2], relative[0])) + 360) % 360
        
        # Uložte výsledky
        results.append({
            "target_x": x,
            "target_y": y,
            "target_angle": angle,
            "target_distance": dist,
            "actual_x": relative[0],
            "actual_y": relative[2],
            "actual_angle": actual_angle,
            "actual_distance": actual_dist,
            "error_x": relative[0] - x,
            "error_y": relative[2] - y,
            "error_angle": (actual_angle - angle + 180) % 360 - 180,
            "error_distance": actual_dist - dist
        })
        
        print(f"Záznam: Cieľ ({x:.1f}, {y:.1f}), Skutočnosť ({relative[0]:.1f}, {relative[2]:.1f})")
        print(f"Chyba: Vzdialenosť {actual_dist - dist:.1f}mm, Uhol {((actual_angle - angle + 180) % 360 - 180):.1f}°")
        
        # Prejdite na ďalší bod
        point_index += 1
    
    # Uložte výsledky do CSV súboru
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "calibration_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nVýsledky kalibrácie uložené do {csv_path}")
        
        # Vytvorte korekčnú mapu
        create_correction_map(df, output_dir)
    else:
        print("Žiadne dáta na analýzu.")

def create_correction_map(results_df, output_dir):
    """
    Vytvorí korekčnú mapu na základe výsledkov kalibrácie.
    """
    # Extrahujte dáta
    points = results_df[['target_x', 'target_y']].values
    dx = results_df['error_x'].values
    dy = results_df['error_y'].values
    
    # Vytvorte mriežku pre interpoláciu
    grid_x, grid_y = np.mgrid[-170:170:20j, -170:170:20j]
    
    # Interpolujte chyby na mriežke
    grid_dx = griddata(points, dx, (grid_x, grid_y), method='cubic', fill_value=0)
    grid_dy = griddata(points, dy, (grid_y, grid_y), method='cubic', fill_value=0)
    
    # Odstráňte NaN hodnoty
    grid_dx = np.nan_to_num(grid_dx)
    grid_dy = np.nan_to_num(grid_dy)
    
    # Vytvorte korekčnú mapu
    correction_map = {
        'grid_x': grid_x.tolist(),
        'grid_y': grid_y.tolist(),
        'dx': grid_dx.tolist(),
        'dy': grid_dy.tolist()
    }
    
    # Uložte korekčnú mapu do súboru
    import json
    with open(os.path.join(output_dir, 'correction_map.json'), 'w') as f:
        json.dump(correction_map, f)
    
    # Vytvorte vizualizáciu korekčnej mapy
    plt.figure(figsize=(12, 10))
    
    # Veľkosť chyby ako farba
    error_magnitude = np.sqrt(grid_dx**2 + grid_dy**2)
    
    # Vykreslite chyby ako vektorové pole
    plt.quiver(grid_x, grid_y, grid_dx, grid_dy, error_magnitude, 
               angles='xy', scale_units='xy', scale=0.2)
    plt.colorbar(label='Veľkosť chyby (mm)')
    
    # Pridajte stred a kruhy pre vizualizáciu terča
    plt.plot(0, 0, 'ro', markersize=10)
    circle_radii = [7, 17, 97, 107, 160, 170]
    for r in circle_radii:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--')
        plt.gca().add_patch(circle)
    
    plt.grid(True)
    plt.axis('equal')
    plt.title('Korekčná mapa pre detekciu šípok')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    
    # Uložte vizualizáciu
    plt.savefig(os.path.join(output_dir, 'correction_map.png'))
    plt.close()
    
    print(f"Korekčná mapa vytvorená a uložená do {output_dir}")

def analyze_calibration_results(output_dir="calibration_results"):
    """
    Analyzuje výsledky kalibrácie a vytvorí korekčné funkcie.
    """
    # Načítajte výsledky
    csv_path = os.path.join(output_dir, "calibration_results.csv")
    if not os.path.exists(csv_path):
        print(f"Súbor s výsledkami neexistuje: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Vytvorte grafy chýb
    plt.figure(figsize=(15, 10))
    
    # 1. Graf chyby vzdialenosti vs. skutočná vzdialenosť
    plt.subplot(2, 2, 1)
    plt.scatter(df['actual_distance'], df['error_distance'], alpha=0.7)
    
    # Pridajte regresný model
    from scipy.optimize import curve_fit
    
    def correction_model(x, a, b, c):
        return a * x**2 + b * x + c
    
    popt, _ = curve_fit(correction_model, df['actual_distance'], df['error_distance'])
    
    # Vytvorte funkciu pre korekciu vzdialenosti
    def distance_correction(distance):
        return distance - correction_model(distance, *popt)
    
    # Zobrazte model korekcie
    x_fit = np.linspace(0, 200, 100)
    y_fit = correction_model(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', label=f'Model: {popt[0]:.6f}x² + {popt[1]:.4f}x + {popt[2]:.2f}')
    
    plt.grid(True)
    plt.title('Chyba vzdialenosti vs. skutočná vzdialenosť')
    plt.xlabel('Skutočná vzdialenosť (mm)')
    plt.ylabel('Chyba vzdialenosti (mm)')
    plt.legend()
    
    # 2. Graf chyby uhla vs. skutočný uhol
    plt.subplot(2, 2, 2)
    plt.scatter(df['actual_angle'], df['error_angle'], alpha=0.7)
    plt.grid(True)
    plt.title('Chyba uhla vs. skutočný uhol')
    plt.xlabel('Skutočný uhol (°)')
    plt.ylabel('Chyba uhla (°)')
    
    # 3. Priestorová distribúcia chýb
    plt.subplot(2, 2, 3)
    plt.quiver(df['actual_x'], df['actual_y'], 
               df['error_x'], df['error_y'], 
               angles='xy', scale_units='xy', scale=0.5)
    
    # Pridajte vizualizáciu terča
    circle_radii = [7, 17, 97, 107, 160, 170]
    for r in circle_radii:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--')
        plt.gca().add_patch(circle)
    
    plt.grid(True)
    plt.axis('equal')
    plt.title('Priestorová distribúcia chýb')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    
    # 4. Histogram absolútnych chýb
    plt.subplot(2, 2, 4)
    abs_errors = np.sqrt(df['error_x']**2 + df['error_y']**2)
    plt.hist(abs_errors, bins=20)
    plt.grid(True)
    plt.title('Histogram absolútnych chýb')
    plt.xlabel('Absolútna chyba (mm)')
    plt.ylabel('Počet')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_analysis.png'))
    plt.close()
    
    # Vytvorte korekčnú funkciu
    with open(os.path.join(output_dir, 'correction_function.py'), 'w') as f:
        f.write(f"""
def correct_distance(distance):
    \"\"\"
    Korekčná funkcia pre vzdialenosť na základe kalibračných dát.
    \"\"\"
    # Koeficienty získané z kalibrácie
    a, b, c = {popt[0]}, {popt[1]}, {popt[2]}
    
    # Vypočítajte korekciu
    correction = a * distance**2 + b * distance + c
    
    # Aplikujte korekciu
    corrected_distance = distance - correction
    
    return corrected_distance
""")

    print(f"Analýza kalibrácie dokončená. Výsledky uložené do {output_dir}")
    print(f"Korekčná funkcia vytvorená v {os.path.join(output_dir, 'correction_function.py')}")
    
    # Vypíšte informácie o presnosti
    mean_error = abs_errors.mean()
    max_error = abs_errors.max()
    print(f"Priemerná absolútna chyba: {mean_error:.2f} mm")
    print(f"Maximálna absolútna chyba: {max_error:.2f} mm")
    
    # Vytvorte a uložte korekčné koeficienty
    correction_coeffs = {
        'distance_correction': {
            'a': popt[0],
            'b': popt[1],
            'c': popt[2]
        }
    }
    
    import json
    with open(os.path.join(output_dir, 'correction_coefficients.json'), 'w') as f:
        json.dump(correction_coeffs, f)
    
    return correction_coeffs