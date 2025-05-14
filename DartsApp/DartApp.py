import streamlit as st
import cv2
import numpy as np
import time
import copy
from PIL import Image
import copy
import yaml

from DartDetection import dartDetection, capture_reference_images, load_calibration
from triangulation import triangulate_shots

# Konštanty
STARTING_SCORE = 501
MAX_SHOTS_PER_ROUND = 3
camera_files = [
    "./Calibration/calibration1000/calib_data/left_calibration.yaml",
    "./Calibration/calibration1000/calib_data/middle_calibration.yaml",
    "./Calibration/calibration1000/calib_data/right_calibration.yaml"
]
cams = [3, 1, 2] # toto su indexy kamier ktore si definujeme podla svojho systemu
frame_size = (1000, 1000)
target_radius_mm = 225
center_3d = np.array([19.10, -24.91, 414.20])  # Centrum terča v 3D priestore

# --- Výpočet mierky ---
center_px = np.array([502, 500])
edge_px = np.array([990, 501])
pixel_radius = np.linalg.norm(edge_px - center_px)
scale = pixel_radius / target_radius_mm  # px per mm

# Načítanie terča
original_target_image = cv2.imread("./Dartboard1000.png")

# Konštanty pre polomery jednotlivých oblastí v mm
BULLSEYE_RADIUS = 7
OUTER_BULL_RADIUS = 17
TRIPLE_INNER_RADIUS = 97
TRIPLE_OUTER_RADIUS = 107
DOUBLE_INNER_RADIUS = 160
DOUBLE_OUTER_RADIUS = 170

# Segment hodnoty
segment_values = [6, 10, 15, 2, 17, 3, 19, 7, 16, 8,
                     11, 14, 9, 12, 5, 20, 1, 18, 4, 13]
segment_size = 18 

@st.cache_resource(show_spinner=False)
def init_cameras():
    cameras = []
    for i in cams:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error(f"Nemožno otvoriť kameru {i}")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        cameras.append(cap)
    return cameras

def cleanup_cameras():
    if 'cameras' in st.session_state and st.session_state.cameras:
        for cam in st.session_state.cameras:
            cam.release()
        st.session_state.cameras = None

if 'cameras' not in st.session_state:
    st.session_state.cameras = init_cameras()
    if st.session_state.cameras is None:
        st.stop()
cameras = st.session_state.cameras

calibration_data = [load_calibration(file) for file in camera_files]

stereo_files = [
        "./Calibration/calibration1000/calib_data/stereo_middle_to_left_calibration.yaml",
        "./Calibration/calibration1000/calib_data/stereo_middle_to_right_calibration.yaml"
    ]

with open(stereo_files[0]) as f:
    left_stereo = yaml.safe_load(f)
R_left = np.array(left_stereo['rotation_matrix'])
T_left = np.array(left_stereo['translation_vector'])

def project_point(point_3d, center_3d, center_px, scale, y_gain=0.0045):
    relative = np.array(point_3d) - center_3d

    dx, dy, dz = relative

    # Kompenzácia podľa výšky
    dz += dy * y_gain * abs(dy) 

    x_img = center_px[0] + relative[0] * scale
    y_img = center_px[1] - relative[2] * scale * 1.12
    return int(round(x_img)), int(round(y_img))

def draw_shot_positions(image, positions_list, center_3d, center_px, scale):
    # Vytvorenie kópie obrázku, aby sme nemodifikovali originál
    result_image = copy.deepcopy(image)
    
    # Farby pre jednotlivé šípky
    colors = [(255, 0, 0)]
    
    # Vykreslenie každej šípky
    for i, position in enumerate(positions_list):
        if position is not None:
            x, y = project_point(position, center_3d, center_px, scale)
            cv2.circle(result_image, (x, y), 5, colors[0], -1)
            cv2.circle(result_image, (x, y), 7, (255, 255, 255), 2)
    
    return result_image

def edit_distance(distance):
    if distance >= 200:
        faktor = 0.86
    elif distance >= 190:
        faktor = 0.87
    elif distance >= 190:
        faktor = 0.88
    elif distance >= 170:
        faktor = 0.89
    elif distance >= 160:
        faktor = 0.90
    elif distance >= 150:
        faktor = 0.91
    elif distance >= 140:
        faktor = 0.92
    elif distance >= 130:
        faktor = 0.93
    elif distance >= 120: 
        faktor = 0.94
    elif distance >= 110:
        faktor = 0.95
    elif distance >= 100:
        faktor = 0.96
    elif distance >= 90:
        faktor = 0.98
    elif distance <= 80:
        faktor = 1
    else:
        faktor = 1  # predvolený faktor

    upravena_hodnota = distance * faktor
    return upravena_hodnota

def apply_correction_map(point_3d, center_3d, correction_map, grid_size=20):

    # Calculate relative position
    relative = np.array(point_3d) - center_3d
    
    # Extract horizontal components (x, z)
    x_mm, _, z_mm = relative
    
    # Find grid position
    grid_x = int(x_mm / grid_size)
    grid_z = int(z_mm / grid_size)
    
    # Get correction factor (default to 1.0 if outside map)
    correction = correction_map.get((grid_x, grid_z), 1.0)
    
    # Apply correction to distance while preserving angle
    distance = np.sqrt(x_mm**2 + z_mm**2)
    angle_rad = np.arctan2(z_mm, x_mm)
    
    corrected_distance = distance * correction
    
    # Convert back to cartesian coordinates
    corrected_x = corrected_distance * np.cos(angle_rad)
    corrected_z = corrected_distance * np.sin(angle_rad)
    
    # Create corrected 3D point (preserve y-coordinate)
    corrected_point = np.array([
        center_3d[0] + corrected_x,
        point_3d[1],  # Keep original y
        center_3d[2] + corrected_z
    ])
    
    return corrected_point  

def adjust_scoring_calculation(point_3d, center_3d):
    """
    Upravená funkcia pre výpočet skóre zohľadňujúca eliptickú deformáciu.
    
    Args:
        point_3d: 3D súradnice bodu dopadu šípky
        center_3d: 3D súradnice stredu terča
        x_scale: Horizontálna mierka
        y_scale: Vertikálna mierka
    
    Returns:
        int: Bodové ohodnotenie hodu
    """
    relative_vector = point_3d - center_3d
    distance = np.linalg.norm(relative_vector)

    # Rovinu berieme ako X-Z, kde Z ide "do hĺbky"
    angle_rad = np.arctan2(-relative_vector[2], relative_vector[0])
    angle_deg = (np.degrees(angle_rad) + 360) % 360
    angle_offset = 9

    #[ 10 15 2, 17 3, 19, 7,  16, 8,  11, 14, 9,  12, 5,  20, 1,  18, 4,  13  6] segment
    #[9 27 35 65 81 99 117 135 153 171 189 207 225 243 261 279 297 315 333 351 ] uhly
    segment_index = int((angle_deg + angle_offset) % 360 // segment_size)
    segment_value = segment_values[segment_index % 20]

    print(f"vzdialenost pred {distance}")

    distance2 = edit_distance(distance)

    if distance2 <= BULLSEYE_RADIUS:
        score = 50
    elif distance2 <= OUTER_BULL_RADIUS:
        score = 25
    elif distance2 <= TRIPLE_INNER_RADIUS:
        score = segment_value
    elif distance2 <= TRIPLE_OUTER_RADIUS:
        score = segment_value * 3
    elif distance2 <= DOUBLE_INNER_RADIUS:
        score = segment_value
    elif distance2 <= DOUBLE_OUTER_RADIUS:
        score = segment_value* 2
    else:
        score = 0

    print(f"Uhol: {angle_deg}° vzdialenost: {distance2}, skore: {score}")
    return score

def detect_dart(cameras, calibration_data):
    """Funkcia na detekciu šípky a výpočet skóre"""
    try:
        # st.session_state.current_image = cv2.remap(original_target_image, mapx, mapy, cv2.INTER_LINEAR) toto je pre mapovanie

        # Kontrola, či už nie je maximum hodov
        if st.session_state.shots_taken >= MAX_SHOTS_PER_ROUND:
            return False
        
        # Detekcia šípky
        ref_images = capture_reference_images(cameras, calibration_data)

        time.sleep(0.5)
        dartPosition = dartDetection(cameras, calibration_data, cams, ref_images)
        if dartPosition is None or not any(dartPosition):
            return False
            
        triangulated = triangulate_shots(dartPosition)
        if triangulated is None:
            return False
        
        # Aplikuj korekciu 
        #print(triangulated)
        #corrected_3d = apply_correction_map(triangulated, center_3d, st.session_state.correction_map)
        #print(corrected_3d)
        
        ## Výpočet skóre s korekciou
        shot_score = adjust_scoring_calculation(triangulated, center_3d)
        
        # Pridanie pozície šípky a aktualizácia skóre
       
        st.session_state.round_scores.append(shot_score)
        st.session_state.dart_positions.append(triangulated)
        
        # Určenie aktuálneho hráča
        current_player = "player1" if st.session_state.current_player == 1 else "player2"
        
        # Uloženie pozície šípky a skóre
        shot_index = st.session_state.shots_taken
        st.session_state[f"{current_player}_throw_{shot_index+1}_position"] = triangulated
        st.session_state[f"{current_player}_throw_{shot_index+1}_score"] = shot_score
        
        # Uloženie pôvodného skóre pred hodmi v tomto kole, ak ešte nie je uložené
        if shot_index == 0:  # Len pre prvý hod v kole
            st.session_state[f"{current_player}_original_score"] = st.session_state[f"{current_player}_score"]
        
        # Výpočet nového skóre (ale zatiaľ ho neaplikujeme)
        new_score = st.session_state[f"{current_player}_score"] - shot_score

        # Zvýšime počet hodených šípok
        st.session_state.shots_taken += 1
        
        # Aktualizujeme obraz s pozíciami
        st.session_state.current_image = draw_shot_positions(
            original_target_image,
            st.session_state.dart_positions,
            center_3d,
            center_px,
            scale,
        )
      
        
        timestamp = int(time.time())
        #cv2.imwrite(f"../DetectedDartsFoto/testing/testing{timestamp}.png", st.session_state.current_image)
        # Predbežne aktualizujeme skóre (môže byť upravené pri review)
        st.session_state[f"{current_player}_score"] = new_score
        
        # Kontrola, či je nové skóre 0 (víťazstvo) alebo pod 0 (bust)
        if new_score < 0:
            st.session_state.potential_bust = True
            st.session_state.ready_to_review = True  # Automaticky pripraviť na review
        elif new_score == 0:
            st.session_state.potential_win = True
            st.session_state.ready_to_review = True  # Automaticky pripraviť na review
        
        # Kontrola, či treba pripraviť na kontrolu hodov
        # Táto kontrola už zahŕňa aj prípady potential_bust a potential_win z predchádzajúcej podmienky
        if st.session_state.shots_taken >= MAX_SHOTS_PER_ROUND:
            st.session_state.ready_to_review = True
        
        # Vrátime True ak sa úspešne detekoval hod
        return True
        
    except Exception as e:
        st.error(f"Chyba pri detekcii: {str(e)}")
        return False
    
def calculate_current_score():
    """Vypočíta aktuálne skóre na základe hodov v kole"""
    current_player = "player1" if st.session_state.current_player == 1 else "player2"
    original_score = st.session_state[f"{current_player}_original_score"]
    
    # Súčet bodov za všetky hody v tomto kole
    round_points = sum(st.session_state.round_scores)
    
    # Nové skóre
    new_score = original_score - round_points
    
    return new_score, original_score, round_points

def update_throw_score(player, throw_number, new_score):
    """Aktualizuje skóre konkrétneho hodu a prepočíta celkové skóre"""
    current_player = f"player{player}"
    throw_key = f"{current_player}_throw_{throw_number}_score"
    
    if throw_key in st.session_state:
        # Uloženie nového skóre pre tento hod
        st.session_state[throw_key] = new_score
        
        # Aktualizácia round_scores pre zobrazenie
        if player == st.session_state.current_player:
            for i, _ in enumerate(st.session_state.round_scores):
                if i == throw_number - 1:
                    st.session_state.round_scores[i] = new_score
        
        # Prepočítanie celkového skóre
        current_score, original_score, _ = calculate_current_score()
        
        # Kontrola potenciálneho prekročenia (bust)
        if current_score < 0:
            st.session_state.potential_bust = True
        else:
            st.session_state.potential_bust = False
            if st.session_state.ready_for_next_player:
                st.session_state.ready_for_next_player = False
            
        # Kontrola potenciálneho víťazstva
        if current_score == 0:
            st.session_state.potential_win = True
        else:
            st.session_state.potential_win = False
        
        # Aktualizácia skóre hráča - ešte nemusí byť finálne, môže sa zmeniť pri potvrdení kola
        st.session_state[f"{current_player}_score"] = current_score

def finalize_round():
    """Potvrdiť skóre kola a pripraviť sa na prepnutie hráča"""
    current_player = "player1" if st.session_state.current_player == 1 else "player2"
    current_score, original_score, _ = calculate_current_score()
    
    # Ak ide o bust (prekročenie cez 0)
    if current_score < 0:
        # Obnovenie pôvodného skóre
        st.session_state[f"{current_player}_score"] = original_score
        st.session_state.bust_round = True
    # Ak je presne 0 (výhra)
    elif current_score == 0:
        st.session_state.game_over = True
        st.session_state.winner = st.session_state.current_player
    
    # V každom prípade pokračujeme na ďalšieho hráča
    st.session_state.ready_for_next_player = True
    st.session_state.ready_to_review = False

def switch_to_next_player():
    """Prepnutie na ďalšieho hráča a reset pre nové kolo"""
    # Reset príznakovej premennej pre bust
    st.session_state.bust_round = False
    st.session_state.potential_bust = False
    st.session_state.potential_win = False
    
    # Zmena aktuálneho hráča
    st.session_state.current_player = 2 if st.session_state.current_player == 1 else 1
    
    # Ak sme dokončili kolo (oba hráči už hrali), zvýšime počítadlo kôl
    if st.session_state.current_player == 1:
        st.session_state.round_number += 1
    
    # Reset pre nové kolo
    st.session_state.shots_taken = 0
    st.session_state.dart_positions = []
    st.session_state.round_scores = []
    st.session_state.current_image = copy.deepcopy(original_target_image)
    st.session_state.ready_for_next_player = False
    st.session_state.ready_to_review = False
    st.session_state.last_detection_time = time.time()
    
    # Reset hodnôt pre originálne skóre pred kolom
    current_player = "player1" if st.session_state.current_player == 1 else "player2"
    st.session_state[f"{current_player}_original_score"] = st.session_state[f"{current_player}_score"]

def main():
    st.set_page_config(page_title="Šípky – Detekcia", layout="wide")

    # Inicializácia stavových premenných v session_state
    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
        st.session_state.last_detection_time = time.time()
        st.session_state.detect_enabled = True  # Automatická detekcia je predvolene zapnutá
        st.session_state.shots_taken = 0
        st.session_state.player1_score = STARTING_SCORE
        st.session_state.player2_score = STARTING_SCORE
        st.session_state.player1_original_score = STARTING_SCORE
        st.session_state.player2_original_score = STARTING_SCORE
        st.session_state.current_player = 1
        st.session_state.dart_positions = []
        st.session_state.round_scores = []
        st.session_state.current_image = copy.deepcopy(original_target_image)
        st.session_state.game_over = False
        st.session_state.ready_for_next_player = False
        st.session_state.ready_to_review = False
        st.session_state.winner = None
        st.session_state.last_successful_detection = None
        st.session_state.bust_round = False
        st.session_state.potential_bust = False  # Potenciálne prekročenie (treba ešte potvrdiť)
        st.session_state.potential_win = False   # Potenciálne víťazstvo (treba ešte potvrdiť)
        st.session_state.round_number = 1
        
        # Inicializácia premenných pre jednotlivé hody
        for player in [1, 2]:
            for throw in range(1, MAX_SHOTS_PER_ROUND + 1):
                st.session_state[f"player{player}_throw_{throw}_position"] = None
                st.session_state[f"player{player}_throw_{throw}_score"] = 0

    # Hlavné menu, keď hra nie je spustená
    if not st.session_state.game_started:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Vitajte v hre šípky pre dvoch hráčov!")
            st.markdown("""
            **Pravidlá hry:**
            1. Každý hráč začína so skóre 501 bodov
            2. Hráči sa striedajú, každý má v kole 3 hody
            3. Cieľom je dostať skóre presne na 0
            4. Víťazí hráč, ktorý prvý dosiahne presne 0 bodov
            5. Ak hráč prehodí (dostane sa pod 0), všetky hody v danom kole sa anulujú
            """)

            start_col, exit_col = st.columns([1, 1])

            with start_col:
                if st.button("▶️ Spustiť hru", key="start_game", use_container_width=True):
                    st.session_state.game_started = True
                    st.session_state.player1_score = STARTING_SCORE
                    st.session_state.player2_score = STARTING_SCORE
                    st.session_state.player1_original_score = STARTING_SCORE
                    st.session_state.player2_original_score = STARTING_SCORE
                    st.session_state.current_player = 1
                    st.session_state.shots_taken = 0
                    st.session_state.dart_positions = []
                    st.session_state.round_scores = []
                    st.session_state.current_image = copy.deepcopy(original_target_image)
                    st.session_state.game_over = False
                    st.session_state.ready_for_next_player = False
                    st.session_state.ready_to_review = False
                    st.session_state.detect_enabled = True
                    st.session_state.last_detection_time = time.time()
                    st.session_state.last_successful_detection = None
                    st.session_state.bust_round = False
                    st.session_state.potential_bust = False
                    st.session_state.potential_win = False
                    st.session_state.round_number = 1
                    
                    # Reset premenných pre jednotlivé hody
                    for player in [1, 2]:
                        for throw in range(1, MAX_SHOTS_PER_ROUND + 1):
                            st.session_state[f"player{player}_throw_{throw}_position"] = None
                            st.session_state[f"player{player}_throw_{throw}_score"] = 0
                    
                    st.rerun()

            with exit_col:
                if st.button("❌ Ukončiť aplikáciu", key="exit_app", use_container_width=True):
                    st.session_state.game_started = False
                    cleanup_cameras()
                    st.success("Aplikácia bola ukončená. Kamery boli uvoľnené.")
                    st.stop()
        with col2:
            st.image(original_target_image, caption="Terč na šípky", width=500)
            
    # Herný režim
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 10px;">
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; font-weight: bold;">
                    🎮 Kolo: {st.session_state.round_number}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Zobrazenie aktuálneho terča so šípkami
            st.image(st.session_state.current_image, caption="Aktuálny stav terča", width=700)
            
            # Tlačidlo na návrat do menu
            if not st.session_state.game_over:
                if st.button("🔙 Späť do menu", key="back_to_menu", use_container_width=True):
                    st.session_state.game_started = False
                    st.session_state.detect_enabled = False
                    st.rerun()
        
        # Informačný panel
        with col2:
            # Aktuálny hráč
            st.markdown(f"### 👤 Na rade: Hráč {st.session_state.current_player}")
            
            score_col1, score_col2 = st.columns(2)
            
            with score_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 40px; color: #666;">Body: Hráč 1</div>
                    <div style="font-size: 120px; font-weight: bold; color: #1e88e5;">
                        {st.session_state.player1_score}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with score_col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px;">
                    <div style="font-size: 40px; color: #666;">Body: Hráč 2</div>
                    <div style="font-size: 120px; font-weight: bold; color: #1e88e5;">
                        {st.session_state.player2_score}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Zobrazenie bodov za aktuálne kolo s možnosťou úpravy
            st.markdown("### 🎯 Body za hody")
            
            # Vizuálne zlepšená tabuľka s bodmi
            current_player = "player1" if st.session_state.current_player == 1 else "player2"
            
            # Vytvorenie tabuľky pre hody
            throw_cols = st.columns(3)
            
            # Nadpisy stĺpcov
            for i, col in enumerate(throw_cols):
                col.markdown(f"**Šípka {i+1}**")
            
            # Hodnoty hodov s možnosťou priameho zadania
            for i, col in enumerate(throw_cols):
                throw_number = i + 1
                throw_key = f"{current_player}_throw_{throw_number}_score"
                
                # Ak už bol hod vykonaný
                if i < len(st.session_state.round_scores):
                    throw_value = st.session_state.round_scores[i]
                    
                    # Použijeme text_input pre priame zadanie
                    new_value_str = col.text_input(
                        "Skore",
                        value=str(int(throw_value)),
                        key=f"edit_throw_{throw_number}"
                    )
                    
                    # Konverzia vstupu a kontrola, či je to číslo
                    try:
                        new_value = int(new_value_str)
                        if new_value != throw_value and 0 <= new_value <= 60:
                            update_throw_score(st.session_state.current_player, throw_number, new_value)
                            st.rerun()
                    except ValueError:
                        col.error("Zadajte platné číslo")
                else:
                    # Ak hod ešte nebol vykonaný
                    col.text("—")
            
            # Tlačidlo na manuálne ukončenie kola
            if not st.session_state.game_over and not st.session_state.ready_for_next_player:
                if not st.session_state.ready_to_review:
                    if st.button("➡️ Ukončiť kolo", key="end_round", use_container_width=True):
                        st.session_state.ready_to_review = True
                        st.rerun()
            
            # Tlačidlo na manuálne prepnutie na ďalšieho hráča
            if st.session_state.ready_to_review and not st.session_state.game_over:
                next_player = 2 if st.session_state.current_player == 1 else 1
                
                # Ak došlo k "bustu", zobrazíme oznámenie
                if st.session_state.bust_round:
                    current_player = "player1" if st.session_state.current_player == 1 else "player2"

                # Tlačidlo pre ďalšieho hráča
                if st.button(f"👤 Hráč {next_player} na rade", key="next_player", use_container_width=True):
                    finalize_round() 
                    switch_to_next_player()
                    st.rerun()

            # Zobrazenie konca hry
            if st.session_state.game_over:
                if st.button("🔄 Nová hra", key="new_game", use_container_width=True):
                    st.session_state.game_started = False
                    st.session_state.game_over = False
                    st.session_state.detect_enabled = False
                    st.rerun()

            # Celkový súčet za kolo
            if len(st.session_state.round_scores) > 0:
                
                current_score, original_score, round_points = calculate_current_score()
                
                # Styling for larger, more prominent score display
                st.markdown(
                    f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <div style="font-size: 90px; font-weight: bold; color: #2ca02c;">
                            {st.session_state.round_scores[-1]}
                        </div>
                        <div style="font-size: 30px; color: #666;">Hodil si</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Display warning or success messages with better styling
                if st.session_state.potential_bust:
                    st.markdown(
                        f"""
                        <div style="background-color: #ffecec; border-left: 5px solid #f44336; padding: 12px; border-radius: 4px; margin: 10px 0;">
                            <span style="font-size: 20px; font-weight: bold;">⚠️ Presah cez 0!</span><br>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                elif st.session_state.potential_win:
                    st.markdown(
                        """
                        <div style="background-color: #ecffec; border-left: 5px solid #4CAF50; padding: 12px; border-radius: 4px; margin: 10px 0;">
                            <span style="font-size: 24px; font-weight: bold;">🎯 Presný zásah, vyhral si!</span><br>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            

            
        # Automatická detekcia šípok
        if (st.session_state.detect_enabled and 
        not st.session_state.ready_to_review and 
        not st.session_state.ready_for_next_player and 
        not st.session_state.game_over and
        not st.session_state.potential_bust and 
        not st.session_state.potential_win):
            print("prisiel som tu")
            detect_dart(cameras, calibration_data)
            st.rerun()

if __name__ == "__main__":
    main()