import cv2
import numpy as np

cameras = []
cams = [3, 1, 2] # toto su indexy kamier ktore si definujeme podla svojho systemu
frame_size = (1000, 1000)

for i in cams:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Camera {i} not found.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    cameras.append(cap)

while True:
    for i in range(len(cameras)):
        ret, frame = cameras[i].read()
        if not ret:
            print(f"Error: Could not read from camera {i}.")
        cv2.imshow(f'Kamera {i}', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break