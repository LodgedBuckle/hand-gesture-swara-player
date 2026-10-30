import os
import time

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# Keep the model and sounds beside this script.
folder = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(folder, "hand_landmarker.task")

sounds_mapping = {
    1: ("Sa", "1_Sa.mp3"),
    2: ("Re", "2_Re.mp3"),
    3: ("Ga", "3_Ga.mp3"),
    4: ("Ma", "4_Ma.mp3"),
    5: ("Pa", "5_Pa.mp3"),
    6: ("Dha", "6_Dha.mp3"),
    7: ("Ni", "7_Ni.mp3"),
    8: ("Sa (upper)", "8_Sa_upper.mp3"),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_landmarks(frame, hand_landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


cap = None
landmarker = None

try:
    pygame.mixer.init()

    # Create the detector once, before the webcam loop.
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open the webcam. Close other camera apps and try again.")

    previous_count = 0
    start_time = time.monotonic()
    timestamp_ms = -1

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Each frame must have a newer timestamp than the previous one.
        timestamp_ms = max(timestamp_ms + 1, int((time.monotonic() - start_time) * 1000))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        count = 0
        for hand in result.hand_landmarks:
            # Four fingers: the tip is above the PIP joint when raised.
            for tip in [8, 12, 16, 20]:
                if hand[tip].y < hand[tip - 2].y:
                    count += 1

            # Thumb: use the index and pinky bases to find the outward direction.
            if hand[5].x < hand[17].x:
                if hand[4].x < hand[3].x:
                    count += 1
            else:
                if hand[4].x > hand[3].x:
                    count += 1

            draw_landmarks(frame, hand)

        # Play only when the count changes, so a held gesture does not stutter.
        if count != previous_count:
            if count in sounds_mapping:
                name, filename = sounds_mapping[count]
                pygame.mixer.music.load(os.path.join(folder, "sounds_swaras", filename))
                pygame.mixer.music.play()
            else:
                pygame.mixer.music.stop()
            previous_count = count

        name = sounds_mapping[count][0] if count in sounds_mapping else "Silence"
        cv2.putText(frame, f"Fingers: {count}  Swara: {name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Swara Player - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    if cap is not None:
        cap.release()
    if landmarker is not None:
        landmarker.close()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
