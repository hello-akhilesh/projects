import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Set screen size (you can use pyautogui.size())
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video
cap = cv2.VideoCapture(0)

# Previous location for smoothing
prev_x, prev_y = 0, 0
smoothening = 7

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip image for natural movement
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of index finger tip (landmark 8)
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Map coordinates to screen
            screen_x = np.interp(x, (0, w), (0, screen_width))
            screen_y = np.interp(y, (0, h), (0, screen_height))

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)  # Mirror movement if needed

            prev_x, prev_y = curr_x, curr_y

            # Optional: Detect pinch (thumb tip and index tip distance)
            x_thumb = int(hand_landmarks.landmark[4].x * w)
            y_thumb = int(hand_landmarks.landmark[4].y * h)
            distance = np.hypot(x - x_thumb, y - y_thumb)

            if distance < 40:
                pyautogui.click()
                time.sleep(0.2)  # Debounce delay

            # Draw fingertip circle
            cv2.circle(image, (x, y), 10, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Virtual Mouse", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
