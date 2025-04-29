import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

finger_tips = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, hand_label):
    fingers = []

    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]

    # Use angle and position relative to palm
    if hand_label == "Right":
        if thumb_tip.x > index_mcp.x and abs(thumb_tip.y - wrist.y) < 0.2:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left
        if thumb_tip.x < index_mcp.x and abs(thumb_tip.y - wrist.y) < 0.2:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other 4 fingers
    for tip in finger_tips[1:]:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        if tip_y < pip_y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = hand_handedness.classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            count = count_fingers(hand_landmarks, label)

            cv2.putText(frame, f'{label} hand: {count} fingers', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
