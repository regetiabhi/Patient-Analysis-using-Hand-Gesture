import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Configuration
output_file = "data/gestures.csv"
data = []

# Key mapping: Key -> Label
key_map = {
    ord('1'): "Emergency_Help",
    ord('2'): "In_Pain",
    ord('3'): "Need_Water",
    ord('4'): "Hungry_Food",
    ord('5'): "Washroom",
    ord('6'): "Yes_OK",
    ord('7'): "No_Stop",
    ord('8'): "Please_Wait"
}

cap = cv2.VideoCapture(0)
print("--- CONTROLS ---")
for k, label in key_map.items():
    print(f"Press '{chr(k)}' to record: {label}")
print("Press 'q' to save and exit.")

current_label = None
collecting = False
sample_count = 0
MAX_SAMPLES = 200 # Set how many samples per key press

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    status_text = f"Mode: {'RECORDING ' + current_label if collecting else 'IDLE'}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if collecting and sample_count < MAX_SAMPLES:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                landmarks.append(current_label)
                data.append(landmarks)
                sample_count += 1
                cv2.putText(frame, f"Count: {sample_count}/{MAX_SAMPLES}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif sample_count >= MAX_SAMPLES:
                collecting = False
                print(f"Finished {current_label}")
                sample_count = 0

    cv2.imshow("Multi-Gesture Collector", frame)
    key = cv2.waitKey(1)
    
    if key in key_map:
        current_label = key_map[key]
        collecting = True
        sample_count = 0
        print(f"Recording {current_label}...")
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save all data at once
if data:
    df = pd.DataFrame(data)
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f"Successfully saved {len(data)} total samples to {output_file}")