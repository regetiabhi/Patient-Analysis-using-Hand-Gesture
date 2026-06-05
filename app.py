import cv2
import mediapipe as mp
import joblib
import numpy as np
import requests
import time
from flask import Flask, render_template, Response

app = Flask(__name__)

# --- CONFIGURATION ---
# Replace with your actual Discord Webhook URL
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE"
MODEL_PATH = 'models/gesture_model.pkl'
LAST_ALERT_TIME = 0
ALERT_COOLDOWN = 15  # Seconds between alerts to avoid spam

# Load the winning Random Forest model
model = joblib.load(MODEL_PATH)

# Initialize MediaPipe Solutions [cite: 36, 59, 205]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

def send_discord_notification(gesture):
    global LAST_ALERT_TIME
    if time.time() - LAST_ALERT_TIME > ALERT_COOLDOWN:
        payload = {
            "embeds": [{
                "title": "🚨 PATIENT ASSISTANCE REQUIRED",
                "description": f"The system has detected a critical gesture: **{gesture}**",
                "color": 15548997, # Vivid Red
                "footer": {"text": "Patient Assist AI - Sathyabama IST"}
            }]
        }
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=payload)
            LAST_ALERT_TIME = time.time()
        except Exception as e:
            print(f"Webhook Error: {e}")

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        prediction_text = "Waiting for Gesture..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the 21 key points [cite: 90-126]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract coordinates for Module-3 Prediction [cite: 63, 152]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                
                # Model Inference
                prediction = model.predict([landmarks])
                prediction_text = prediction[0]

                # Trigger Discord for Critical Needs 
                if prediction_text in ["Emergency_Help", "In_Pain"]:
                    send_discord_notification(prediction_text)

        # UI Overlay
        cv2.putText(frame, f"STATUS: {prediction_text}", (20, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (44, 62, 80), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home(): return render_template('index.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/detector')
def detector(): return render_template('detector.html')

@app.route('/details')
def details(): return render_template('details.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)