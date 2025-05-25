import cv2
from deepface import DeepFace
import time
import requests
import json
import numpy as np
import threading

# Your script URL
GAS_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbzEs_amnDJTAVbbx5ntNNRx_QXv_DAO2bDU6QyOZpGV6ryDTnYL8lmu0Pl5kgbLp-T7/exec"

# Webcam settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State variables
last_log_time = 0
log_interval = 10
confidence_threshold = 0.60
last_analysis = {"dominant_emotion": "Detecting...", "dominant_confidence": 0}
analysis_lock = threading.Lock()
next_analysis_time = 0

# Background analyzer
def background_analyze(frame_snapshot):
    global last_analysis, last_log_time

    analysis = get_emotion_analysis(frame_snapshot)
    if analysis["face_detected"]:
        with analysis_lock:
            last_analysis = analysis

        if (time.time() - last_log_time > log_interval and
            analysis['dominant_confidence'] >= confidence_threshold and 
            analysis['dominant_emotion'].lower() != 'neutral'):

            send_to_google_sheets(analysis)
            last_log_time = time.time()

# Google Sheets logger
def send_to_google_sheets(emotion_data):
    try:
        payload = {"dominant_emotion": str(emotion_data["dominant_emotion"])}
        headers = {"Content-Type": "application/json"}
        response = requests.post(GAS_WEB_APP_URL, json=payload, headers=headers, timeout=5)
        print("Sent:", payload, "| Status:", response.status_code)
        return response.status_code == 200
    except Exception as e:
        print("Send error:", str(e))
        return False

# Emotion detector
def get_emotion_analysis(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',  # Changed to lightweight backend
            silent=True
        )
        emotions = result[0]['emotion']
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        return {
            'dominant_emotion': dominant_emotion[0],
            'dominant_confidence': dominant_emotion[1] / 100,
            'face_detected': True
        }
    except Exception as e:
        print("Detection failed:", str(e))
        return {
            'dominant_emotion': 'unknown',
            'dominant_confidence': 0,
            'face_detected': False
        }

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Run analysis every 3–5 seconds max
        if current_time >= next_analysis_time:
            snapshot = frame.copy()
            threading.Thread(target=background_analyze, args=(snapshot,), daemon=True).start()
            next_analysis_time = current_time + 3  # Analyze every 3s

        # Draw the emotion overlay
        with analysis_lock:
            emotion_text = f"{last_analysis['dominant_emotion']} ({last_analysis['dominant_confidence']:.0%})"

        cv2.putText(frame, emotion_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Mood Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
