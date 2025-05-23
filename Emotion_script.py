import cv2
from deepface import DeepFace
import time
import requests
import json

# Script URL
GAS_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbx6omXwa2Dt3IhEhzoKfHVgFNtzSl6syuMuKiF0kBXGrx-zrUDFpjj4RzOVTbAXweOm/exec"

# webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Medium resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Timing
last_log_time = 0
log_interval = 10 
confidence_threshold = 0.75  

def send_to_google_sheets(emotion, confidence):
   
    if emotion.lower() == "neutral":
        print(f"Skipping Neutral emotion ({confidence:.0%})")
        return
        
    payload = {
        "mood": emotion,
        "confidence": confidence,  # Send raw confidence value (0.0-1.0)
        "confidence_percentage": f"{confidence:.0%}"  # Also send formatted percentage
    }
    
    try:
        response = requests.post(
            GAS_WEB_APP_URL,
            json=payload,
            timeout=3
        )
        print(f"Logged: {emotion} ({confidence:.0%}) | Status: {response.status_code}")
    except Exception as e:
        print(f"Upload failed: {str(e)}")

def get_dominant_emotion(frame):
    """Get emotion with highest confidence score"""
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True  # Reduce console output
        )
        emotions = result[0]['emotion']
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]/100
    except Exception as e:
        print(f"Detection error: {str(e)}")
        return None, 0

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        current_time = time.time()
        
        # Analyze frame
        emotion, confidence = get_dominant_emotion(frame)
        if emotion:
            # Display emotion on frame
            cv2.putText(frame, f"{emotion} ({confidence:.0%})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Log periodically if above confidence threshold
            if (current_time - last_log_time > log_interval and 
                confidence >= confidence_threshold):
                send_to_google_sheets(emotion, confidence)
                last_log_time = current_time
        
        # Show frame
        cv2.imshow("Emotion Detection", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Ensure resources are released
    cap.release()
    cv2.destroyAllWindows()