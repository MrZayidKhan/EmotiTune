import cv2
from deepface import DeepFace
import time
import requests
import json
import numpy as np
import threading
from flask import Flask, request, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask_cors import CORS

# Google Sheets Configuration
GAS_WEB_APP_URL = "google_app_script_url"

# Spotify Configuration
SPOTIFY_CLIENT_ID = ''
SPOTIFY_CLIENT_SECRET = ''
SPOTIFY_REDIRECT_URI = ''
SPOTIFY_SCOPE = 'user-modify-playback-state user-read-playback-state'

# Emotion-to-Playlist mapping
EMOTION_PLAYLISTS = {
    'happy': 'spotify:playlist:1eb0FaWzIoUeZA7jopUn9?si=67e6227662184ccd', 
    'sad': 'spotify:playlist:6yES0bJLnNTT0q6HfTuzu4?si=edc93e74cddc4517',   
    'angry': 'spotify:playlist:7LMbUSSaSMLl9RItWjoxEl?si=d800eada174e4fed',  
    'neutral': 'spotify:playlist:1ifKcmoSiW4ONbY5GXHEyt?si=b492af66ad8e4f07',
    'fear': None,
    'surprise': None
}

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=SPOTIFY_SCOPE
))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Global variables
current_playback_emotion = None
playback_active = False
esp32_ip = None
last_gesture_time = 0
GESTURE_TIMEOUT = 5  # seconds

@app.route('/play_playlist', methods=['POST'])
def play_playlist():
    global current_playback_emotion, playback_active
    data = request.json
    emotion = data.get('emotion', 'neutral').lower()
    
    # Don't switch if already playing this emotion
    if emotion == current_playback_emotion and playback_active:
        return jsonify({'status': 'no_change', 'message': 'Already playing this emotion'})
    
    playlist_uri = EMOTION_PLAYLISTS.get(emotion, EMOTION_PLAYLISTS['neutral'])
    
    if not playlist_uri:
        return jsonify({
            'status': 'error',
            'message': f'No playlist available for {emotion} emotion'
        }), 400
    
    try:
        # Start playback of the playlist
        sp.start_playback(context_uri=playlist_uri)
        current_playback_emotion = emotion
        playback_active = True
        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'playlist': playlist_uri
        })
    except Exception as e:
        playback_active = False
        return jsonify({
            'status': 'error',
            'message': str(e),
            'solution': 'Make sure Spotify is open on an active device'
        }), 500

@app.route('/esp32/gesture', methods=['POST'])
def handle_gesture():
    global last_gesture_time
    data = request.json
    gesture = data.get('gesture')
    last_gesture_time = time.time()
    
    try:
        if gesture == 'swipe_left':
            sp.next_track()
            return jsonify({'status': 'success', 'action': 'next_track'})
        elif gesture == 'swipe_right':
            sp.previous_track()
            return jsonify({'status': 'success', 'action': 'previous_track'})
        elif gesture == 'wave':
            current = sp.current_playback()
            if current and current['is_playing']:
                sp.pause_playback()
                return jsonify({'status': 'success', 'action': 'pause'})
            else:
                sp.start_playback()
                return jsonify({'status': 'success', 'action': 'play'})
        else:
            return jsonify({'status': 'error', 'message': 'Unknown gesture'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/esp32/connect', methods=['POST'])
def esp32_connect():
    global esp32_ip
    data = request.json
    esp32_ip = data.get('ip')
    return jsonify({'status': 'success', 'message': f'ESP32 connected from {esp32_ip}'})

def check_playback_status():
    global playback_active, current_playback_emotion
    while True:
        try:
            current = sp.current_playback()
            playback_active = current and current['is_playing']
            time.sleep(5)
        except:
            playback_active = False
            time.sleep(5)

def run_flask():
    app.run(port=5001, host='0.0.0.0')

# Start Flask server and playback monitor in separate threads
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

playback_monitor = threading.Thread(target=check_playback_status, daemon=True)
playback_monitor.start()

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
last_emotion_change = 0
EMOTION_CHANGE_COOLDOWN = 30  # seconds

def background_analyze(frame_snapshot):
    global last_analysis, last_log_time, last_emotion_change, current_playback_emotion

    analysis = get_emotion_analysis(frame_snapshot)
    if analysis["face_detected"]:
        with analysis_lock:
            last_analysis = analysis

        current_time = time.time()
        emotion = analysis['dominant_emotion'].lower()
        
        # Log to Google Sheets if conditions met
        if (current_time - last_log_time > log_interval and
            analysis['dominant_confidence'] >= confidence_threshold and 
            emotion != 'neutral'):
            
            send_to_google_sheets(analysis)
            last_log_time = current_time
        
        # Change playlist if emotion changed and cooldown passed
        if (emotion in EMOTION_PLAYLISTS and 
            emotion != current_playback_emotion and
            current_time - last_emotion_change > EMOTION_CHANGE_COOLDOWN):
            
            try:
                requests.post('http://localhost:5001/play_playlist', 
                            json={'emotion': emotion},
                            timeout=2)
                last_emotion_change = current_time
            except:
                pass

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

def get_emotion_analysis(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
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

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        if current_time >= next_analysis_time:
            snapshot = frame.copy()
            threading.Thread(target=background_analyze, args=(snapshot,), daemon=True).start()
            next_analysis_time = current_time + 3  # Analyze every 3s

        with analysis_lock:
            emotion_text = f"{last_analysis['dominant_emotion']} ({last_analysis['dominant_confidence']:.0%})"

        cv2.putText(frame, emotion_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show ESP32 connection status
        esp_status = f"ESP32: {esp32_ip if esp32_ip else 'Disconnected'}"
        cv2.putText(frame, esp_status, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Mood Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()