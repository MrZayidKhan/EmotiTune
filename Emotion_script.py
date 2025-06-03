import cv2
from deepface import DeepFace
import time
import requests
import threading
from flask import Flask, request, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask_cors import CORS
from werkzeug.serving import make_server

GAS_WEB_APP_URL = "" #google appscript url
SPOTIFY_CLIENT_ID = '' # spotify account client id (spotify for devs)
SPOTIFY_CLIENT_SECRET = ''# spotify account client secret (spotify for devs)
SPOTIFY_REDIRECT_URI = 'http://127.0.0.1:5000/callback'

EMOTION_PLAYLISTS = {
    'happy': 'spotify:playlist:5Ao3bBXmAkwGt5D3oxUUKK?si=de1b57d19a1d442a',
    'sad': 'spotify:playlist:2evsoqz6SI3X68tUVrn7LM?si=72f48450657c446b',
    'angry': 'spotify:playlist:7LMbUSSaSMLl9RItWjoxEl?si=5df921d8c95e48b0',
    'neutral': 'spotify:playlist:24bpsB9QScttfWQ2LCM7rX?si=ea8dc44277984713'
}

#Flask Setup 
app = Flask(__name__)
CORS(app)

current_state = {
    'emotion': None,
    'playback_active': False,
    'esp32_ip': None,
    'last_gesture': None,
    'last_sheets_check': 0,
    'sheets_data': [],
    'latest_analysis': {'emotion': 'neutral', 'confidence': 1.0, 'face_detected': False},
    'latest_frame': None
}

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope='user-modify-playback-state user-read-playback-state'
))

class FlaskThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.server = make_server('0.0.0.0', 5001, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

@app.route('/play_playlist', methods=['POST'])
def play_playlist():
    data = request.json
    emotion = data.get('emotion', 'neutral').lower()
    playlist_uri = EMOTION_PLAYLISTS.get(emotion)

    if not playlist_uri:
        return jsonify({'status': 'error', 'message': f'No playlist for {emotion}'}), 400

    try:
        devices = sp.devices().get('devices', [])
        if not devices:
            raise Exception("No Spotify devices found")

        # use first available device, even if not 'active'
        device = next((d for d in devices if d['is_active']), devices[0])

        sp.start_playback(device_id=device['id'], context_uri=playlist_uri)
        current_state['emotion'] = emotion
        current_state['playback_active'] = True
        return jsonify({'status': 'success'})

    except Exception as e:
        current_state['playback_active'] = False
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/esp32/gesture', methods=['POST'])
def handle_gesture():
    data = request.json
    gesture = data.get('gesture')
    current_state['last_gesture'] = gesture
    try:
        devices = sp.devices().get('devices', [])
        if not devices:
            raise Exception("No active Spotify device found")

        active_device = next((d for d in devices if d['is_active']), None)
        if not active_device:
            raise Exception("No active Spotify device found")

        if gesture == 'next':
            sp.next_track(device_id=active_device['id'])
        elif gesture == 'previous':
            sp.previous_track(device_id=active_device['id'])
        elif gesture == 'pause':
            sp.pause_playback(device_id=active_device['id'])
        elif gesture == 'play':
            sp.start_playback(device_id=active_device['id'])
        else:
            return jsonify({'status': 'error', 'message': 'Unknown gesture'}), 400

        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def analyze_emotion_background():
    last_analysis_time = 0
    last_frame_gray = None

    while True:
        if current_state.get('latest_frame') is not None:
            current_time = time.time()
            if current_time - last_analysis_time >= 3:  # analyze every 3 seconds
                try:
                    # Convert to grayscale for quick diff check
                    gray = cv2.cvtColor(current_state['latest_frame'], cv2.COLOR_BGR2GRAY)
                    if last_frame_gray is not None:
                        diff = cv2.absdiff(gray, last_frame_gray)
                        non_zero_count = cv2.countNonZero(diff)
                        if non_zero_count < 5000:  # low difference, skip analysis
                            time.sleep(0.1)
                            continue
                    last_frame_gray = gray

                    small = cv2.resize(current_state['latest_frame'], (320, 240))
                    result = DeepFace.analyze(
                        small,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True,
                        detector_backend='opencv'
                    )[0]

                    emotions = result['emotion']
                    dominant_emotion, confidence = max(emotions.items(), key=lambda x: x[1])
                    confidence /= 100

                    if confidence < 0.4 or dominant_emotion == 'unknown':
                        dominant_emotion, confidence = 'neutral', 1.0

                    current_state['latest_analysis'] = {
                        'emotion': dominant_emotion,
                        'confidence': confidence,
                        'face_detected': True
                    }
                    last_analysis_time = current_time

                except Exception as e:
                    current_state['latest_analysis'] = {
                        'emotion': 'neutral',
                        'confidence': 1.0,
                        'face_detected': False
                    }
            else:
                time.sleep(0.1)
        else:
            time.sleep(0.1)


# main loop
def main():
    flask_thread = FlaskThread()
    flask_thread.daemon = True
    flask_thread.start()

    threading.Thread(target=analyze_emotion_background, daemon=True).start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_log = 0
    last_emotion_change = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            current_state['latest_frame'] = frame.copy()
            current_time = time.time()

            analysis = current_state['latest_analysis']
            if analysis.get('face_detected', False):
                if analysis['confidence'] >= 0.6 and current_time - last_log >= 10:
                    try:
                        requests.post(GAS_WEB_APP_URL, json={"dominant_emotion": analysis['emotion']}, timeout=5)
                        last_log = current_time
                    except: pass

                if analysis['emotion'] in EMOTION_PLAYLISTS and analysis['emotion'] != current_state['emotion'] and current_time - last_emotion_change >= 20:
                    try:
                        requests.post('http://localhost:5001/play_playlist', json={'emotion': analysis['emotion']}, timeout=5)
                        last_emotion_change = current_time
                    except Exception as e:
                        print("Playlist switch error:", e)

            cv2.putText(frame, f"{analysis['emotion']} ({int(analysis['confidence'] * 100)}%)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Mood Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        flask_thread.shutdown()

if __name__ == '__main__':
    main()
