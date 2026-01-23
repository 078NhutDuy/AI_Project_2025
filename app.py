from flask import Flask, render_template, Response
import cv2
import numpy as np
import pickle
import pandas as pd
import os
import threading
from datetime import datetime
from keras_facenet import FaceNet

app = Flask(__name__)

if not hasattr(np, "object"):
    np.object = object
    np.bool = bool
    np.int = int

INPUT_IMAGE_SIZE = 160
THRESHOLD = 0.65
ENCODINGS_PATH = "encodings_facenet.pickle"
EXCEL_FILE_PREFIX = "diemdanh"

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)
embedder = FaceNet()
excel_lock = threading.Lock()

try:
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    data = {"names": [], "encodings": []}

checked_in_session = set()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def save_excel_thread(name):
    with excel_lock:
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        filename = f"{EXCEL_FILE_PREFIX}_{date_str}.xlsx"

        try:
            if not os.path.exists(filename):
                df = pd.DataFrame(columns=["Name", "Time", "Date"])
            else:
                df = pd.read_excel(filename)

            if name not in df["Name"].values:
                new_row = pd.DataFrame([{"Name": name, "Time": time_str, "Date": date_str}])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_excel(filename, index=False)
        except Exception:
            pass


def generate_frames():
    frame_count = 0
    skip_frames = 3
    last_detected_faces = []

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        if frame_count % skip_frames == 0:
            last_detected_faces = []
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray_frame, 1.1, 5, minSize=(20, 20))

            for (x, y, w, h) in faces:
                x, y, w, h = x * 4, y * 4, w * 4, h * 4

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_crop = rgb_frame[y:y + h, x:x + w]

                if face_crop.size > 0:
                    try:
                        face_resized = cv2.resize(face_crop, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
                        face_embeddings = embedder.embeddings(np.expand_dims(face_resized, axis=0))
                        current_encoding = face_embeddings[0]

                        distances = np.linalg.norm(data["encodings"] - current_encoding, axis=1)
                        min_dist = np.min(distances)

                        if min_dist < THRESHOLD:
                            idx = np.argmin(distances)
                            name = data["names"][idx]
                            color = (0, 255, 0)
                            accuracy = round(max(0, 100 - (min_dist * 70)))

                            if name not in checked_in_session:
                                checked_in_session.add(name)
                                threading.Thread(target=save_excel_thread, args=(name,)).start()
                        else:
                            name = "Unknown"
                            color = (0, 0, 255)
                            accuracy = 0

                        last_detected_faces.append((x, y, w, h, name, color, accuracy))
                    except Exception:
                        pass

        for (x, y, w, h, name, color, accuracy) in last_detected_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if name != "Unknown":
                cv2.putText(frame, f"{name} ({accuracy}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)