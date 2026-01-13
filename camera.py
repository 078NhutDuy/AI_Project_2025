import numpy as np
import cv2
import pickle
import os
from keras_facenet import FaceNet

# --- Sửa lỗi tương thích NumPy ---
if not hasattr(np, "object"):
    np.object = object
    np.bool = bool
    np.int = int

# --- Cấu hình ---
INPUT_IMAGE_SIZE = 160

# 1. Khởi tạo Haar Cascade (Sử dụng đường dẫn mặc định của OpenCV)
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("[ERROR] Không tìm thấy file xml của Haar Cascade!")
    exit()

# 2. Khởi tạo FaceNet
embedder = FaceNet()

# 3. Load dữ liệu đặc trưng
try:
    with open("encodings_facenet.pickle", "rb") as f:
        data = pickle.load(f)
    print("[INFO] Đã load dữ liệu thành công.")
except FileNotFoundError:
    print("[ERROR] Không tìm thấy file encodings_facenet.pickle!")
    exit()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    # Haar Cascade hoạt động tốt nhất trên ảnh xám
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Phát hiện mặt bằng Haar Cascade
    faces = face_detector.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Tọa độ vùng mặt
        x2, y2 = x + w, y + h

        # --- QUY TRÌNH NHẬN DIỆN ---
        # Bước A: Cắt khuôn mặt (Sử dụng RGB cho FaceNet)
        face_crop = rgb_frame[y:y2, x:x2]
        
        if face_crop.size > 0:
            # Bước B: Resize và nhúng (Embeddings)
            face_resized = cv2.resize(face_crop, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
            # Tạo embedding
            face_embeddings = embedder.embeddings(np.expand_dims(face_resized, axis=0))
            current_encoding = face_embeddings[0]

            # Bước C: So sánh khoảng cách Euclidean
            distances = np.linalg.norm(data["encodings"] - current_encoding, axis=1)
            min_dist = np.min(distances)
            
            name = "Unknown"
            color = (0, 0, 255) # Đỏ
            
            # Ngưỡng (Threshold): 0.7 là mức phổ biến cho FaceNet
            if min_dist < 0.7:
                idx = np.argmin(distances)
                name = data["names"][idx]
                color = (0, 255, 0) # Xanh lá

            accuracy = max(0, 100 - (min_dist * 70))

            # 5. Vẽ kết quả
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({accuracy:.1f}%)", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition (Haar Cascade)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()