import os
import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Khởi tạo model
detector = MTCNN()
embedder = FaceNet()

dataset_path = "dataset"
knownEncodings = []
knownNames = []

print("[INFO] Đang bắt đầu encode...")

# Duyệt qua các thư mục con trong dataset
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir): continue

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = cv2.imread(image_path)
        if image is None: continue
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Phát hiện mặt bằng MTCNN
        results = detector.detect_faces(rgb)
        
        if len(results) > 0:
            # Lấy khuôn mặt đầu tiên tìm thấy
            x, y, w, h = results[0]['box']
            face = rgb[y:y+h, x:x+w]
            
            # 2. Tạo embedding 512 chiều từ FaceNet
            # extract tự động chuẩn hóa và resize ảnh mặt về 160x160
            detections = embedder.extract(rgb, threshold=0.95)
            if len(detections) > 0:
                encoding = detections[0]['embedding']
                knownEncodings.append(encoding)
                knownNames.append(person_name)
                print(f"[INFO] Đã encode: {person_name} - {image_name}")

# Lưu vào file pickle
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings_facenet.pickle", "wb") as f:
    f.write(pickle.dumps(data))
print("[INFO] Đã lưu xong file encodings_facenet.pickle!")