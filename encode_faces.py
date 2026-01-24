import os
import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from tqdm import tqdm  # Thư viện hiển thị thanh tiến trình

# CẤU HÌNH & KHỞI TẠO
BATCH_SIZE = 128  # Số lượng khuôn mặt xử lý cùng lúc (tăng lên nếu RAM/GPU mạnh)
IMG_SIZE = (160, 160)  # Kích thước chuẩn đầu vào của FaceNet
DATASET_PATH = "dataset/train/train"
OUTPUT_FILE = "encodings_facenet.pickle"

# Khởi tạo model
# min_face_size: Bỏ qua các mặt quá nhỏ để tăng tốc độ
detector = MTCNN()
embedder = FaceNet()

print("[INFO] Đang khởi tạo và quét dữ liệu...")


# =============================================================================
# HÀM HỖ TRỢ
# =============================================================================
def process_batch(faces, names, encodings_list, names_list):
    """
    Hàm nhận vào 1 lô (batch) các khuôn mặt đã cắt,
    tính toán embedding và lưu vào danh sách chính.
    """
    if len(faces) == 0:
        return

    # Chuyển list thành numpy array
    faces_array = np.array(faces)

    # Tính toán embedding cho cả lô (nhanh hơn xử lý lẻ)
    # Dùng hàm embeddings() thay vì extract() để bỏ qua bước detect lại
    embeddings = embedder.embeddings(faces_array)

    # Lưu kết quả
    for i in range(len(embeddings)):
        encodings_list.append(embeddings[i])
        names_list.append(names[i])


# XỬ LÝ CHÍNH
knownEncodings = []
knownNames = []

# Bộ đệm (buffer) để chứa dữ liệu cho batch
batch_faces = []
batch_names = []

# Lấy danh sách ảnh trước để dùng thanh tiến trình (tqdm)
image_paths = []
for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_dir): continue
    for image_name in os.listdir(person_dir):
        image_paths.append((person_name, os.path.join(person_dir, image_name)))

print(f"[INFO] Tìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")

for (person_name, image_path) in tqdm(image_paths, desc="Encoding"):
    image = cv2.imread(image_path)
    if image is None: continue

    # Chuyển màu
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OPTION: Resize ảnh nếu quá lớn để MTCNN chạy nhanh hơn
    # (Chỉ resize để detect, không ảnh hưởng chất lượng crop vì crop xong sẽ resize về 160x160)
    h, w = rgb.shape[:2]
    if w > 800:
        scale = 800 / w
        rgb = cv2.resize(rgb, (800, int(h * scale)))

    # 1. Phát hiện khuôn mặt
    results = detector.detect_faces(rgb)

    if len(results) > 0:
        # Lấy khuôn mặt có độ tin cậy cao nhất (thường là cái đầu tiên hoặc lớn nhất)
        bounding_box = results[0]['box']
        keypoints = results[0]['keypoints']

        x, y, w, h = bounding_box
        # Đảm bảo tọa độ không âm
        x, y = max(0, x), max(0, y)

        # Cắt vùng mặt
        face = rgb[y:y + h, x:x + w]

        # Nếu cắt lỗi hoặc rỗng thì bỏ qua
        if face.size == 0: continue

        # 2. Tiền xử lý để đưa vào FaceNet
        # Resize về đúng 160x160 pixel
        face_resized = cv2.resize(face, IMG_SIZE)

        # Thêm vào bộ đệm batch
        batch_faces.append(face_resized)
        batch_names.append(person_name)

        # 3. Nếu bộ đệm đầy (đủ BATCH_SIZE) thì xử lý 1 lần
        if len(batch_faces) >= BATCH_SIZE:
            process_batch(batch_faces, batch_names, knownEncodings, knownNames)
            # Reset bộ đệm
            batch_faces = []
            batch_names = []

# Xử lý nốt những ảnh còn sót lại trong bộ đệm (nếu chưa đủ batch cuối)
if len(batch_faces) > 0:
    process_batch(batch_faces, batch_names, knownEncodings, knownNames)

# LƯU FILE
print("\n[INFO] Đang lưu file pickle...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(OUTPUT_FILE, "wb") as f:
    f.write(pickle.dumps(data))
print(f"[INFO] Hoàn tất! Đã lưu {len(knownEncodings)} khuôn mặt.")