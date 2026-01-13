Hệ thống nhận diện khuôn mặt thời gian thực sử dụng mô hình **FaceNet** để trích xuất đặc trưng (embeddings) và **Haar Cascade** (OpenCV) để phát hiện khuôn mặt. Dự án bao gồm hai giai đoạn chính: Trích xuất dữ liệu mẫu (Encoding) và Nhận diện thực tế qua Camera.

🛠 Yêu cầu hệ thống (Dependencies)

Đảm bảo đã cài đặt thư viện sau:
pip install numpy opencv-python keras-facenet tensorflow mtcnn

📂 Cấu trúc thư mục dự án

├── dataset/            
│   ├── Anh_A/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── Anh_B/
├── train_encoder.py     # File dùng để trích xuất đặc trưng (Encoding)
├── camera_recognition.py # File chạy nhận diện thời gian thực
└── encodings_facenet.pickle # File lưu trữ dữ liệu sau khi train (tự động tạo)


### Bước 1: Chuẩn bị dữ liệu

1. Tạo thư mục `dataset/`.
2. Bên trong `dataset/`, tạo các thư mục con đặt tên theo tên người bạn muốn nhận diện.
3. Bỏ ít nhất 5-10 ảnh khuôn mặt của người đó vào thư mục tương ứng.

### Bước 2: Trích xuất đặc trưng (Encoding)

Chạy file encoding (sử dụng MTCNN để đạt độ chính xác cao khi lấy mẫu):
python train_encoder.py

* Hệ thống sẽ quét qua thư mục `dataset/`, tìm mặt, chuyển thành vector và lưu vào file `encodings_facenet.pickle`.

### Bước 3: Chạy nhận diện Real-time

Bật camera và bắt đầu nhận diện:
python camera.py


## 📊 Thông số kỹ thuật & Đơn vị đo

* **Model Detection:** Haar Cascade (`haarcascade_frontalface_default.xml`).
* **Model Recognition:** FaceNet (InceptionResNetV1).
* **Đơn vị so sánh:** Khoảng cách Euclidean ( distance).
* **Ngưỡng nhận diện (Threshold):**  (Dưới 0.7 được coi là người quen).
* **Đơn vị hiệu suất:** Frames Per Second (FPS).
