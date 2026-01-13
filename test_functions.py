try:
    import mediapipe as mp
    print("Mediapipe version:", mp.__version__)
    print("Solutions found:", "face_detection" in dir(mp.solutions))
    print("--- CÀI ĐẶT THÀNH CÔNG ---")
except Exception as e:
    print("--- LỖI ---")
    print(e)