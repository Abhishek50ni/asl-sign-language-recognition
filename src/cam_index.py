import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"DroidCam found at index {i}")
        cap.release()