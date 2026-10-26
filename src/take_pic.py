import cv2, time, os
os.makedirs("dataset/raw", exist_ok=True)
# pipeline = ("udpsrc port=5600 ! application/x-rtp,encoding-name=H264 ! "
#             "rtph264depay ! avdec_h264 ! videoconvert ! appsink")

pipeline = (
    "udpsrc port=5600 caps=\"application/x-rtp, media=(string)video, "
    "clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
last, n = 0, 0
while True:
    ok, frame = cap.read()
    if not ok: continue
    if time.time() - last > 0.5:      # saniyede 2 kare
        cv2.imwrite(f"dataset/raw/img_{n:05d}.jpg", frame)
        n += 1; last = time.time()
    cv2.imshow("rec", frame)
    if cv2.waitKey(1) == 27: break    # ESC