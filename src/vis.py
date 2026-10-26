# vision_test.py
import time
import cv2
from ultralytics import YOLO

PIPELINE = (
    "udpsrc port=5600 caps=\"application/x-rtp, media=(string)video, "
    "clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise SystemExit("Pipeline açılamadı (OpenCV GStreamer desteğini kontrol et)")

model = YOLO("yolov8n.pt")   # ilk çalışmada otomatik iner
# GPU varsa otomatik kullanılır; zorlamak için: model.to("cuda")

t_prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    t0 = time.time()
    results = model(frame, verbose=False)[0]      # çıkarım
    t_inf = (time.time() - t0) * 1000

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / (now - t_prev))
    t_prev = now
    cv2.putText(frame, f"FPS {fps:.1f} | infer {t_inf:.1f} ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("drone", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()