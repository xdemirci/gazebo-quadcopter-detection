import cv2, time
from ultralytics import YOLO

model = YOLO("/home/demirci/Downloads/modelim/quad_v1/runs/detect/train/weights/best.pt")

# pipeline = ("udpsrc port=5600 ! application/x-rtp,encoding-name=H264 ! "
#             "rtph264depay ! avdec_h264 ! videoconvert ! appsink")

pipeline = (
    "udpsrc port=5600 caps=\"application/x-rtp, media=(string)video, "
    "clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
)


cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

import os; os.makedirs("hard_cases", exist_ok=True)
n = 0
while True:
    ok, frame = cap.read()
    if not ok: continue
    t = time.time()
    res = model(frame, imgsz=1280, conf=0.25, verbose=False)[0]
    out = res.plot()
    cv2.putText(out, f"{len(res.boxes)} quad | {(time.time()-t)*1000:.0f} ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("quad", out)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):                       # modelin yanıldığı kareyi kaydet
        cv2.imwrite(f"hard_cases/hard_{n:04d}.jpg", frame); n += 1
    if k == 27: break                       # ESC