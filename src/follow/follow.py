# follow.py  (A: 101 takip eden, B: hedef)
import cv2, time, math
from ultralytics import YOLO
from pymavlink import mavutil

# ============ AYARLAR ============
MODEL_PATH = "./model/best.pt"

CONN = "mcast:239.255.145.50:14550"
SYS_A, COMP_A = 101, 1

IMGSZ = 640          # 1280'den düşürdük; yavaşsa/yanlışsa geri al
F = 628.0            # px, kamera frame çözünürlüğüne bağlı (imgsz'den bağımsız)
W_REAL = 0.615
C_OFF = 0.0151
CONF = 0.25
CAM_PITCH = 19.47    # deg, 'p' ile yeniden kalibre edilir

D_DES = 5.0
D_STEP = 0.5
D_DEAD = 0.2         # m
ANG_DEAD = 1.0       # deg

KP_YAW = 2.5         # deg/s per deg
KP_V = 0.8           # m/s per m
KP_Z = 0.08          # m/s per deg
YAW_MAX = 60.0       # deg/s
VX_MAX = 2.5         # m/s
VZ_MAX = 1.0         # m/s
LOST_S = 0.7         # hedef bu süre görünmezse dur
EMA = 0.5            # mesafe filtresi
TURN_FIRST = 30.0    # deg, bu açıdan sonra ileri hız 0

# ============ MAVLINK ============
master = mavutil.mavlink_connection(CONN)
MASK = 0b0000_0111_1100_0111   # vel + yaw_rate aktif


def send_vel(vx, vy, vz, yaw_rate_deg):
    master.mav.set_position_target_local_ned_send(
        0, SYS_A, COMP_A, mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, MASK,
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, math.radians(yaw_rate_deg))


def clip(x, lim):
    return max(-lim, min(lim, x))


def dead(x, d):
    return 0.0 if abs(x) < d else x - math.copysign(d, x)


# ============ KAMERA + MODEL ============
model = YOLO(MODEL_PATH)
pipeline = (
    "udpsrc port=5600 caps=\"application/x-rtp, media=(string)video, "
    "clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

follow = False
d_f = None
last_seen = 0.0
elev_img = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    H, Wd = frame.shape[:2]

    t0 = time.time()
    res = model(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    ms = (time.time() - t0) * 1000
    out = res.plot()
    now = time.time()

    vx = vz = yaw_rate = 0.0
    info = "hedef yok"

    if len(res.boxes):
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        ang = math.atan((x2 - Wd / 2) / F) - math.atan((x1 - Wd / 2) / F)
        x_ang = 2 * math.tan(ang / 2)
        est = W_REAL / max(x_ang - C_OFF, 1e-3)
        d_f = est if d_f is None or now - last_seen > 1.0 else (1 - EMA) * d_f + EMA * est
        last_seen = now

        brg = math.degrees(math.atan2(cx - Wd / 2, F))        # + sağ
        elev_img = math.degrees(math.atan2(cy - H / 2, F))    # + aşağı
        elev = elev_img + CAM_PITCH                           # ufka göre

        yaw_rate = clip(KP_YAW * dead(brg, ANG_DEAD), YAW_MAX)
        vz = clip(KP_Z * dead(elev, ANG_DEAD), VZ_MAX)
        scale = max(0.0, 1 - abs(brg) / TURN_FIRST)
        vx = clip(KP_V * dead(d_f - D_DES, D_DEAD), VX_MAX) * scale
        info = f"d={d_f:.1f}/{D_DES:.1f} m brg={brg:+.1f} elev={elev:+.1f} (img {elev_img:+.1f})"

    target_ok = (now - last_seen) < LOST_S
    if follow and target_ok:
        send_vel(vx, 0.0, vz, yaw_rate)
    else:
        send_vel(0, 0, 0, 0)

    cv2.putText(out, f"{'FOLLOW' if follow else 'IDLE'} | {info}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(out, f"cmd vx={vx:+.2f} vz={vz:+.2f} yr={yaw_rate:+.1f} | {ms:.0f} ms", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("follow", out)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('f'):
        follow = not follow
    elif k == ord('w'):
        D_DES += D_STEP
    elif k == ord('s'):
        D_DES = max(1.0, D_DES - D_STEP)
    elif k == ord('p'):     # A ve B aynı irtifadayken bas
        CAM_PITCH = -elev_img
        print(f"CAM_PITCH = {CAM_PITCH:.2f}")
    elif k == 27:
        break

send_vel(0, 0, 0, 0)
cap.release()
cv2.destroyAllWindows()