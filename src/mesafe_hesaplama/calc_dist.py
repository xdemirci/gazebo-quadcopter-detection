import cv2, time, math, threading, csv, os, statistics
from ultralytics import YOLO
from pymavlink import mavutil

# ================== AYARLAR ==================
MODEL_PATH = "./model/best.pt"

# MAVLink bağlantıları. Her iki araç da tek porta geliyorsa tek eleman yeter.
# Araçlar ayrı portlardaysa her birini listeye ekle. Sysid mesajın içinden okunur.
# Başka bir GCS ile çakışmaması için MAVProxy'ye ek bir --out ver:
#   --out 127.0.0.1:14560
# CONNS = ["udpin:0.0.0.0:14560"]
CONNS = ["mcast:239.255.145.50:14550"]


SYS_A, SYS_B = 101, 102          # A: takip eden (kamera), B: hedef
HFOV = 1.151917                  # model.sdf horizontal_fov (rad)
W_REAL = 0.5                     # B'nin gerçek açıklığı (m). 'c' tuşuyla kalibre edilir
CONF = 0.25
STALE_S = 1.0                    # bu süreden eski GPS verisini kullanma

# ================== PAYLAŞILAN VERİ ==================
state = {SYS_A: {}, SYS_B: {}}   # {101: {lat, lon, alt, yaw, t}, 102: {...}}
shared = {"gps": None}           # gps thread'inin hesapladığı gerçek mesafe
lock = threading.Lock()
stop = threading.Event()


def mav_worker():
    """Tüm bağlantılardan blocking=False ile okur, sysid'ye göre sözlüğe yazar."""
    conns = [mavutil.mavlink_connection(c) for c in CONNS]
    while not stop.is_set():
        got_any = False
        for master in conns:
            msg = master.recv_match(type=["GLOBAL_POSITION_INT", "ATTITUDE"],
                                    blocking=False)
            if msg is None:
                continue
            got_any = True
            sid = msg.get_srcSystem()
            if sid not in state:
                continue
            with lock:
                s = state[sid]
                if msg.get_type() == "GLOBAL_POSITION_INT":
                    s["lat"] = msg.lat / 1e7
                    s["lon"] = msg.lon / 1e7
                    s["alt"] = msg.alt / 1000.0      # MSL, metre
                else:
                    s["yaw"] = math.degrees(msg.yaw)  # -180..180, kuzey=0
                s["t"] = time.time()
        if not got_any:
            time.sleep(0.002)


def dist_worker():
    """İki aracın GPS konumundan metre cinsinden gerçek mesafeyi hesaplar."""
    while not stop.is_set():
        with lock:
            a, b = dict(state[SYS_A]), dict(state[SYS_B])
        need = ("lat", "lon", "alt", "t")
        if all(k in a for k in need) and all(k in b for k in need) \
                and time.time() - min(a["t"], b["t"]) < STALE_S:
            dN = (b["lat"] - a["lat"]) * 111319.5
            dE = (b["lon"] - a["lon"]) * 111319.5 * math.cos(math.radians(a["lat"]))
            dU = b["alt"] - a["alt"]
            horiz = math.hypot(dN, dE)
            d3 = math.sqrt(horiz ** 2 + dU ** 2)
            brg = math.degrees(math.atan2(dE, dN))           # A'dan B'ye, kuzey=0
            rel = (brg - a.get("yaw", 0.0) + 180) % 360 - 180  # + = B sağda
            with lock:
                shared["gps"] = {"horiz": horiz, "d3": d3, "rel": rel,
                                 "t": time.time()}
        else:
            with lock:
                shared["gps"] = None
        time.sleep(0.1)


threading.Thread(target=mav_worker, daemon=True).start()
threading.Thread(target=dist_worker, daemon=True).start()

# ================== KAMERA + MODEL ==================
model = YOLO(MODEL_PATH)
pipeline = (
    "udpsrc port=5600 caps=\"application/x-rtp, media=(string)video, "
    "clock-rate=(int)90000, encoding-name=(string)H264\" ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

os.makedirs("hard_cases", exist_ok=True)
log_f = open("mesafe_log.csv", "w", newline="")
log = csv.writer(log_f)
log.writerow(["t", "size_px", "est_m", "gps_3d_m", "gps_horiz_m",
              "implied_W_m", "img_bearing_deg", "gps_bearing_deg"])

F = None
n_saved = 0
implied_ws = []

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    H, Wd = frame.shape[:2]
    if F is None:
        F = (Wd / 2) / math.tan(HFOV / 2)      # odak uzaklığı (piksel)

    t0 = time.time()
    res = model(frame, imgsz=1280, conf=CONF, verbose=False)[0]
    out = res.plot()
    ms = (time.time() - t0) * 1000

    with lock:
        gps = shared["gps"]

    lines = [f"{len(res.boxes)} quad | {ms:.0f} ms | W={W_REAL:.2f} m"]

    if len(res.boxes):
        i = int(res.boxes.conf.argmax())            # en güvenilir kutu
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        size = max(x2 - x1, y2 - y1)                # piksel
        cx = (x1 + x2) / 2
        est = F * W_REAL / size                     # model mesafesi
        img_brg = math.degrees(math.atan2(cx - Wd / 2, F))   # + = sağda

        lines.append(f"model: {est:.1f} m | yon(img): {img_brg:+.1f} deg")
        if gps:
            err = est - gps["d3"]
            implied_w = gps["d3"] * size / F        # bu mesafe için W olmalıydı
            implied_ws.append(implied_w)
            lines.append(f"GPS 3D: {gps['d3']:.1f} m (yatay {gps['horiz']:.1f})"
                         f" | hata: {err:+.1f} m ({100 * err / gps['d3']:+.0f}%)")
            lines.append(f"yon(GPS): {gps['rel']:+.1f} deg | W_gerek: {implied_w:.2f} m")
            log.writerow([f"{time.time():.2f}", f"{size:.1f}", f"{est:.2f}",
                          f"{gps['d3']:.2f}", f"{gps['horiz']:.2f}",
                          f"{implied_w:.3f}", f"{img_brg:.2f}", f"{gps['rel']:.2f}"])
        else:
            lines.append("GPS verisi yok / eski")
    elif gps:
        lines.append(f"tespit yok | GPS 3D: {gps['d3']:.1f} m")

    for j, txt in enumerate(lines):
        cv2.putText(out, txt, (10, 30 + 32 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)
    cv2.imshow("quad", out)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):                               # modelin yanildigi kare (ham)
        cv2.imwrite(f"hard_cases/hard_{n_saved:04d}.jpg", frame)
        n_saved += 1
    elif k == ord('c') and len(implied_ws) >= 10:   # W'yu GPS'ten kalibre et
        W_REAL = statistics.median(implied_ws)
        print(f"Kalibre edildi: W_REAL = {W_REAL:.3f} m "
              f"({len(implied_ws)} ornek, min {min(implied_ws):.2f}, "
              f"max {max(implied_ws):.2f})")
        implied_ws.clear()
    elif k == 27:
        break

stop.set()
cap.release()
log_f.close()
cv2.destroyAllWindows()