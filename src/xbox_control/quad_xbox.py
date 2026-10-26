import argparse
import os
import signal
import time
import tkinter as tk

import pygame
from pymavlink import mavutil

# (custom_mode_id, isim) - Copter
MODES = {
    "LT": (16, "POSHOLD"),
    "RT": (4, "GUIDED"),   # velocity komutları sadece GUIDED'da çalışır
    "Y": (5, "LOITER"),
    "X": (9, "LAND"),
}

TAKEOFF_ALT = 5.0          # m, A tuşu
TAKEOFF_TIME = 8.0         # s, bu sürede velocity gönderme (yoksa takeoff iptal olur)
DT = 0.05                  # döngü periyodu (s)
SMOOTH_ALPHA = 0.10         # yüksek = hızlı tepki, düşük = yumuşak
DEADBAND = 0.08

# Stick -> body-frame hız. sign: stick yönü düzeltmesi, scale: tam stick hızı
# vx: ileri (m/s), vy: sağ (m/s), vz: aşağı pozitif (m/s), yaw: rad/s
STICKS = [
    {"key": "vx", "name": "Vx fwd", "axis": 4, "sign": -1, "scale": 3.0},
    {"key": "vy", "name": "Vy right", "axis": 3, "sign": 1, "scale": 3.0},
    {"key": "vz", "name": "Vz down", "axis": 1, "sign": 1, "scale": 1.5},
    {"key": "yaw", "name": "Yaw rate", "axis": 0, "sign": 1, "scale": 1.0},
]

# Gimbal: RC6=Mount roll, RC7=Mount pitch, RC8=Mount yaw (RCx_OPTION 212/213/214)
# Stick pozisyonu PWM'i integre eder, bırakınca gimbal açısı korunur.
GIMBAL = [
    {"ch": 6, "name": "Roll", "axis": 0, "sign": 1},
    {"ch": 7, "name": "Pitch", "axis": 4, "sign": -1},
    {"ch": 8, "name": "Yaw", "axis": 3, "sign": 1},
]
GIMBAL_RATE = 150          # PWM/s (tam stick)
GIMBAL_MIN, GIMBAL_MAX = 1100, 1900

# Sadece vx,vy,vz + yaw_rate kullan (pos, accel, yaw ignore)
VEL_TYPE_MASK = 0b010111000111


def parse_args():
    ap = argparse.ArgumentParser(
        description="F310/Xbox ile Gazebo SITL yer kontrol istasyonu (velocity + gimbal)",
        epilog="Örnekler:  python3 quad_control.py 101   |   python3 quad_control.py 101 102   |   python3 quad_control.py 0",
    )
    ap.add_argument(
        "ids",
        nargs="*",
        type=int,
        default=[0],
        help="Hedef SYS_ID(ler). 0 veya boş = tüm sürü",
    )
    args = ap.parse_args()
    for i in args.ids:
        if not 0 <= i <= 255:
            ap.error(f"Geçersiz SYS_ID: {i} (0-255 olmalı)")
    return args


class GazeboF310GCS:

    def __init__(self, root, target_ids):
        self.root = root
        self.root.geometry("500x270")
        self.root.configure(padx=15, pady=15)
        self.root.protocol("WM_DELETE_WINDOW", self.request_stop)

        self._stop = False
        self._closed = False

        # Hedef seçimi: 0 varsa tüm sürü, yoksa sadece verilen ID'ler
        self.target_ids = set(target_ids)
        self.all_mode = 0 in self.target_ids
        if self.all_mode:
            self.target_label = "TÜM SÜRÜ"
        else:
            self.target_label = "SYS_ID " + ",".join(str(i) for i in sorted(self.target_ids))
        self.root.title(f"YER KONTROL İSTASYONU - {self.target_label}")

        # Ağda görülen araçlar (sadece araç heartbeat'leri)
        self.seen_ids = set()

        # --- MAVLink Bağlantısı ---
        print("MAVLink bağlantısı dinleniyor...")
        self.master = mavutil.mavlink_connection("mcast:239.255.145.50:14550")
        # self.master = mavutil.mavlink_connection("udp:localhost:14550")
        self.wait_for_targets()

        # --- Pygame & Mixer (Ses) ---
        pygame.init()
        pygame.mixer.init()
        pygame.joystick.init()

        self.kill_sound = None
        sound_filename = "intestaller_killing_main_engine.mp3"
        if os.path.exists(sound_filename):
            try:
                self.kill_sound = pygame.mixer.Sound(sound_filename)
                print("Ses efekti yüklendi.")
            except Exception as e:
                print(f"Ses yükleme hatası: {e}")
        else:
            print(f"UYARI: '{sound_filename}' bulunamadı, ses çalmayacak.")

        self.joystick = None
        self.setup_joystick()

        # --- Durum ---
        self.vel = {s["key"]: 0.0 for s in STICKS}          # filtrelenmiş hız komutları
        self.gimbal = {g["ch"]: 1500 for g in GIMBAL}       # gimbal PWM (açı)
        self.gimbal_mode = False                            # RB: 0 = hız, 1 = gimbal
        self.takeoff_until = 0.0                            # takeoff süresince velocity susar

        self.prev_button_states = {}
        self.bars = {}
        self.val_labels = {}

        self.create_widgets()

        # Ctrl+C: sadece bayrak kaldırır, asıl kapanış döngü içinde yapılır
        signal.signal(signal.SIGINT, lambda *_: self.request_stop())

        self.update_and_send_loop()

    # ==========================================
    # HEDEF YÖNETİMİ
    # ==========================================
    @staticmethod
    def is_vehicle(msg):
        """GCS/MAVProxy gibi araç olmayan heartbeat'leri ele."""
        return (
            msg.type != mavutil.mavlink.MAV_TYPE_GCS
            and msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID
            and msg.get_srcSystem() > 0
        )

    def wait_for_targets(self):
        """Tüm sürü modunda ilk araç, ID modunda istenen tüm ID'ler görünene kadar bekler."""
        pending = set(self.target_ids) if not self.all_mode else None
        if self.all_mode:
            print("Herhangi bir araç bekleniyor...")
        else:
            print(f"Araç(lar) bekleniyor: {sorted(pending)} (iptal: Ctrl+C)")

        while True:
            msg = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if not msg or not self.is_vehicle(msg):
                continue
            sys_id = msg.get_srcSystem()
            self.seen_ids.add(sys_id)
            if self.all_mode:
                print(f"İlk Heartbeat alındı! SYS_ID: {sys_id}")
                return
            if sys_id in pending:
                pending.discard(sys_id)
                print(f"SYS_ID {sys_id} bulundu.")
            if not pending:
                return

    def cmd_targets(self):
        """command_long için hedefler: tüm sürüde tek yayın (0), değilse tek tek ID."""
        return [0] if self.all_mode else sorted(self.target_ids)

    def stream_targets(self):
        """Velocity / gimbal için hedefler: tüm sürüde görülen tüm araçlar, değilse verilen ID'ler."""
        return sorted(self.seen_ids) if self.all_mode else sorted(self.target_ids)

    def send_cmd(self, command, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0):
        for t in self.cmd_targets():
            self.master.mav.command_long_send(
                t, 0, command, 0, p1, p2, p3, p4, p5, p6, p7
            )

    def setup_joystick(self):
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Kol aktif: {self.joystick.get_name()}")
        else:
            print("HATA: Kol bulunamadı!")

    # ==========================================
    # SÜREKLİ GÖNDERİM (velocity + gimbal)
    # ==========================================
    def send_velocity(self, vx, vy, vz, yaw_rate):
        for t in self.stream_targets():
            try:
                self.master.mav.set_position_target_local_ned_send(
                    0, t, 0,
                    mavutil.mavlink.MAV_FRAME_BODY_NED,
                    VEL_TYPE_MASK,
                    0, 0, 0,
                    vx, vy, vz,
                    0, 0, 0,
                    0, yaw_rate,
                )
            except Exception:
                pass

    def send_gimbal(self):
        vals = [65535] * 18   # 65535 = bu kanalı değiştirme
        for ch, pwm in self.gimbal.items():
            vals[ch - 1] = int(pwm)
        for t in self.stream_targets():
            try:
                self.master.mav.rc_channels_override_send(t, 0, *vals)
            except Exception:
                pass

    # ==========================================
    # KAPANIŞ
    # ==========================================
    def request_stop(self):
        self._stop = True

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        print("\nKapatılıyor: hız sıfırlanıyor, gimbal override bırakılıyor...")

        release = [65535] * 18
        for g in GIMBAL:
            release[g["ch"] - 1] = 0   # 0 = override'dan çıkar
        for _ in range(3):
            self.send_velocity(0, 0, 0, 0)
            for t in self.stream_targets():
                try:
                    self.master.mav.rc_channels_override_send(t, 0, *release)
                except Exception:
                    pass
            time.sleep(0.05)

        try:
            self.master.close()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        self.root.destroy()
        print("Çıkış yapıldı.")

    # ==========================================
    # MAVLINK KOMUTLARI
    # ==========================================
    def set_mode(self, mode_id, mode_name):
        try:
            self.send_cmd(mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, mode_id)
            print(f" Mod değiştirildi ({self.target_label}): {mode_name}")
        except Exception as e:
            print(f"Mod değiştirme hatası: {e}")

    def arm(self):
        try:
            self.send_cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            print(f" ARM gönderildi ({self.target_label}).")
        except Exception as e:
            print(f"ARM hatası: {e}")

    def disarm(self):
        try:
            self.send_cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0)
            print(f" DISARM gönderildi ({self.target_label}).")
        except Exception as e:
            print(f"DISARM hatası: {e}")

    def takeoff(self):
        """GUIDED'a al + kalkış. (Guided'da velocity tek başına kalkış yaptırmaz.)"""
        try:
            self.set_mode(*MODES["RT"])
            self.send_cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, TAKEOFF_ALT)
            self.takeoff_until = time.time() + TAKEOFF_TIME
            print(f" TAKEOFF {TAKEOFF_ALT} m ({self.target_label}).")
        except Exception as e:
            print(f"Takeoff hatası: {e}")

    def kill_switch(self):
        """EMERGENCY MOTOR CUT + SES"""
        if self.kill_sound:
            self.kill_sound.play()
        try:
            self.send_cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)
            print(f"🚨 KILL SWITCH! Motorlar kapatıldı ({self.target_label}).")
        except Exception as e:
            print(f"Kill Switch hatası: {e}")

    def toggle_gimbal(self):
        self.gimbal_mode = not self.gimbal_mode
        self.update_status()
        print(f" Gimbal modu: {'AKTİF (RB=1)' if self.gimbal_mode else 'KAPALI, hız modu (RB=0)'}")

    # ==========================================
    # BUTON OKUMA
    # ==========================================
    def read_physical_buttons(self):
        if not self.joystick:
            return

        js = self.joystick
        nb = js.get_numbuttons()
        na = js.get_numaxes()

        def btn(i):
            return js.get_button(i) if nb > i else 0

        lt_axis = js.get_axis(2) if na > 2 else -1.0
        rt_axis = (
            js.get_axis(5) if na > 5
            else (js.get_axis(4) if na > 4 else -1.0)
        )

        # Yükselen kenar (basılma anı) tespiti
        def pressed(name, state):
            was = self.prev_button_states.get(name, False)
            self.prev_button_states[name] = bool(state)
            return bool(state) and not was

        if pressed("B", btn(1)):        # B     -> KILL
            self.kill_switch()
        if pressed("RB", btn(5)):       # RB    -> gimbal toggle
            self.toggle_gimbal()
        if pressed("LB", btn(4)):       # LB    -> DISARM
            self.disarm()
        if pressed("START", btn(7)):    # Start -> ARM
            self.arm()
        if pressed("A", btn(0)):        # A     -> TAKEOFF
            self.takeoff()
        if pressed("X", btn(2)):        # X     -> LAND
            self.set_mode(*MODES["X"])
        if pressed("Y", btn(3)):        # Y     -> LOITER
            self.set_mode(*MODES["Y"])
        if pressed("LT", lt_axis > 0.5):  # LT -> POSHOLD
            self.set_mode(*MODES["LT"])
        if pressed("RT", rt_axis > 0.5):  # RT -> GUIDED
            self.set_mode(*MODES["RT"])

    def stick(self, axis):
        if axis >= self.joystick.get_numaxes():
            return 0.0
        v = self.joystick.get_axis(axis)
        return 0.0 if abs(v) < DEADBAND else v

    # ==========================================
    # ARAYÜZ
    # ==========================================
    def create_widgets(self):
        self.status = tk.Label(self.root, text="", font=("Courier", 11, "bold"), anchor="w")
        self.status.pack(fill="x", pady=(0, 4))

        for s in STICKS:
            row = tk.Frame(self.root)
            row.pack(fill="x", pady=5)

            tk.Label(
                row, text=s["name"], width=10, anchor="w", font=("Arial", 10),
            ).pack(side="left")

            canvas = tk.Canvas(
                row, width=320, height=22, bg="lightgray",
                highlightthickness=1, highlightbackground="gray",
            )
            canvas.pack(side="left", padx=10)

            bar = canvas.create_rectangle(0, 0, 160, 22, fill="limegreen", outline="")
            canvas.create_line(160, 0, 160, 22, fill="black", dash=(2, 2))

            lbl = tk.Label(row, text="0.00", width=6, font=("Courier", 11, "bold"))
            lbl.pack(side="left")

            self.bars[s["key"]] = (canvas, bar)
            self.val_labels[s["key"]] = lbl

        self.update_status()

    def update_status(self):
        g = self.gimbal
        mode = "GİMBAL" if self.gimbal_mode else "HIZ"
        self.status.config(
            text=f"RB={int(self.gimbal_mode)} [{mode}]  G6:{int(g[6])} G7:{int(g[7])} G8:{int(g[8])}"
        )

    # ==========================================
    # ANA DÖNGÜ
    # ==========================================
    def update_and_send_loop(self):
        if self._stop:
            self.shutdown()
            return

        pygame.event.pump()

        # 1. Ağdaki araç Heartbeat'lerini topla
        while True:
            msg = self.master.recv_match(type="HEARTBEAT", blocking=False)
            if not msg:
                break
            if not self.is_vehicle(msg):
                continue
            sys_id = msg.get_srcSystem()
            if sys_id not in self.seen_ids:
                self.seen_ids.add(sys_id)
                if self.all_mode:
                    print(f" Sürüye yeni İHA eklendi! SYS_ID: {sys_id}")

        # 2. Joystick -> velocity / gimbal
        if self.joystick:
            self.read_physical_buttons()

            # Hız: gimbal modunda hedef 0 (filtre ile yumuşakça durur)
            for s in STICKS:
                target = 0.0 if self.gimbal_mode else self.stick(s["axis"]) * s["sign"] * s["scale"]
                k = s["key"]
                self.vel[k] += SMOOTH_ALPHA * (target - self.vel[k])

                canvas, bar = self.bars[k]
                canvas.coords(bar, 0, 0, (self.vel[k] / s["scale"] + 1) / 2 * 320, 22)
                self.val_labels[k].config(text=f"{self.vel[k]:+.2f}")

            # Gimbal: sadece RB=1 iken stick PWM'i değiştirir
            if self.gimbal_mode:
                for g in GIMBAL:
                    d = self.stick(g["axis"]) * g["sign"] * GIMBAL_RATE * DT
                    self.gimbal[g["ch"]] = min(GIMBAL_MAX, max(GIMBAL_MIN, self.gimbal[g["ch"]] + d))
                self.update_status()

            # 3. Hedef araç(lar)a gönder
            if time.time() >= self.takeoff_until:
                self.send_velocity(self.vel["vx"], self.vel["vy"], self.vel["vz"], self.vel["yaw"])
            self.send_gimbal()

        self.root.after(int(DT * 1000), self.update_and_send_loop)


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    try:
        app = GazeboF310GCS(root, args.ids)
    except KeyboardInterrupt:
        print("\nİptal edildi.")
        root.destroy()
        raise SystemExit(0)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.request_stop()
    finally:
        if not app._closed:
            app.shutdown()