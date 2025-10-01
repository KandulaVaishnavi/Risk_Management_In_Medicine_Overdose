import os
import json
import csv
import time
import math
import threading
import traceback
from datetime import datetime
from threading import Thread
import cv2
import mediapipe as mp
import pygame
import pyttsx3
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None
SLEEP_FILE = "audio_alert.wav"   
DROWSY_FILE = "dddd.wav"         
LOG_FILE = "drowsiness_log.csv"
SETTINGS_FILE = "settings.json"
EAR_THRESH_OPEN = 0.25
EAR_THRESH_DROWSY = 0.21
FRAME_THRESH = 6
ALERT_COOLDOWN = 30.0
pygame.init()
try:
    pygame.mixer.init()
except Exception:
    pass
sleep_sound = pygame.mixer.Sound(SLEEP_FILE) if os.path.exists(SLEEP_FILE) else None
drowsy_sound = pygame.mixer.Sound(DROWSY_FILE) if os.path.exists(DROWSY_FILE) else None
alarm_channel = None
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_lock = threading.Lock()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check your webcam.")
running = False
status = "ACTIVE"
sleep_count = drowsy_count = active_count = 0
active_timer_start = time.time()
total_active_time = 0.0
drowsy_events = 0
sleep_events = 0
_last_alert_times = {"DROWSY": 0.0, "SLEEPING": 0.0}
def euclidean_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
def get_ear(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    up = euclidean_dist(pts[1], pts[5]) + euclidean_dist(pts[2], pts[4])
    across = euclidean_dist(pts[0], pts[3]) + 1e-8
    return up / (2.0 * across)
def speak_async(text):
    def _work():
        with tts_lock:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception:
                pass
    Thread(target=_work, daemon=True).start()
def safe_append_csv(path, row):
    try:
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception:
        print("Failed to write log:", traceback.format_exc())
def default_settings():
    return {
        "email_sender": "",
        "email_password": "",
        "email_receiver": "",
        "enable_email": False,
        "twilio_sid": "",
        "twilio_auth": "",
        "twilio_number": "",
        "phone_number": "",
        "enable_sms": False,
        "ear_thresh_open": EAR_THRESH_OPEN,
        "ear_thresh_drowsy": EAR_THRESH_DROWSY,
        "frame_thresh": FRAME_THRESH,
        "car_mode": "Normal",
        "enable_sound": True,
        "enable_voice": True
    }
def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                s = json.load(f)
                # ensure keys exist
                defs = default_settings()
                for k, v in defs.items():
                    if k not in s:
                        s[k] = v
                return s
    except Exception:
        print("Failed to load settings:", traceback.format_exc())
    return default_settings()
def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception:
        print("Failed to save settings:", traceback.format_exc())
def send_email_async(settings, car_mode, alert_status):
    if not settings.get("enable_email", False):
        return
    def _send():
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            now = time.time()
            if now - _last_alert_times.get(alert_status, 0.0) < ALERT_COOLDOWN:
                return
            msg = MIMEMultipart()
            msg["From"] = settings["email_sender"]
            msg["To"] = settings["email_receiver"]
            msg["Subject"] = f"Drowsiness Alert: {alert_status}"
            body = f"Driver Alert\n\nStatus: {alert_status}\nCar Mode: {car_mode}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(body, "plain"))
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
            server.starttls()
            server.login(settings["email_sender"], settings["email_password"])
            server.sendmail(settings["email_sender"], settings["email_receiver"], msg.as_string())
            server.quit()
            print("[EMAIL] Sent:", alert_status)
        except Exception as e:
            print("[EMAIL ERROR]", e)
        finally:
            _last_alert_times[alert_status] = time.time()
    Thread(target=_send, daemon=True).start()
def send_sms_async(settings, car_mode, alert_status):
    if not settings.get("enable_sms", False):
        return
    if TwilioClient is None:
        print("[SMS] Twilio library not installed.")
        return
    def _send():
        try:
            now = time.time()
            if now - _last_alert_times.get(alert_status, 0.0) < ALERT_COOLDOWN:
                return
            sid = settings.get("twilio_sid", "").strip()
            auth = settings.get("twilio_auth", "").strip()
            from_num = settings.get("twilio_number", "").strip()
            to_num = settings.get("phone_number", "").strip()
            if not sid or not auth or not from_num or not to_num:
                print("[SMS] Twilio settings incomplete.")
                return
            client = TwilioClient(sid, auth)
            body = f"Drowsiness Alert: {alert_status}\nCar Mode: {car_mode}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg = client.messages.create(body=body, from_=from_num, to=to_num)
            print("[SMS] Sent:", msg.sid)
        except Exception as e:
            print("[SMS ERROR]", e)
        finally:
            _last_alert_times[alert_status] = time.time()
    Thread(target=_send, daemon=True).start()
if not os.path.exists(LOG_FILE):
    try:
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Car Mode", "Alert Status", "EAR"])
    except Exception:
        pass
class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection System (Mediapipe)")
        self.root.geometry("1000x860")
        self.root.configure(bg="#263238")
        self.settings = load_settings()
        self.status_label = tk.Label(root, text="Status: ACTIVE", font=("Helvetica", 20, "bold"),
                                     fg="lime", bg="#263238")
        self.status_label.pack(pady=(10,0))
        self.timer_label = tk.Label(root, text="Active Time: 00:00", font=("Helvetica", 16), fg="white", bg="#263238")
        self.timer_label.pack(pady=(2,10))
        self.video_frame = tk.Frame(root, bg="#000")
        self.video_frame.pack(pady=6)
        self.video_label = tk.Label(self.video_frame, bg="#000")
        self.video_label.pack()
        settings_frame = tk.Frame(root, bg="#2b3a3f")
        settings_frame.pack(fill=tk.BOTH, padx=12, pady=10, expand=False)
        left_col = tk.Frame(settings_frame, bg="#2b3a3f")
        left_col.pack(side=tk.LEFT, padx=12, pady=8)
        tk.Label(left_col, text="Car Mode", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.car_mode_var = tk.StringVar(value=self.settings.get("car_mode", "Normal"))
        ttk.Combobox(left_col, textvariable=self.car_mode_var, values=["Normal", "Automatic"], state="readonly", width=18).pack(pady=4)
        self.enable_sound_var = tk.BooleanVar(value=self.settings.get("enable_sound", True))
        self.enable_voice_var = tk.BooleanVar(value=self.settings.get("enable_voice", True))
        tk.Checkbutton(left_col, text="Enable Sound Alerts", variable=self.enable_sound_var, bg="#2b3a3f", fg="white").pack(anchor="w")
        tk.Checkbutton(left_col, text="Enable Voice Alerts", variable=self.enable_voice_var, bg="#2b3a3f", fg="white").pack(anchor="w")
        tk.Label(left_col, text="EAR Open Threshold", bg="#2b3a3f", fg="white").pack(anchor="w", pady=(8,0))
        self.ear_open_var = tk.DoubleVar(value=self.settings.get("ear_thresh_open", EAR_THRESH_OPEN))
        ttk.Scale(left_col, from_=0.08, to=0.45, variable=self.ear_open_var, orient=tk.HORIZONTAL, length=220).pack()
        tk.Label(left_col, text="EAR Drowsy Threshold", bg="#2b3a3f", fg="white").pack(anchor="w", pady=(6,0))
        self.ear_drowsy_var = tk.DoubleVar(value=self.settings.get("ear_thresh_drowsy", EAR_THRESH_DROWSY))
        ttk.Scale(left_col, from_=0.05, to=0.35, variable=self.ear_drowsy_var, orient=tk.HORIZONTAL, length=220).pack()
        tk.Label(left_col, text="Frame Threshold", bg="#2b3a3f", fg="white").pack(anchor="w", pady=(6,0))
        self.frame_thresh_var = tk.IntVar(value=self.settings.get("frame_thresh", FRAME_THRESH))
        tk.Spinbox(left_col, from_=1, to=40, textvariable=self.frame_thresh_var, width=6).pack(anchor="w", pady=(2,0))
        btns = tk.Frame(left_col, bg="#2b3a3f")
        btns.pack(pady=12)
        tk.Button(btns, text="Start", bg="#27ae60", fg="white", width=10, command=self.start_detection).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Stop", bg="#c0392b", fg="white", width=10, command=self.stop_detection).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Summary", bg="#3498db", fg="white", width=10, command=self.show_summary).pack(side=tk.LEFT, padx=6)
        right_col = tk.Frame(settings_frame, bg="#2b3a3f")
        right_col.pack(side=tk.LEFT, padx=30, pady=8)
        tk.Label(right_col, text="Email Alerts (Gmail recommended)", bg="#2b3a3f", fg="yellow").pack(anchor="w")
        tk.Label(right_col, text="Sender (Gmail)", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.email_sender_ent = tk.Entry(right_col, width=36)
        self.email_sender_ent.insert(0, self.settings.get("email_sender", ""))
        self.email_sender_ent.pack()
        tk.Label(right_col, text="App Password", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.email_pass_ent = tk.Entry(right_col, width=36, show="*")
        self.email_pass_ent.insert(0, self.settings.get("email_password", ""))
        self.email_pass_ent.pack()
        tk.Label(right_col, text="Receiver Email", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.email_recv_ent = tk.Entry(right_col, width=36)
        self.email_recv_ent.insert(0, self.settings.get("email_receiver", ""))
        self.email_recv_ent.pack()
        self.enable_email_var = tk.BooleanVar(value=self.settings.get("enable_email", False))
        tk.Checkbutton(right_col, text="Enable Email Alerts", variable=self.enable_email_var, bg="#2b3a3f", fg="white").pack(anchor="w", pady=(4,8))
        tk.Label(right_col, text="SMS/WhatsApp Alerts (Twilio)", bg="#2b3a3f", fg="lightgreen").pack(anchor="w")
        tk.Label(right_col, text="Twilio SID", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.tw_sid_ent = tk.Entry(right_col, width=36)
        self.tw_sid_ent.insert(0, self.settings.get("twilio_sid", ""))
        self.tw_sid_ent.pack()
        tk.Label(right_col, text="Twilio Auth Token", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.tw_auth_ent = tk.Entry(right_col, width=36, show="*")
        self.tw_auth_ent.insert(0, self.settings.get("twilio_auth", ""))
        self.tw_auth_ent.pack()
        tk.Label(right_col, text="Twilio Number (e.g. +1XXX...)", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.tw_from_ent = tk.Entry(right_col, width=36)
        self.tw_from_ent.insert(0, self.settings.get("twilio_number", ""))
        self.tw_from_ent.pack()
        tk.Label(right_col, text="Your Phone (e.g. +1XXX...) or WhatsApp 'whatsapp:+...'", bg="#2b3a3f", fg="white").pack(anchor="w")
        self.tw_to_ent = tk.Entry(right_col, width=36)
        self.tw_to_ent.insert(0, self.settings.get("phone_number", ""))
        self.tw_to_ent.pack()
        self.enable_sms_var = tk.BooleanVar(value=self.settings.get("enable_sms", False))
        tk.Checkbutton(right_col, text="Enable SMS/WhatsApp Alerts", variable=self.enable_sms_var, bg="#2b3a3f", fg="white").pack(anchor="w", pady=(4,8))
        bottom_frame = tk.Frame(root, bg="#263238")
        bottom_frame.pack(pady=8)
        tk.Button(bottom_frame, text="Save Settings", bg="#8e44ad", fg="white", command=self.save_settings, width=14).pack(side=tk.LEFT, padx=8)
        tk.Button(bottom_frame, text="Export Log", bg="#2c3e50", fg="white", command=self.export_log, width=14).pack(side=tk.LEFT, padx=8)
        tk.Button(bottom_frame, text="Quit", bg="#7f8c8d", fg="white", command=self.quit_app, width=14).pack(side=tk.LEFT, padx=8)
        self.stats_label = tk.Label(root, text="Drowsy: 0 | Sleeping: 0", font=("Helvetica", 14), fg="cyan", bg="#263238")
        self.stats_label.pack(pady=(6,0))
        self.ear_open_var.set(self.settings.get("ear_thresh_open", EAR_THRESH_OPEN))
        self.ear_drowsy_var.set(self.settings.get("ear_thresh_drowsy", EAR_THRESH_DROWSY))
        self.frame_thresh_var.set(self.settings.get("frame_thresh", FRAME_THRESH))
        self.car_mode_var.set(self.settings.get("car_mode", "Normal"))
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self._update_loop()
    def save_settings(self):
        s = {
            "email_sender": self.email_sender_ent.get().strip(),
            "email_password": self.email_pass_ent.get(),
            "email_receiver": self.email_recv_ent.get().strip(),
            "enable_email": bool(self.enable_email_var.get()),
            "twilio_sid": self.tw_sid_ent.get().strip(),
            "twilio_auth": self.tw_auth_ent.get(),
            "twilio_number": self.tw_from_ent.get().strip(),
            "phone_number": self.tw_to_ent.get().strip(),
            "enable_sms": bool(self.enable_sms_var.get()),
            "ear_thresh_open": float(self.ear_open_var.get()),
            "ear_thresh_drowsy": float(self.ear_drowsy_var.get()),
            "frame_thresh": int(self.frame_thresh_var.get()),
            "car_mode": self.car_mode_var.get(),
            "enable_sound": bool(self.enable_sound_var.get()),
            "enable_voice": bool(self.enable_voice_var.get())
        }
        save_settings(s)
        messagebox.showinfo("Settings", "Settings saved to settings.json")
        global EAR_THRESH_OPEN, EAR_THRESH_DROWSY, FRAME_THRESH
        EAR_THRESH_OPEN = s["ear_thresh_open"]
        EAR_THRESH_DROWSY = s["ear_thresh_drowsy"]
        FRAME_THRESH = s["frame_thresh"]
    def export_log(self):
        try:
            dst = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="drowsiness_log_export.csv")
            if dst:
                with open(LOG_FILE, "r", newline="") as src, open(dst, "w", newline="") as dstf:
                    dstf.write(src.read())
                messagebox.showinfo("Export", f"Log exported to {dst}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    def start_detection(self):
        global running, active_timer_start
        running = True
        active_timer_start = time.time()
    def stop_detection(self):
        global running, alarm_channel
        running = False
        if alarm_channel:
            try:
                alarm_channel.stop()
            except Exception:
                pass
    def quit_app(self):
        global running, alarm_channel
        running = False
        if alarm_channel:
            try:
                alarm_channel.stop()
            except Exception:
                pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
            pygame.quit()
        except Exception:
            pass
        try:
            face_mesh.close()
        except Exception:
            pass
        self.root.destroy()
    def show_summary(self):
        global drowsy_events, sleep_events
        msg = f"Session Summary\n\nActive Time: {get_timer_text()}\nDrowsy Events: {drowsy_events}\nSleeping Events: {sleep_events}"
        messagebox.showinfo("Summary", msg)
    def _trigger_alerts(self, alert_status, ear_val):
        """
        Handles logging, playing alerts, sending email/sms according to mode & settings.
        Runs in main loop (non-blocking network calls).
        """
        global drowsy_events, sleep_events, alarm_channel
        car_mode = self.car_mode_var.get()
        settings = {
            "email_sender": self.email_sender_ent.get().strip(),
            "email_password": self.email_pass_ent.get(),
            "email_receiver": self.email_recv_ent.get().strip(),
            "enable_email": bool(self.enable_email_var.get()),
            "twilio_sid": self.tw_sid_ent.get().strip(),
            "twilio_auth": self.tw_auth_ent.get(),
            "twilio_number": self.tw_from_ent.get().strip(),
            "phone_number": self.tw_to_ent.get().strip(),
            "enable_sms": bool(self.enable_sms_var.get())
        }
        # Log row
        safe_append_csv(LOG_FILE, [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), car_mode, alert_status, f"{ear_val:.4f}"])
        send_email_async(settings, car_mode, alert_status)
        send_sms_async(settings, car_mode, alert_status)
        if car_mode == "Automatic":
            if self.enable_voice_var.get():
                if alert_status == "DROWSY":
                    speak_async("Warning. Driver appears drowsy.")
                else:
                    speak_async("Warning. Driver may be sleeping. Please take control.")
            return
        if alert_status == "DROWSY":
            drowsy_events += 1
            if self.enable_sound_var.get() and drowsy_sound is not None:
                try:
                    ch = pygame.mixer.find_channel() or pygame.mixer.Channel(0)
                    ch.play(drowsy_sound)
                except Exception:
                    pass
            if self.enable_voice_var.get():
                speak_async("Stay alert. You are feeling drowsy.")
        else:
            sleep_events += 1
            if self.enable_sound_var.get() and sleep_sound is not None:
                try:
                    ch = pygame.mixer.find_channel() or pygame.mixer.Channel(0)
                    ch.play(sleep_sound)
                except Exception:
                    pass
            if self.enable_voice_var.get():
                speak_async("Wake up. You are falling asleep!")
    def _update_loop(self):
        """
        Main continuous updating loop using Tkinter's after().
        """
        global running, status, sleep_count, drowsy_count, active_count, total_active_time, active_timer_start
        global drowsy_events, sleep_events
        if running:
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                new_status = status
                ear_val = 0.0
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    left = get_ear(lm, LEFT_EYE, w, h)
                    right = get_ear(lm, RIGHT_EYE, w, h)
                    ear_val = (left + right) / 2.0
                    ear_open = float(self.ear_open_var.get())
                    ear_drowsy = float(self.ear_drowsy_var.get())
                    frame_thresh = int(self.frame_thresh_var.get())
                    if ear_val < ear_drowsy:
                        sleep_count += 1
                        drowsy_count = active_count = 0
                        if sleep_count >= frame_thresh:
                            new_status = "SLEEPING"
                    elif ear_val < ear_open:
                        drowsy_count += 1
                        sleep_count = active_count = 0
                        if drowsy_count >= frame_thresh:
                            new_status = "DROWSY"
                    else:
                        active_count += 1
                        sleep_count = drowsy_count = 0
                        if active_count >= frame_thresh:
                            new_status = "ACTIVE"
                    if new_status != status:
                        if status == "ACTIVE":
                            total_active_time += time.time() - active_timer_start
                        if new_status == "ACTIVE":
                            active_timer_start = time.time()
                        try:
                            ch = pygame.mixer.find_channel()
                            if ch: ch.stop()
                        except Exception:
                            pass
                        self._trigger_alerts(new_status, ear_val)
                        status = new_status
                    color_bgr = (0, 255, 0) if status == "ACTIVE" else (0, 255, 255) if status == "DROWSY" else (0, 0, 255)
                    cv2.putText(frame, f"{status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_bgr, 3)
                    cv2.putText(frame, f"EAR: {ear_val:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                else:
                    cv2.putText(frame, "NO FACE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                display = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (800, 600))
                img = Image.fromarray(display)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        self.status_label.configure(text=f"Status: {status}",
            fg="red" if status=="SLEEPING" else "orange" if status=="DROWSY" else "lime")
        self.timer_label.configure(text=f"Active Time: {get_timer_text()}")
        self.stats_label.configure(text=f"Drowsy: {drowsy_events} | Sleeping: {sleep_events}")
        self.root.after(30, self._update_loop)
def get_timer_text():
    elapsed = total_active_time
    if status == "ACTIVE":
        elapsed += time.time() - active_timer_start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    return f"{mins:02d}:{secs:02d}"
if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Car Mode", "Alert Status", "EAR"])
        except Exception:
            pass
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()