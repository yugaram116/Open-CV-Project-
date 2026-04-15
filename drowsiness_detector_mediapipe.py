"""
Driver Drowsiness Detection System
Uses OpenCV + MediaPipe Face Mesh (NO dlib required)
Detects drowsiness via EAR, yawning via MAR, and PERCLOS metric.

Install: pip install mediapipe opencv-python scipy numpy pygame imutils
Run:     python drowsiness_detector.py
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import time
import pygame
from collections import deque
import sys

# ─────────────────────────────────────────────────────────
#  THRESHOLDS  (tune these if you get false alerts)
# ─────────────────────────────────────────────────────────
EAR_THRESHOLD      = 0.22   # Eye Aspect Ratio: below → eyes closing
MAR_THRESHOLD      = 0.65   # Mouth Aspect Ratio: above → yawning
EAR_CONSEC_FRAMES  = 20     # Frames below EAR threshold before alert
MAR_CONSEC_FRAMES  = 15     # Frames above MAR threshold before yawn alert
PERCLOS_WINDOW     = 300    # Rolling frame window for PERCLOS (~15s at 20fps)
PERCLOS_THRESHOLD  = 0.30   # >30% eyes closed = drowsy

# ─────────────────────────────────────────────────────────
#  MEDIAPIPE FACE MESH LANDMARK INDICES
#  (from the 468-point Face Mesh model)
# ─────────────────────────────────────────────────────────
# Left eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
# Mouth (outer lips)
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]


# ─────────────────────────────────────────────────────────
#  METRIC CALCULATIONS
# ─────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Uses 6 landmark points per eye.
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    """
    MAR measures how open the mouth is.
    Uses 8 landmark points around the lips.
    """
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    # Vertical distances
    A = dist.euclidean(pts[2], pts[6])
    B = dist.euclidean(pts[3], pts[5])
    # Horizontal distance
    C = dist.euclidean(pts[0], pts[1])
    return (A + B) / (2.0 * C) if C != 0 else 0.0


def get_eye_points(landmarks, eye_indices, w, h):
    return np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices],
        dtype=np.int32
    )


# ─────────────────────────────────────────────────────────
#  SOUND ALERT
# ─────────────────────────────────────────────────────────
class AlertSound:
    def __init__(self):
        self.enabled = False
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._make_beep()
            self.enabled = True
            print("[Sound] Audio alerts enabled.")
        except Exception as e:
            print(f"[Sound] pygame not available ({e}). Visual alerts only.")

    def _make_beep(self):
        sr, dur, freq = 44100, 0.4, 880
        t    = np.linspace(0, dur, int(sr * dur), False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        stereo = np.column_stack([wave, wave])
        self.sound = pygame.sndarray.make_sound(stereo)

    def play(self):
        if self.enabled:
            try:
                if not pygame.mixer.get_busy():
                    self.sound.play()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────
#  HUD DRAWING HELPERS
# ─────────────────────────────────────────────────────────
COLORS = {
    "AWAKE":   (0, 220, 100),
    "WARNING": (0, 200, 255),
    "DROWSY":  (0, 50,  255),
    "YAWNING": (0, 165, 255),
    "NO FACE": (150, 150, 150),
}

def draw_hud(frame, ear, mar, perclos, alert_level, drowsy_count, yawn_count, elapsed):
    h, w = frame.shape[:2]
    color = COLORS.get(alert_level, (200, 200, 200))

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Left column — metrics
    cv2.putText(frame, f"EAR : {ear:.3f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.putText(frame, f"MAR : {mar:.3f}", (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
    cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%", (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # Right column — status
    cv2.putText(frame, alert_level, (w - 210, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

    # EAR progress bar
    bx, by, bw, bh = w - 210, 70, 190, 12
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    fill = int(min(ear / 0.40, 1.0) * bw)
    bc   = (0, 200, 100) if ear > EAR_THRESHOLD else (0, 50, 255)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), bc, -1)
    cv2.putText(frame, "EAR", (bx - 42, by + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Bottom session bar
    mins, secs = divmod(int(elapsed), 60)
    session_txt = f"Session {mins:02d}:{secs:02d}   Drowsy alerts: {drowsy_count}   Yawns: {yawn_count}"
    cv2.putText(frame, session_txt, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)


def draw_alert_banner(frame, message, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 85), (w, h - 20), color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, message,
                (w // 2 - len(message) * 9, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


def draw_eye_contour(frame, pts, color):
    hull = cv2.convexHull(pts)
    cv2.drawContours(frame, [hull], -1, color, 1)


# ─────────────────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────────────────
class DrowsinessDetector:

    def __init__(self, camera_index=0, save_video=False):
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera
        print(f"[INFO] Opening camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera index {camera_index}. "
                          "Try a different index (e.g. -c 1).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # State counters
        self.ear_counter    = 0
        self.mar_counter    = 0
        self.drowsy_count   = 0
        self.yawn_count     = 0
        self.total_frames   = 0
        self.perclos_buffer = deque(maxlen=PERCLOS_WINDOW)
        self.last_alert_ts  = 0
        self.alert_cooldown = 3.0

        self.sound = AlertSound()

        # Optional save
        self.writer = None
        if save_video:
            fourcc     = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

        print("[INFO] Ready! Press  Q  to quit.")
        print(f"       EAR threshold : {EAR_THRESHOLD}")
        print(f"       MAR threshold : {MAR_THRESHOLD}")
        print(f"       PERCLOS limit : {PERCLOS_THRESHOLD*100:.0f}%\n")

    # ── Frame processing ────────────────────────────────
    def process(self, frame):
        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = self.face_mesh.process(rgb)

        ear         = 0.0
        mar         = 0.0
        alert_level = "NO FACE"
        self.total_frames += 1

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ── Compute metrics ──────────────────────
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ear       = (left_ear + right_ear) / 2.0
            mar       = mouth_aspect_ratio(lm, MOUTH, w, h)

            # ── Draw eye / mouth contours ────────────
            le_pts = get_eye_points(lm, LEFT_EYE,  w, h)
            re_pts = get_eye_points(lm, RIGHT_EYE, w, h)
            mo_pts = get_eye_points(lm, MOUTH,     w, h)
            eye_color = (0, 50, 255) if ear < EAR_THRESHOLD else (0, 220, 100)
            draw_eye_contour(frame, le_pts, eye_color)
            draw_eye_contour(frame, re_pts, eye_color)
            draw_eye_contour(frame, mo_pts, (0, 165, 255))

            # ── PERCLOS ──────────────────────────────
            eyes_closed = ear < EAR_THRESHOLD
            self.perclos_buffer.append(1 if eyes_closed else 0)
            perclos = float(np.mean(self.perclos_buffer))

            # ── Alert logic ──────────────────────────
            alert_level = "AWAKE"

            if eyes_closed:
                self.ear_counter += 1
            else:
                self.ear_counter = 0

            if mar > MAR_THRESHOLD:
                self.mar_counter += 1
            else:
                if self.mar_counter >= MAR_CONSEC_FRAMES:
                    self.yawn_count += 1
                    print(f"[EVENT] Yawn #{self.yawn_count} detected")
                self.mar_counter = 0

            # Yawn
            if self.mar_counter >= MAR_CONSEC_FRAMES:
                alert_level = "YAWNING"
                draw_alert_banner(frame, "⚠  YAWNING DETECTED", (0, 100, 180))
                self._alert()

            # Drowsiness (consecutive frames OR PERCLOS)
            if self.ear_counter >= EAR_CONSEC_FRAMES or perclos > PERCLOS_THRESHOLD:
                alert_level = "DROWSY"
                draw_alert_banner(frame, "  DROWSINESS ALERT! WAKE UP!", (0, 0, 200))
                self._alert()
                if self.ear_counter >= EAR_CONSEC_FRAMES:
                    self.drowsy_count += 1
                    print(f"[ALERT] Drowsiness event #{self.drowsy_count}")
                    self.ear_counter = 0
            elif self.ear_counter >= EAR_CONSEC_FRAMES // 2:
                alert_level = "WARNING"

        else:
            perclos = float(np.mean(self.perclos_buffer)) if self.perclos_buffer else 0.0

        draw_hud(frame, ear, mar, perclos, alert_level,
                 self.drowsy_count, self.yawn_count,
                 time.time() - self.start_time)
        return frame

    def _alert(self):
        now = time.time()
        if now - self.last_alert_ts > self.alert_cooldown:
            self.sound.play()
            self.last_alert_ts = now

    # ── Main loop ────────────────────────────────────────
    def run(self):
        self.start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Cannot read from camera.")
                break

            frame = self.process(frame)
            cv2.imshow("Driver Drowsiness Detection  |  Q = quit", frame)

            if self.writer:
                self.writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        self._cleanup()

    def _cleanup(self):
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

        elapsed = int(time.time() - self.start_time)
        m, s    = divmod(elapsed, 60)
        print("\n" + "=" * 45)
        print("  SESSION SUMMARY")
        print("=" * 45)
        print(f"  Duration          : {m:02d}:{s:02d}")
        print(f"  Frames processed  : {self.total_frames}")
        print(f"  Drowsiness alerts : {self.drowsy_count}")
        print(f"  Yawn events       : {self.yawn_count}")
        print("=" * 45)


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection (MediaPipe)")
    parser.add_argument("-c", "--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save video to output.avi")
    args = parser.parse_args()

    try:
        detector = DrowsinessDetector(camera_index=args.camera, save_video=args.save)
        detector.run()
    except IOError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)
