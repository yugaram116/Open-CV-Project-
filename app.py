import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import math

st.set_page_config(page_title="Drowsiness Detector", layout="centered")
st.title("😴 Drowsiness Detection (WebRTC)")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def calculate_EAR(eye_points, landmarks, w, h):
    """Compute Eye Aspect Ratio (EAR)"""
    def dist(p1, p2):
        return math.hypot(p1.x * w - p2.x * w, p1.y * h - p2.y * h)

    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]

    vertical1 = dist(p2, p6)
    vertical2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.counter = 0
        self.threshold = 0.25   # EAR threshold
        self.frames_closed = 15 # frames to trigger alert

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Calculate EAR for both eyes
                left_ear = calculate_EAR(LEFT_EYE, landmarks, w, h)
                right_ear = calculate_EAR(RIGHT_EYE, landmarks, w, h)
                ear = (left_ear + right_ear) / 2.0

                # Draw EAR value
                cv2.putText(img, f"EAR: {ear:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Drowsiness detection
                if ear < self.threshold:
                    self.counter += 1
                else:
                    self.counter = 0

                if self.counter > self.frames_closed:
                    cv2.putText(img, "DROWSY!", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# WebRTC config (important for deployment)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start webcam stream
webrtc_streamer(
    key="drowsiness",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)
