import streamlit as st
import numpy as np
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

import mediapipe as mp
from scipy.spatial import distance as dist

st.title("🚗 Driver Drowsiness Detection (NO OpenCV)")

# -------- CONFIG --------
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

mp_face_mesh = mp.solutions.face_mesh


# -------- FUNCTIONS --------
def calc_ear(lm, idx, w, h):
    p = [(lm[i].x * w, lm[i].y * h) for i in idx]
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A + B) / (2.0 * C) if C else 0.0


def calc_mar(lm, idx, w, h):
    p = [(lm[i].x * w, lm[i].y * h) for i in idx]
    A = dist.euclidean(p[2], p[6])
    B = dist.euclidean(p[3], p[5])
    C = dist.euclidean(p[0], p[1])
    return (A + B) / (2.0 * C) if C else 0.0


# -------- PROCESSOR --------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.eye_frames = 0
        self.yawn_frames = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = img[:, :, ::-1]  # BGR → RGB (no cv2)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            ear = (calc_ear(lm, LEFT_EYE, w, h) +
                   calc_ear(lm, RIGHT_EYE, w, h)) / 2.0

            mar = calc_mar(lm, MOUTH, w, h)

            if ear < EAR_THRESHOLD:
                self.eye_frames += 1
            else:
                self.eye_frames = 0

            if mar > MAR_THRESHOLD:
                self.yawn_frames += 1
            else:
                self.yawn_frames = 0

            if self.eye_frames > 15 or self.yawn_frames > 10:
                status = "SLEEPY 😴"
            else:
                status = "ACTIVE 😃"
        else:
            status = "NO FACE"

        # Draw text using numpy (simple overlay)
        img[20:80, 20:400] = [0, 0, 0]
        st_text = f"STATUS: {status}"
        for i, c in enumerate(st_text):
            if i*10+30 < img.shape[1]:
                img[50, i*10+30:i*10+35] = [255, 255, 255]

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------- WEBRTC --------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsiness",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=DrowsinessProcessor,
)
