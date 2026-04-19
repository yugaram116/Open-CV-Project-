import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import mediapipe as mp
from scipy.spatial import distance as dist

st.title("🚗 Driver Drowsiness Detection System")

# --------- CONFIG ----------
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65

LEFT_EYE = [362,385,387,263,373,380]
RIGHT_EYE = [33,160,158,133,153,144]
MOUTH = [61,291,39,181,0,17,269,405]

mp_face_mesh = mp.solutions.face_mesh

# --------- FUNCTIONS ----------
def calc_ear(lm, idx, w, h):
    p = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx]
    A = dist.euclidean(p[1], p[5])
    B = dist.euclidean(p[2], p[4])
    C = dist.euclidean(p[0], p[3])
    return (A+B)/(2.0*C) if C else 0.0

def calc_mar(lm, idx, w, h):
    p = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx]
    A = dist.euclidean(p[2], p[6])
    B = dist.euclidean(p[3], p[5])
    C = dist.euclidean(p[0], p[1])
    return (A+B)/(2.0*C) if C else 0.0

# --------- VIDEO PROCESSOR ----------
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        status = "NO FACE"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            ear = (calc_ear(lm, LEFT_EYE, w, h) +
                   calc_ear(lm, RIGHT_EYE, w, h)) / 2.0

            mar = calc_mar(lm, MOUTH, w, h)

            if ear < EAR_THRESHOLD:
                status = "DROWSY 😴"
                color = (0, 0, 255)
            elif mar > MAR_THRESHOLD:
                status = "YAWNING 🥱"
                color = (0, 165, 255)
            else:
                status = "AWAKE 🙂"
                color = (0, 255, 0)

            cv2.putText(img, status, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------- STREAM ----------
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

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
