# 🚗 Driver Drowsiness Detection System

A **real-time AI-powered driver monitoring system** that detects fatigue using **computer vision techniques**. The system analyzes facial landmarks to identify **eye closure, yawning, and drowsiness levels**, and triggers **audio-visual alerts** to improve road safety.

---

## 🧠 Overview

This project uses:

* **OpenCV** for real-time video processing
* **MediaPipe** Face Mesh for facial landmark detection
* **Streamlit** for browser-based webcam app
* **NumPy** for calculations

It detects drowsiness using three key metrics:

* **EAR (Eye Aspect Ratio)** → detects eye closure
* **MAR (Mouth Aspect Ratio)** → detects yawning
* **PERCLOS** → percentage of eye closure over time

---

## 🚀 Features

* 👁️ **Eye Closure Detection (EAR)**
* 😮 **Yawning Detection (MAR)**
* 📊 **PERCLOS Fatigue Measurement**
* 🔊 **Audio Alerts (Beep Sound)**
* 🎥 **Real-time Webcam Monitoring**
* 🧾 **Session Summary (alerts, duration, yawns)**
* 🌐 **Streamlit Web App (Face Detection Demo)**
* ⚡ **Lightweight – No dlib required**

---

## 🖥️ Applications Included

### 1️⃣ Drowsiness Detection (Main System)

```bash
python drowsiness_detector_mediapipe.py
```

* Uses MediaPipe Face Mesh (468 landmarks)
* Full-featured detection system
* Audio + visual alerts
* Optional video recording

---

### 2️⃣ Compact Version

```bash
python drowsiness_detector.py
```

* Simplified and faster version
* Core detection logic only

---

### 3️⃣ Streamlit Face Detection App

```bash
streamlit run app.py
```

* Real-time webcam in browser
* Face detection using Haar Cascades

---

## 📊 Detection Metrics

| Metric  | Formula                                          | Purpose               |
| ------- | ------------------------------------------------ | --------------------- |
| EAR     | (vertical distances) / (2 × horizontal distance) | Eye closure detection |
| MAR     | (vertical distances) / (2 × horizontal distance) | Yawning detection     |
| PERCLOS | closed frames / total frames                     | Fatigue level         |

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or for full MediaPipe version:

```bash
pip install mediapipe opencv-python numpy scipy pygame imutils
```

---

## 🔑 Requirements

### Python Dependencies

* streamlit
* streamlit-webrtc
* opencv-python-headless
* numpy
* protobuf

### System Packages (Linux)

* libgl1
* libsm6
* libxext6
* libxrender1

---

## ▶️ Usage Options

### Default Camera

```bash
python drowsiness_detector_mediapipe.py
```

### External Camera

```bash
python drowsiness_detector_mediapipe.py -c 1
```

### Save Video Output

```bash
python drowsiness_detector_mediapipe.py -s
```

---

## 🎮 Controls

* Press **Q** or **ESC** → Quit application

---

## ⚙️ Configuration

Modify thresholds in code:

```python
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65
PERCLOS_THRESHOLD = 0.30
```

### Tuning Tips

* Bright light → increase EAR
* Low light → decrease EAR
* False alerts → adjust MAR/PERCLOS

---

## 📈 Output

### Real-Time Display

* EAR, MAR, PERCLOS values
* Status: **AWAKE / WARNING / DROWSY / YAWNING**
* Colored overlays & contours
* Session timer

### Alert System

| Alert   | Trigger                    | Action            |
| ------- | -------------------------- | ----------------- |
| Drowsy  | Eyes closed / high PERCLOS | 🔴 Visual + Sound |
| Yawning | Mouth open long            | 🟠 Visual + Sound |
| Warning | Partial closure            | 🟡 Visual only    |

---

## 📁 Project Structure

```
.
├── app.py
├── drowsiness_detector.py
├── drowsiness_detector_mediapipe.py
├── requirements.txt
├── runtime.txt
├── packages.txt
└── README.md
```

---

## 🔮 Future Improvements

* 📱 Mobile app integration
* 🚗 Vehicle system integration
* 🤖 Deep learning-based detection
* 🌙 Night vision support
* ☁️ Cloud dashboard

---

## 🤝 Contributing

```bash
fork → branch → commit → push → pull request
```

---

## 📜 License

MIT License

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It should NOT be used as a **real-world safety system** without proper validation.

---

## 👨‍💻 Author

**Yugaram TS**
GitHub: [https://github.com/yugaram116](https://github.com/yugaram116)

---
