# Open-CV-Project-

🚗 Driver Drowsiness Detection System

A real-time Driver Drowsiness Detection System using Computer Vision and MediaPipe Face Mesh.
This project monitors eye closure, yawning, and fatigue levels to alert drivers and prevent accidents.

📌 Features
👁️ Eye closure detection using EAR (Eye Aspect Ratio)
😮 Yawning detection using MAR (Mouth Aspect Ratio)
⏱️ Fatigue measurement using PERCLOS (Percentage of Eye Closure)
🔊 Real-time audio alerts
📊 Live on-screen metrics (EAR, MAR, PERCLOS)
🎥 Optional video recording
⚡ Lightweight (No dlib required)
🧠 How It Works

The system uses MediaPipe Face Mesh (468 landmarks) to track facial features.

Key Metrics:
EAR (Eye Aspect Ratio)
Detects eye closure
Lower value → eyes closed
MAR (Mouth Aspect Ratio)
Detects yawning
Higher value → mouth open
PERCLOS
% of time eyes remain closed over a period
Strong indicator of drowsiness
🏗️ Project Structure
📁 Driver-Drowsiness-Detection
│── drowsiness_detector.py              # Basic version
│── drowsiness_detector_mediapipe.py    # Advanced modular version
│── requirements_mediapipe.txt          # Dependencies
│── README.md
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/drowsiness-detection.git
cd drowsiness-detection
2️⃣ Install dependencies
pip install -r requirements_mediapipe.txt
▶️ Usage
Run the system:
python drowsiness_detector_mediapipe.py
Optional arguments:
python drowsiness_detector_mediapipe.py -c 0     # Camera index
python drowsiness_detector_mediapipe.py -s       # Save output video
🧪 Threshold Configuration

You can tune detection sensitivity in the code:

EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.65
PERCLOS_THRESHOLD = 0.30

Adjust these values based on lighting, camera quality, and user behavior.

📸 Output Preview
Real-time webcam feed
Eye & mouth contours
Status labels:
🟢 AWAKE
🟡 WARNING
🟠 YAWNING
🔴 DROWSY
🔊 Alert System
Plays a beep sound when:
Drowsiness is detected
Yawning exceeds threshold
Falls back to visual alerts if audio is unavailable
💡 Key Concepts Used
Computer Vision
Facial Landmark Detection
Signal Processing (EAR, MAR)
Real-time Video Processing
Human Attention Monitoring
🚀 Future Improvements
📱 Mobile app integration
🚘 Integration with vehicle systems
🧠 Deep learning-based fatigue detection
🌙 Night vision optimization
☁️ Cloud monitoring dashboard
🛠️ Tech Stack
Python
OpenCV
MediaPipe
NumPy
SciPy
Pygame
🤝 Contributing

Contributions are welcome!

Fork the repo
Create a new branch
Make your changes
Submit a pull request
📜 License

This project is open-source and available under the MIT License.

👨‍💻 Author

Yugaram TS

GitHub: yugaram116
⭐ Support

If you like this project:

⭐ Star the repo
🍴 Fork it
📢 Share it
⚠️ Disclaimer

This project is for educational purposes only.
It should not be used as a sole safety system in real-world driving conditions.
