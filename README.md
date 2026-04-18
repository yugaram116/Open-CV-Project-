# Driver Drowsiness Detection System Project(Open-CV)


Driver Drowsiness Detection System
Real-time driver fatigue monitoring using MediaPipe Face Mesh and OpenCV. Detects eye closure (EAR), yawning (MAR), and calculates PERCLOS to issue audio-visual alerts.

Features
Eye closure detection using Eye Aspect Ratio (EAR)

Yawning detection using Mouth Aspect Ratio (MAR)

Fatigue measurement using PERCLOS (rolling window)

Real-time audio alerts (880 Hz beep)

Live on-screen metrics with color-coded status

Optional video recording

Session summary with alert statistics

No dlib required - lightweight MediaPipe

How It Works

The system detects 468 facial landmarks using MediaPipe, then calculates:

Metric	Formula	Purpose
EAR	(vertical distances) / (2 × horizontal distance)	Lower value = eyes closed
MAR	(vertical distances) / (2 × horizontal distance)	Higher value = mouth open
PERCLOS	(closed frames) / (window size)	Percentage of eye closure over time
Landmark indices used:

Left eye: [362, 385, 387, 263, 373, 380]

Right eye: [33, 160, 158, 133, 153, 144]

Mouth: [61, 291, 39, 181, 0, 17, 269, 405]

Installation

bash
#Clone repository
git clone https://github.com/yugaram116/Open-CV-Project-.git
cd Open-CV-Project-

#Install dependencies
pip install -r requirements_mediapipe.txt
Requirements:

text
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
pygame>=2.5.0
imutils>=0.5.4
Usage
bash
#Basic usage (default camera)
python drowsiness_detector_mediapipe.py

#Use external camera
python drowsiness_detector_mediapipe.py -c 1

#Save video recording
python drowsiness_detector_mediapipe.py -s

#Compact version
python drowsiness_detector.py
Controls: Press Q or ESC to quit

Configuration
Adjust thresholds in the source code:

python
EAR_THRESHOLD      = 0.22   # Eye closure threshold
MAR_THRESHOLD      = 0.65   # Yawning threshold
EAR_CONSEC_FRAMES  = 20     # Frames below EAR before alert
MAR_CONSEC_FRAMES  = 15     # Frames above MAR before yawn alert
PERCLOS_THRESHOLD  = 0.30   # PERCLOS percentage threshold
Tuning guide:

Bright lighting → Increase EAR_THRESHOLD by 0.02-0.03

Poor lighting → Decrease EAR_THRESHOLD by 0.01-0.02

Small eyes → Decrease EAR_THRESHOLD by 0.02-0.03

Output
Real-time Display
Element	Location	Description
EAR, MAR, PERCLOS	Top-left	Current metric values
Status	Top-right	AWAKE / WARNING / DROWSY / YAWNING
EAR progress bar	Top-right	Visual indicator
Session timer	Bottom-left	Duration and alert counts
Eye contours	Green/Red	Red when eyes closed
Mouth contour	Orange	Visible during yawning
Status Colors
AWAKE - Green

WARNING - Yellow

DROWSY - Red

YAWNING - Orange

Session Summary (on quit)
text
=============================================
  SESSION SUMMARY
=============================================
  Duration          : 02:35
  Frames processed  : 3100
  Drowsiness alerts : 3
  Yawn events       : 5
=============================================
Alert System
Alert Type	Trigger	Visual	Audio
Drowsy	20+ frames closed OR PERCLOS > 30%	Red banner	880 Hz beep
Yawning	15+ frames mouth open	Orange banner	880 Hz beep
Warning	10+ frames closed	Yellow status	None
Cooldown: 3 seconds between audio alerts

Project Structure
text
Open-CV-Project-/
├── drowsiness_detector_mediapipe.py   # Main application
├── drowsiness_detector.py             # Compact version
├── requirements_mediapipe.txt         # Dependencies
├── README.md                          # Documentation
├── screenshots/                       # Images for README
└── output.avi                         # Recording (if -s used)
Troubleshooting
Issue	Solution
Camera not opening	Try -c 1 or -c 2
Poor detection	Improve lighting, sit 1-2 feet from camera
False alerts	Adjust EAR/MAR thresholds
No audio	Run pip install pygame
Low FPS	Reduce resolution in code
Future Improvements
Mobile app integration for fleet monitoring

Vehicle system integration (CAN bus)

Deep learning-based fatigue detection

Night vision with infrared camera

Cloud dashboard for real-time monitoring

Contributing
Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -m 'Add feature')

Push to branch (git push origin feature/improvement)

Open a Pull Request

License
MIT License - see repository for details.

Author
Yugaram TS
GitHub: @yugaram116

Disclaimer
This project is for educational purposes only. It should NOT be used as a sole safety system in real-world driving conditions. Always drive responsibly and take breaks when tired. The authors are not responsible for any accidents or damages resulting from use of this software.
