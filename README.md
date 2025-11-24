# Video Motion Sonification Prototype

This project records a short video from the webcam, extracts motion between frames, and turns that movement into sound. It outputs a WAV audio file and a motion plot.

---

## How to Run the Project

### 1. Create and activate a virtual environment

**Mac/Linux:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
py -m venv venv
venv\Scripts\activate
```

---

### 2. Install required packages
```bash
pip install opencv-contrib-python numpy mediapipe ultralytics scipy
```

---

### 3. Run the program
```bash
python main.py
```

---

## Config
### To use a Video File instead of Webcam:
**Edit config.py:**
```bash
INPUT_SOURCE = "dance.mp4"
```
