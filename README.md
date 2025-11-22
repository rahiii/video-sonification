# Video Motion Sonification Prototype

This project records a short video from the webcam, extracts motion between frames, and turns that movement into sound. It outputs a WAV audio file and a motion plot.

---

## How to Run the Project

### 1. Create and activate a virtual environment

**Mac/Linux:**
```bash
python3 -m venv venv
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
pip install opencv-python numpy scipy sounddevice matplotlib
```

---

### 3. Run the program
```bash
python main.py
```

---

## What the program does
- Opens your webcam and records ~4 seconds of video  
- Extracts motion between frames  
- Converts the motion into sound  
- Saves:  
  - **motion_sonification.wav** (audio)  
  - **motion_plot.png** (motion graph)

---

## Output files
- `motion_sonification.wav` — audio generated from your movement  
- `motion_plot.png` — plot of motion over time  
