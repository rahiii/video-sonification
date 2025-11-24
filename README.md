# Video Motion Sonification

Real-time motion-to-sound conversion system that transforms human movement into synthesized audio. Uses pose detection, optical flow, and spectral synthesis to create dynamic soundscapes from video input.

## Installation

### 1. Create and activate virtual environment

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

### 2. Install required packages

```bash
pip install opencv-python numpy mediapipe librosa soundfile scipy moviepy
```

**Required packages:**
- `opencv-python` - Video processing and optical flow
- `numpy` - Numerical operations
- `mediapipe` - Pose detection and segmentation
- `librosa` - Audio analysis and synthesis
- `soundfile` - Audio file I/O
- `scipy` - Scientific computing (filtering, etc.)
- `moviepy` - Video/audio merging

### 3. Download pose model

The `pose_landmarker_full.task` file must be in the project root directory. If missing, download it from [MediaPipe Pose Solutions](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).

## Usage

### Run the program

```bash
python main.py
```

When prompted:
- **Enter video path**: Type path to video file (e.g., `Samples/dance.mp4`) or press Enter for webcam
- **Press 'q'**: Stop recording early (webcam mode)

### Configuration

Edit `config.py` to customize:

```python
INPUT_SOURCE = None  # Set to video path or None for webcam
CAMERA_ID = 0        # Webcam device ID
WIDTH = 640          # Video width
HEIGHT = 480         # Video height
SHOW_SKELETON = False  # Display pose skeleton overlay
```

## Output Files

### Final Output
- **`final_performance.mp4`** - Final video with synthesized audio (saved in project root)

### Temporary Files (auto-deleted)
- **`temp_video.avi`** - Temporary video file (deleted after processing)
- **`temp_audio.wav`** - Temporary audio file (deleted after processing)

## Project Structure

```
video_sonification/
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── pose_landmarker_full.task  # MediaPipe pose model
├── engine/
│   ├── audio.py           # Audio synthesis engine
│   ├── visuals.py         # Visual processing (segmentation, flow)
│   ├── pose.py            # Pose detection
│   └── data.py            # Data collection
└── final_performance.mp4  # Output file (generated)
```

## How It Works

1. **Video Input**: Captures from webcam or video file
2. **Motion Analysis**: Extracts optical flow and pose landmarks
3. **Spectral Generation**: Converts motion patterns to frequency spectra
4. **Audio Synthesis**: Generates audio using Griffin-Lim and mode-based effects
5. **Video Merge**: Combines processed video with synthesized audio

## Audio Modes

The system automatically selects synthesis modes based on motion:
- **FM**: Fast motion with torso activity
- **Rhythmic**: High motion variance
- **Granular**: Wide arm spreads
- **Ambient**: Low motion, stillness
