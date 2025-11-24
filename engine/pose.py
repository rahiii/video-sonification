# engine/pose.py
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import cv2
import config
import os

class PoseEngine:
    CONNECTIONS = [
        (11, 12), (11, 23), (12, 24), (23, 24),  # Torso box
        (11, 13), (13, 15),                      # Left arm
        (12, 14), (14, 16),                      # Right arm
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg + foot
        (24, 26), (26, 28), (28, 30), (30, 32)   # Right leg + foot
    ]
    RELEVANT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    
    def __init__(self):
        if not os.path.exists(config.POSE_MODEL_PATH):
            raise FileNotFoundError(f"Missing model: {config.POSE_MODEL_PATH}")

        base_opts = mp_python.BaseOptions(model_asset_path=config.POSE_MODEL_PATH)
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=config.MAX_PEOPLE,
            min_pose_detection_confidence=config.DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.DETECTION_CONFIDENCE,
            min_tracking_confidence=config.TRACKING_CONFIDENCE
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)

    def process(self, rgb_frame, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))
        return result

    def draw_overlay(self, frame, pose_result):
        if not pose_result.pose_landmarks:
            return frame

        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        def get_xy(landmarks, idx):
            return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

        for landmarks in pose_result.pose_landmarks:
            for start, end in self.CONNECTIONS:
                try:
                    pt1 = get_xy(landmarks, start)
                    pt2 = get_xy(landmarks, end)
                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 2)
                except (IndexError, AttributeError):
                    pass

            for idx in self.RELEVANT_INDICES:
                try:
                    pt = get_xy(landmarks, idx)
                    cv2.circle(annotated_frame, pt, 5, (0, 0, 255), -1)
                except (IndexError, AttributeError):
                    pass

        return annotated_frame
