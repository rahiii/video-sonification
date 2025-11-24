# engine/visuals.py
import cv2
import numpy as np
import mediapipe as mp
import config

class VisualEngine:
    def __init__(self):
        self.w = config.WIDTH
        self.h = config.HEIGHT
        self.mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        try:
            self.dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)
        except Exception:
            self.dis = None

        self.prev_gray = None
        self.canvas = np.zeros((self.h, self.w, 2), dtype=np.float32)
        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.pi_180 = 180.0 / np.pi
        self.hsv_scale = self.pi_180 / 2

    def process(self, frame, mirror_mode=True):
        frame = cv2.resize(frame, (self.w, self.h))

        if mirror_mode:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Segmentation ---
        res = self.mp_seg.process(rgb)
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        cx, cy = 0.5, 0.5

        if res.segmentation_mask is not None:
            bin_mask = (res.segmentation_mask > 0.5).astype(np.uint8)
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, self.morph_kernel)
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, self.morph_kernel)
            mask = cv2.GaussianBlur(bin_mask.astype(np.float32), (5, 5), 0)

            M = cv2.moments(bin_mask)
            if M["m00"] != 0:
                cx = (M["m10"] / M["m00"]) / self.w
                cy = (M["m01"] / M["m00"]) / self.h

        if self.prev_gray is None:
            self.prev_gray = gray
            c_mag = np.zeros_like(mask)
            c_ang = np.zeros_like(mask)
            return frame, np.zeros((self.h, self.w)), c_ang, c_mag, cx, cy
        if self.dis:
            flow = self.dis.calc(self.prev_gray, gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

        flow[..., 0] *= mask
        flow[..., 1] *= mask

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = np.maximum(0, mag - 0.5) * config.FLOW_SENSITIVITY
        x_flow, y_flow = cv2.polarToCart(mag, ang)

        self.canvas = cv2.addWeighted(
            np.dstack((x_flow, y_flow)), config.TRAIL_SPEED,
            self.canvas, config.TRAIL_DECAY, 0
        )

        c_mag, c_ang = cv2.cartToPolar(self.canvas[..., 0], self.canvas[..., 1])
        hsv = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        hsv[..., 0] = (c_ang * self.hsv_scale).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.clip(c_mag * 10, 0, 255).astype(np.uint8)

        final = cv2.add(
            (frame * 0.6).astype(np.uint8),
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        )

        self.prev_gray = gray
        return final, mag, c_ang, c_mag, cx, cy
