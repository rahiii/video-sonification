# engine/data.py
import numpy as np
import math
import config

class DataCollector:
    def __init__(self):
        self.motion_hist = []
        self.mod_hist = []
        self.pose_hist = []
        self.spectral_hist = []
        self.current_energy = 0.0
        self.current_spread = 0.0
        self.current_gesture = None

    def process(self, mag, c_ang, c_mag, cx, cy, pose_result):
        if mag is not None:
            avg_speed = np.clip(np.mean(mag) / 10.0, 0, 1)
        else:
            avg_speed = 0.0

        self.current_energy = float(avg_speed)
        self.motion_hist.append(float(avg_speed))
        self.mod_hist.append((float(cx), float(cy), float(avg_speed)))
        flat_ang = c_ang.flatten()
        flat_mag = c_mag.flatten()
        act = flat_mag > 0.1

        S_frame = [np.zeros(config.N_BINS, dtype=np.float32) for _ in range(3)]
        if np.any(act):
            hist, _ = np.histogram(
                flat_ang[act],
                bins=config.N_BINS,
                range=(0, 2 * np.pi),
                weights=flat_mag[act]
            )
            hist = hist.astype(np.float32)
            S_frame[0] = hist * (1.0 - avg_speed)
            S_frame[1] = hist
            S_frame[2] = hist * avg_speed

        self.spectral_hist.append(S_frame)
        frame_feats = []
        if pose_result and pose_result.pose_landmarks:
            for lm in pose_result.pose_landmarks:
                try:
                    lw, rw, ls, rs = lm[15], lm[16], lm[11], lm[12]
                    angle = math.atan2(rs.y - ls.y, rs.x - ls.x)
                    center_x = (ls.x + rs.x) * 0.5
                    frame_feats.append((lw.x, lw.y, rw.x, rw.y, angle, center_x))
                except (IndexError, AttributeError):
                    continue

        self.pose_hist.append(frame_feats)
        if frame_feats:
            feats_arr = np.array(frame_feats)
            self.current_spread = float(np.mean(np.abs(feats_arr[:, 2] - feats_arr[:, 0])))
        else:
            self.current_spread = 0.0

        return self.current_energy, self.current_spread
