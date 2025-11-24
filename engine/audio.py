# engine/audio.py
import numpy as np
import librosa
import soundfile as sf
import scipy.ndimage
import os
import config
from moviepy.editor import VideoFileClip, AudioFileClip
import moviepy.video.fx.all as vfx

class AudioEngine:
    def __init__(self):
        self.sr = config.SR
        self.n_bins = config.N_BINS
        self.n_fft = config.N_FFT
        self.sr_inv = 1.0 / config.SR
        self.scale_len = len(config.CIRCLE_OF_FIFTHS)


    def _pick_scale(self, cx):
        idx = int(cx * self.scale_len)
        return config.CIRCLE_OF_FIFTHS[max(0, min(idx, self.scale_len - 1))]

    def _create_dynamic_mask(self, active_scale, richness=1.0):
        mask = np.zeros(self.n_bins, dtype=np.float32)
        num_octaves = int(2 + (richness * 3))
        freq_scale = self.n_fft * self.sr_inv

        # Pre-compute all frequencies
        for octave in range(num_octaves):
            mult = 2 ** octave
            for f in active_scale:
                bin_idx = int(f * mult * freq_scale)
                if 0 <= bin_idx < self.n_bins:
                    start = max(0, bin_idx - 2)
                    end = min(self.n_bins, bin_idx + 3)
                    mask[start:end] = 1.0
        return mask

    def _add_reverb(self, audio, delay_s=0.3, decay=0.5):
        delay_samples = int(self.sr * delay_s)
        if delay_samples >= len(audio):
            return audio
        output = audio.copy()
        output[delay_samples:] += output[:-delay_samples] * decay
        return output

    def _granular_process(self, audio, grain_ms=50, overlap=0.5):
        n = len(audio)
        grain_len = max(10, int(self.sr * grain_ms * 0.001))
        step = max(1, int(grain_len * (1.0 - overlap)))
        
        if grain_len >= n:
            return audio

        # Extract grains
        grain_starts = np.arange(0, n - grain_len, step)
        grains = np.array([audio[i:i+grain_len] for i in grain_starts])
        window = np.hanning(grain_len)
        grains *= window

        # Shuffle and overlap-add
        np.random.shuffle(grains)
        out = np.zeros(n, dtype=audio.dtype)
        for i, grain in enumerate(grains):
            pos = i * step
            end = min(pos + grain_len, n)
            out[pos:end] += grain[:end-pos]

        max_val = np.max(np.abs(out))
        return (out / (max_val + 1e-6)) * 0.9 if max_val > 0 else out

    def _rhythmic_gate(self, audio, motion_curve):
        m_smooth = scipy.ndimage.gaussian_filter1d(
            motion_curve, sigma=max(1, len(motion_curve) // 200)
        )
        threshold = np.mean(m_smooth) + 0.5 * np.std(m_smooth)
        peaks = np.where(
            (m_smooth[1:-1] > m_smooth[:-2]) &
            (m_smooth[1:-1] > m_smooth[2:]) &
            (m_smooth[1:-1] > threshold)
        )[0] + 1

        env = np.full(len(audio), 0.15, dtype=np.float32)
        width = int(0.06 * self.sr)
        motion_len = len(motion_curve)
        audio_len = len(audio)
        
        for p in peaks:
            center = int(p * audio_len / motion_len)
            start = max(0, center - width)
            end = min(audio_len, center + width)
            indices = np.arange(start, end)
            env[indices] = np.maximum(env[indices], 1.0 - np.abs(indices - center) / width)
        
        return audio * env

    def _fm_synth(self, duration, scale, motion_curve, torso_activity, spread):
        n = int(duration * self.sr)
        t = np.linspace(0.0, duration, n, endpoint=False)
        base_freq = scale[len(scale)//2] if scale else 220.0
        two_pi = 2.0 * np.pi

        m = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(motion_curve)),
            motion_curve
        )

        index = 2.0 * m * (1.0 + min(torso_activity * 50, 4.0)) * (1.0 + spread)
        mod = np.sin(two_pi * (base_freq * 2) * t)
        y = np.sin(two_pi * base_freq * t + index * mod) * np.hanning(n)

        max_val = np.max(np.abs(y))
        return (y / (max_val + 1e-6)) * 0.9 if max_val > 0 else y
    
    def _harmonic_arpeggios(self, duration, scale, motion_curve):
        """Harmonic arpeggios for raise_arms gesture"""
        n = int(duration * self.sr)
        t = np.linspace(0.0, duration, n, endpoint=False)
        two_pi = 2.0 * np.pi
        scale = scale or config.CIRCLE_OF_FIFTHS[3]
        
        m = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(motion_curve)), motion_curve)
        arp_speed = 2.0 + m * 3.0
        phase = np.cumsum(arp_speed * 0.01) % (len(scale) * 2)
        freqs = np.array(scale)
        indices = (phase / 2).astype(int) % len(freqs)
        current_freqs = freqs[indices]
        
        y = np.array([np.sin(two_pi * current_freqs[i] * t[i]) for i in range(n)], dtype=np.float32)
        envelope = np.hanning(n) * (0.3 + 0.7 * m)
        y *= envelope
        
        max_val = np.max(np.abs(y))
        return (y / (max_val + 1e-6)) * 0.9 if max_val > 0 else y
    
    def _doppler_fm_synth(self, duration, scale, motion_curve, spin_intensity):
        """Doppler FM for spin gesture"""
        n = int(duration * self.sr)
        t = np.linspace(0.0, duration, n, endpoint=False)
        base_freq = scale[len(scale)//2] if scale else 220.0
        two_pi = 2.0 * np.pi
        
        m = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(motion_curve)), motion_curve)
        doppler_freq = base_freq * (1.0 + spin_intensity * np.sin(two_pi * 0.5 * t))
        index = 2.0 * m * (1.0 + spin_intensity * 2.0)
        mod = np.sin(two_pi * (doppler_freq * 2) * t)
        y = np.sin(two_pi * doppler_freq * t + index * mod) * np.hanning(n)
        
        max_val = np.max(np.abs(y))
        return (y / (max_val + 1e-6)) * 0.9 if max_val > 0 else y

    def _classify_mode(self, motion_hist, pose_hist, gesture=None):
        # Gesture-based mode selection
        if gesture:
            gesture_map = {
                "wave": "granular",
                "spin": "doppler_fm",
                "raise_arms": "harmonic",
                "stillness": "ambient",
                "jump": "rhythmic"
            }
            if gesture in gesture_map:
                return gesture_map[gesture]
        
        if not motion_hist:
            return "ambient"

        m = np.array(motion_hist, dtype=np.float32)
        high_ratio = np.mean(m > 0.6)
        low_ratio = np.mean(m < 0.2)
        var_m = np.var(m)
        mean_m = np.mean(m)

        # Vectorized spread calculation
        all_spreads = []
        torso_deltas = []
        prev_torsos = None

        for frame in pose_hist:
            if not frame:
                continue
                
            # Extract spreads
            frame_arr = np.array(frame)
            spreads = np.abs(np.clip(frame_arr[:, 2], 0, 1) - np.clip(frame_arr[:, 0], 0, 1))
            all_spreads.extend(spreads.tolist())

            # Torso tracking
            current_torsos = sorted([(f[5], f[4]) for f in frame], key=lambda x: x[0])
            if prev_torsos:
                count = min(len(prev_torsos), len(current_torsos))
                if count > 0:
                    deltas = np.abs(np.array([current_torsos[i][1] for i in range(count)]) - 
                                   np.array([prev_torsos[i][1] for i in range(count)]))
                    torso_deltas.append(float(np.mean(deltas)))
            prev_torsos = current_torsos

        torso_act = np.mean(torso_deltas) if torso_deltas else 0.0
        avg_spread = np.mean(all_spreads) if all_spreads else 0.3

        if high_ratio > 0.4 and var_m > 0.02:
            return "fm" if (torso_act > 0.03 and avg_spread > 0.4) else "rhythmic"
        if avg_spread > 0.5:
            return "granular"
        if mean_m < 0.2 and low_ratio > 0.5:
            return "ambient"

        return "ambient"

    def generate(self, collector, total_time):

        spectral_hist = collector.spectral_hist
        mod_hist = collector.mod_hist
        motion_hist = collector.motion_hist
        pose_hist = collector.pose_hist

        if not spectral_hist:
            raise RuntimeError("No spectral data collected for audio synthesis.")

        # 1. Build smoothed spectral layers
        spectral_array = np.array([s for s in spectral_hist]).transpose(1, 2, 0)  # (3, N_BINS, frames)
        sigmas = [(2, 2), (1, 2), (0.5, 1)]
        S_layers = [
            scipy.ndimage.gaussian_filter(spectral_array[i], sigma=sig)
            for i, sig in enumerate(sigmas)
        ]
        S_low, S_mid, S_high = S_layers

        # 2. Per-frame masking & shaping
        n_frames = S_low.shape[1]
        factors_base = np.array([1.2, 1.0, 0.3])
        
        for t in range(n_frames):
            cx, cy, speed = mod_hist[t]
            mask = self._create_dynamic_mask(self._pick_scale(cx), richness=speed)
            factors = factors_base + np.array([0, 0, speed])

            for S, factor in zip([S_low, S_mid, S_high], factors):
                S[:, t] *= mask * factor
                cutoff = int(self.n_bins * (1.0 - cy * 0.8))
                if cutoff < self.n_bins:
                    S[cutoff:, t] = 0

            if t < len(pose_hist) and pose_hist[t]:
                feats = np.array(pose_hist[t])
                h_avg = np.mean((2 - feats[:, 1] - feats[:, 3]) / 2)
                s_avg = np.mean(np.abs(feats[:, 2] - feats[:, 0]))
                S_mid[:, t] *= (0.9 + 0.3 * s_avg)
                S_low[:, t] *= (1.1 - 0.5 * h_avg)

        # 3. Collapse to final spectrogram, Griffin-Lim, stretch to time
        S_total = np.log1p(sum([S_low, S_mid, S_high]) + 1e-6)
        if S_total.max() > 0:
            S_total = S_total / S_total.max() * 60.0

        raw = librosa.griffinlim(S_total, n_fft=self.n_fft, hop_length=config.HOP_LEN)
        curr_dur = len(raw) * self.sr_inv
        audio = librosa.effects.time_stretch(raw, rate=curr_dur / total_time) if total_time > 0 else raw

        # 4. Mode selection & additional synthesis
        gesture = getattr(collector, 'current_gesture', None)
        mode = self._classify_mode(motion_hist, pose_hist, gesture)

        m_interp = np.interp(
            np.linspace(0, 1, len(audio)),
            np.linspace(0, 1, len(motion_hist)),
            np.clip(motion_hist, 0, 1)
        )

        # Recalculate torso stats for FM parameters (vectorized)
        torso_deltas = []
        prev = None
        for frame in pose_hist:
            if not frame:
                continue
            cur = sorted([(f[5], f[4]) for f in frame], key=lambda x: x[0])
            if prev and len(prev) == len(cur):
                deltas = np.abs(np.array([c[1] for c in cur]) - np.array([p[1] for p in prev]))
                torso_deltas.append(float(np.mean(deltas)))
            prev = cur
        torso_act = np.mean(torso_deltas) if torso_deltas else 0.0

        # Vectorized spread calculation
        if pose_hist:
            spreads = [abs(np.clip(f[2], 0, 1) - np.clip(f[0], 0, 1)) for frame in pose_hist for f in frame]
            avg_spread = np.mean(spreads) if spreads else 0.3
        else:
            avg_spread = 0.3
            
        mean_cx = np.mean([m[0] for m in mod_hist]) if mod_hist else 0.5

        if mode == "fm":
            fm = self._fm_synth(len(audio) * self.sr_inv,
                                self._pick_scale(mean_cx),
                                m_interp,
                                torso_act,
                                avg_spread)
            fm = fm[:len(audio)]
            mixed = audio * 0.6 + fm * 0.6
            max_mixed = np.max(np.abs(mixed))
            final = self._add_reverb((mixed / (max_mixed + 1e-6)) * 0.9)

        elif mode == "rhythmic":
            final = self._add_reverb(self._rhythmic_gate(audio, m_interp), delay_s=0.2)

        elif mode == "granular":
            final = self._add_reverb(self._granular_process(audio), delay_s=0.4)
        
        elif mode == "harmonic":
            harmonic = self._harmonic_arpeggios(len(audio) * self.sr_inv, self._pick_scale(mean_cx), m_interp)
            harmonic = harmonic[:len(audio)]
            mixed = audio * 0.4 + harmonic * 0.8
            max_mixed = np.max(np.abs(mixed))
            final = self._add_reverb((mixed / (max_mixed + 1e-6)) * 0.9, delay_s=0.3)
        
        elif mode == "doppler_fm":
            spin_intensity = min(torso_act * 10, 1.0)
            doppler = self._doppler_fm_synth(len(audio) * self.sr_inv, self._pick_scale(mean_cx), m_interp, spin_intensity)
            doppler = doppler[:len(audio)]
            mixed = audio * 0.5 + doppler * 0.7
            max_mixed = np.max(np.abs(mixed))
            final = self._add_reverb((mixed / (max_mixed + 1e-6)) * 0.9, delay_s=0.25)

        else:
            final = self._add_reverb(audio, delay_s=0.5)

        sf.write("temp_audio.wav", final, self.sr)
        return "temp_audio.wav"

    def merge_video(self, video_path, audio_path, output_path, total_time):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        clip_v = VideoFileClip(video_path).fx(vfx.speedx, final_duration=total_time)
        clip_a = AudioFileClip(audio_path)
        clip_v.set_audio(clip_a).write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
