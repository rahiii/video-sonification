import cv2
import os
import config
from engine.visuals import VisualEngine
from engine.pose import PoseEngine
from engine.data import DataCollector
from engine.audio import AudioEngine

def main():
    user_input = input("Enter video path (Enter for Webcam): ").strip()
    if user_input and os.path.exists(user_input):
        source = user_input
        is_video_file = True
    else:
        source = config.CAMERA_ID
        is_video_file = False

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_time = int(1000 / fps) if is_video_file else 1
    should_mirror = not is_video_file

    temp_video_path = "temp_video.avi"
    write_fps = fps if is_video_file else 30.0
    writer = cv2.VideoWriter(
        temp_video_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        write_fps,
        (config.WIDTH, config.HEIGHT)
    )

    visuals = VisualEngine()
    pose_tracker = PoseEngine()
    collector = DataCollector()
    audio_synth = AudioEngine()
    show_skeleton = config.SHOW_SKELETON

    frame_idx = 0
    fps_inv = 1.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        visual_frame, flow_mag, c_ang, c_mag, cx, cy = visuals.process(frame, mirror_mode=should_mirror)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = pose_tracker.process(rgb, frame_idx * fps_inv * 1000.0)

        collector.process(flow_mag, c_ang, c_mag, cx, cy, pose_result)

        final_output = pose_tracker.draw_overlay(visual_frame, pose_result) if show_skeleton else visual_frame

        cv2.imshow('Camera-as-Synth', final_output)
        writer.write(final_output)

        frame_idx += 1
        if cv2.waitKey(delay_time) == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if not collector.spectral_hist:
        return

    total_duration = frame_idx * fps_inv
    wav_path = audio_synth.generate(collector, total_duration)

    output_filename = "final_performance.mp4"
    audio_synth.merge_video(temp_video_path, wav_path, output_filename, total_duration)

    try:
        os.remove(temp_video_path)
        os.remove(wav_path)
    except:
        pass

if __name__ == "__main__":
    main()
