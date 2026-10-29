
import cv2
import pyrealsense2 as rs
import numpy as np
import winsound
import threading

from detection.pose_detector import PoseDetector
from detection.fall_detector import FallDetector, classify_posture

def play_alarm():
    try:
        winsound.Beep(2000, 400)
    except RuntimeError as error:
        print(f"Alarm error: {error}")

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30
    )

    pose_detector = None
    started = False

    try:
        pipeline.start(config)
        started = True

        pose_detector = PoseDetector()
        fall_detector = FallDetector()
        fall_detected = False

        print("RealSense + Pose + Fall test started")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(
                color_frame.get_data()
            ).copy()

            height, width = frame.shape[:2]

            landmarks = pose_detector.detect_pose(frame)

            new_fall = fall_detector.update(
                landmarks,
                image_size=(width, height)
            )

            posture = classify_posture(
                landmarks,
                (width, height)
            )

            if new_fall:
                fall_detected = True
                print("FALL DETECTED")

                threading.Thread(
                    target=play_alarm,
                    daemon=True
                ).start()

            if landmarks is not None:
                pose_detector.draw_pose(
                    frame,
                    landmarks
                )

            if fall_detected:
                label = "FALL DETECTED"
                color = (0, 0, 255)
            else:
                label = f"Posture: {posture}"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                "Q: quit  R: reset",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "RealSense Pose + Fall Test",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                fall_detector.reset()
                pose_detector.previous_points.clear()
                fall_detected = False
                print("Reset")

    finally:
        cv2.destroyAllWindows()

        if pose_detector is not None:
            pose_detector.close()

        if started:
            pipeline.stop()

if __name__ == "__main__":
    main()