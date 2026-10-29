
import cv2
import numpy as np
import pyrealsense2 as rs
import winsound
import threading
import time

from detection.pose_detector import PoseDetector
from detection.fall_detector import FallDetector, classify_posture

# Festival demonstration settings
LYING_CHECK_SECONDS = 5
ASSISTANCE_SECONDS = 10

# Do not send real emergency notifications
DEMO_MODE = True

def play_alarm():
    try:
        for _ in range(3):
            winsound.Beep(1800, 300)
            time.sleep(0.15)
    except RuntimeError as error:
        print("Alarm error:", error)

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
    pipeline_started = False

    lying_start = None
    assistance_alerted = False
    fall_detected = False

    last_posture = None
    last_message = None

    try:
        pipeline.start(config)
        pipeline_started = True

        pose_detector = PoseDetector()
        fall_detector = FallDetector()

        print("Safe Festival Demo Started")
        print("Q: Quit | R: Reset")
        print("No real emergency messages will be sent.")

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            frame = np.asanyarray(
                color_frame.get_data()
            ).copy()

            height, width = frame.shape[:2]
            current_time = time.monotonic()

            # STEP 1: Pose Detection
            landmarks = pose_detector.detect_pose(frame)

            # STEP 2: Fall Detection
            new_fall = fall_detector.update(
                landmarks,
                image_size=(width, height)
            )

            # STEP 3: Posture Classification
            posture = classify_posture(
                landmarks,
                (width, height)
            )

            posture_text = str(posture).lower()

            # STEP 4: Lying Timer
            if posture_text == "lying":
                if lying_start is None:
                    lying_start = current_time
                    print("LYING started")

                lying_duration = (
                    current_time - lying_start
                )

            else:
                lying_start = None
                lying_duration = 0

                # Reset assistance state only after
                # a confirmed non-lying posture.
                if posture_text in (
                    "standing",
                    "sitting"
                ):
                    assistance_alerted = False

            # STEP 5: Decide Screen Status
            status = f"Posture: {posture}"
            color = (0, 255, 0)

            if lying_start is not None:
                if lying_duration >= ASSISTANCE_SECONDS:
                    status = "ASSISTANCE NEEDED"
                    color = (0, 0, 255)

                    if not assistance_alerted:
                        assistance_alerted = True

                        print(
                            "DEMO ALERT: Assistance needed"
                        )

                        threading.Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()

                elif lying_duration >= LYING_CHECK_SECONDS:
                    status = "CHECKING..."
                    color = (0, 165, 255)

                else:
                    status = "LYING"
                    color = (0, 255, 255)

            # STEP 6: Real FallDetector Result
            if new_fall:
                fall_detected = True

                print("FALL DETECTED")

                threading.Thread(
                    target=play_alarm,
                    daemon=True
                ).start()

            if fall_detected:
                status = "FALL DETECTED"
                color = (0, 0, 255)

            # STEP 7: Draw Pose
            if landmarks is not None:
                pose_detector.draw_pose(
                    frame,
                    landmarks
                )

            # STEP 8: Show Status
            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            # STEP 9: Show Timer
            if lying_start is not None:
                timer_text = (
                    f"Lying: {lying_duration:.1f}s"
                )
            else:
                timer_text = "Lying: 0.0s"

            cv2.putText(
                frame,
                timer_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # STEP 10: Show Instructions
            cv2.putText(
                frame,
                "Q: Quit | R: Reset",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "DEMO MODE",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # STEP 11: Debug Output
            if posture_text != last_posture:
                print(
                    "Pose:",
                    landmarks is not None,
                    "| Posture:",
                    posture
                )
                last_posture = posture_text

            if status != last_message:
                print("Status:", status)
                last_message = status

            # STEP 12: Display
            cv2.imshow(
                "RealSense Safe Festival Demo",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                fall_detector.reset()
                pose_detector.previous_landmarks.clear()
                pose_detector.previous_points.clear()

                lying_start = None
                assistance_alerted = False
                fall_detected = False
                last_posture = None
                last_message = None

                print("System Reset")

    finally:
        cv2.destroyAllWindows()

        if pose_detector is not None:
            pose_detector.close()

        if pipeline_started:
            pipeline.stop()

if __name__ == "__main__":
    main()