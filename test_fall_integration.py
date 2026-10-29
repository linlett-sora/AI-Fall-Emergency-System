"""Camera integration test for existing PoseDetector and FallDetector.
Run from the project root: python test_fall_integration.py
"""

import cv2
import winsound
import threading

from detection.pose_detector import PoseDetector
from detection.fall_detector import FallDetector, classify_posture
from detection.pose_processor_v2 import process_mediapipe_pose


def play_test_alarm():
    """Play a short alarm sound without freezing the camera."""
    try:
        winsound.Beep(2000, 400)
    except RuntimeError as error:
        print(f"Alarm error: {error}")


def main():

    # 1. Initialize detectors
    pose_detector = PoseDetector()
    fall_detector = FallDetector()

    # 2. Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        pose_detector.close()
        print("Could not open the camera")
        return

    fall_detected = False

    try:
        while True:

            # 3. Read camera frame
            ok, frame = cap.read()

            if not ok:
                print("Could not read camera frame")
                break

            height, width = frame.shape[:2]

            # 4. Detect body landmarks
            landmarks = pose_detector.detect_pose(frame)

            # 5. Fall detection
            # Pass every frame, including missing poses
            new_fall = fall_detector.update(
                landmarks,
                image_size=(width, height)
            )

            # 6. Posture classification
            posture = classify_posture(
                landmarks,
                (width, height)
            )

            # 7. Trigger alarm only when a new fall is detected
            if new_fall:

                fall_detected = True

                print("FALL DETECTED")

                # Play alarm in a separate thread
                threading.Thread(
                    target=play_test_alarm,
                    daemon=True
                ).start()

            # 8. Draw skeleton
            if landmarks is not None:
                pose_detector.draw_pose(
                    frame,
                    landmarks
                )

            # 9. Display detection result
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

            # 10. Display controls
            cv2.putText(
                frame,
                "Q: quit  R: reset",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # 11. Show camera
            cv2.imshow(
                "Pose + Fall Integration Test",
                frame
            )

            # 12. Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key == ord("q"):
                break

            # Reset
            if key == ord("r"):

                fall_detector.reset()

                pose_detector.previous_points.clear()

                fall_detected = False

                print("Reset")

    finally:

        # 13. Release resources
        cap.release()

        cv2.destroyAllWindows()

        pose_detector.close()


if __name__ == "__main__":
    main()