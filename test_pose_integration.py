import cv2

from detection.pose_detector import PoseDetector
from detection.pose_processor_v2 import process_mediapipe_pose

def main():
    detector = PoseDetector()
    camera = cv2.VideoCapture(0)

    try:
        if not camera.isOpened():
            print("Camera cannot be opened")
            return

        print("Camera started. Press Q to quit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read camera frame")
                break

            landmarks = detector.detect_pose(frame)

            if landmarks is not None:
                height, width = frame.shape[:2]

                features = process_mediapipe_pose(
                    landmarks,
                    frame_width=width,
                    frame_height=height
                )

                print("Pose Processing:", features)

                detector.draw_pose(frame, landmarks)

            cv2.imshow("1A + 1B Integration Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()