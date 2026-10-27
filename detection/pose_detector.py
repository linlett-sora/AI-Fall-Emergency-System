import cv2
import mediapipe as mp
import time
from pathlib import Path

<<<<<<< HEAD
from detection.pose_processor_v2 import process_mediapipe_pose
=======

>>>>>>> main
class PoseDetector:

    def __init__(self):

        # 1. Load MediaPipe pose model
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "pose_landmarker_lite.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}"
            )

        # 2. Configure Pose Landmarker
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.detector = (
            mp.tasks.vision.PoseLandmarker
            .create_from_options(options)
        )

        self.timestamp_ms = 0

        # Minimum visibility for drawing
        self.visibility_threshold = 0.5

        # 3. Body skeleton connections
        self.connections = [

            # Shoulders
            (11, 12),

            # Left arm
            (11, 13),
            (13, 15),

            # Right arm
            (12, 14),
            (14, 16),

            # Torso
            (11, 23),
            (12, 24),
            (23, 24),

            # Left leg
            (23, 25),
            (25, 27),

            # Right leg
            (24, 26),
            (26, 28),

            # Feet
            (27, 29),
            (29, 31),
            (28, 30),
            (30, 32)
        ]

    # 4. Detect pose
    def detect_pose(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb_frame = rgb_frame.copy()

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Video timestamp must increase
        self.timestamp_ms = max(
            self.timestamp_ms + 1,
            int(time.monotonic() * 1000)
        )

        results = self.detector.detect_for_video(
            image,
            self.timestamp_ms
        )

        # No pose detected
        if not results.pose_landmarks:
            return None

        # Return the first detected pose
        landmarks = results.pose_landmarks[0]

        return landmarks

    # 5. Draw skeleton
    def draw_pose(self, frame, landmarks):

        if landmarks is None:
            return frame

        height, width = frame.shape[:2]

        points = {}

        for index, landmark in enumerate(landmarks):

            # Ignore landmarks with low visibility
            if (
                landmark.visibility
                < self.visibility_threshold
            ):
                continue

            # Ignore invalid coordinates
            if not (
                0.0 <= landmark.x <= 1.0
                and 0.0 <= landmark.y <= 1.0
            ):
                continue

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points[index] = (x, y)

        # Draw skeleton connections
        for start, end in self.connections:

            if start in points and end in points:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

        # Draw visible landmarks
        for point in points.values():

            cv2.circle(
                frame,
                point,
                4,
                (0, 255, 0),
                -1
            )

        return frame

    # 6. Get body coordinates for Person 1B
    def get_body_coordinates(self, landmarks):

        if landmarks is None:
            return None

        body_parts = {

            "left_shoulder": 11,
            "right_shoulder": 12,

            "left_hip": 23,
            "right_hip": 24,

            "left_knee": 25,
            "right_knee": 26,

            "left_ankle": 27,
            "right_ankle": 28
        }

        coordinates = {}

        for name, index in body_parts.items():

            landmark = landmarks[index]

            coordinates[name] = {

                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,

                "visibility": landmark.visibility
            }

        return coordinates

    # 7. Close detector
    def close(self):
        self.detector.close()


# 8. Test camera
if __name__ == "__main__":

    pose_detector = PoseDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Camera could not be opened.")

        pose_detector.close()

        raise SystemExit

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print("Could not read camera frame.")
                break

            # Detect pose
            landmarks = pose_detector.detect_pose(frame)

            if landmarks is not None:

                # Draw skeleton
                pose_detector.draw_pose(
                    frame,
                    landmarks
                )

                # Get body coordinates
                coordinates = (
                    pose_detector.get_body_coordinates(
                        landmarks
                    )
                )

                cv2.putText(
                    frame,
                    "POSE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NO POSE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # Display camera
            cv2.imshow(
                "Pose Detection",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()

        cv2.destroyAllWindows()

        pose_detector.close()