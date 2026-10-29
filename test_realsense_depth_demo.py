
import cv2
import numpy as np
import pyrealsense2 as rs
import winsound
import threading
import time
import math

from detection.pose_detector import PoseDetector
from detection.fall_detector import FallDetector, classify_posture

# ==========================================
# CONFIGURATION
# ==========================================

CHECKING_SECONDS = 5.0
ASSISTANCE_SECONDS = 10.0

MIN_VISIBILITY = 0.5
MAX_DEPTH_METERS = 5.0

# Experimental supported-lying thresholds
SUPPORTED_BODY_ANGLE = 25.0
LOW_BODY_IMAGE_Y = 0.58

ENABLE_SUPPORTED_LYING = True

WINDOW_NAME = "RealSense Depth Festival Demo"

# ==========================================
# ALARM
# ==========================================

def play_alarm():
    try:
        for _ in range(3):
            winsound.Beep(1800, 300)
            time.sleep(0.15)
    except Exception as error:
        print("Alarm error:", error)

def start_alarm():
    threading.Thread(
        target=play_alarm,
        daemon=True
    ).start()

# ==========================================
# LANDMARK UTILITIES
# ==========================================

def valid_point(landmarks, index):
    if landmarks is None:
        return None

    if len(landmarks) <= index:
        return None

    point = landmarks[index]

    visibility = getattr(point, "visibility", 1.0)

    if visibility < MIN_VISIBILITY:
        return None

    if not (
        math.isfinite(point.x)
        and math.isfinite(point.y)
    ):
        return None

    if not (
        0 <= point.x <= 1
        and 0 <= point.y <= 1
    ):
        return None

    return point

def midpoint(landmarks, first, second):
    p1 = valid_point(landmarks, first)
    p2 = valid_point(landmarks, second)

    if p1 is None and p2 is None:
        return None

    if p1 is None:
        return (p2.x, p2.y)

    if p2 is None:
        return (p1.x, p1.y)

    return (
        (p1.x + p2.x) / 2.0,
        (p1.y + p2.y) / 2.0
    )

# ==========================================
# DEPTH UTILITIES
# ==========================================

def get_depth(depth_frame, point, width, height):
    if point is None:
        return None

    x = int(point[0] * width)
    y = int(point[1] * height)

    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))

    distance = depth_frame.get_distance(x, y)

    if not math.isfinite(distance):
        return None

    if distance <= 0:
        return None

    if distance > MAX_DEPTH_METERS:
        return None

    return distance

# ==========================================
# POSTURE ESTIMATION
# ==========================================

def estimate_posture(
    landmarks,
    depth_frame,
    width,
    height
):
    basic_posture = classify_posture(
        landmarks,
        (width, height)
    )

    if landmarks is None:
        return "unknown", None

    shoulders = midpoint(
        landmarks, 11, 12
    )

    hips = midpoint(
        landmarks, 23, 24
    )

    ankles = midpoint(
        landmarks, 27, 28
    )

    if shoulders is None or hips is None:
        return basic_posture, None

    shoulder_depth = get_depth(
        depth_frame,
        shoulders,
        width,
        height
    )

    hip_depth = get_depth(
        depth_frame,
        hips,
        width,
        height
    )

    ankle_depth = get_depth(
        depth_frame,
        ankles,
        width,
        height
    )

    dx = (
        shoulders[0] - hips[0]
    ) * width

    dy = (
        shoulders[1] - hips[1]
    ) * height

    body_angle = math.degrees(
        math.atan2(
            abs(dx),
            abs(dy)
        )
    )

    depth_info = {
        "shoulder": shoulder_depth,
        "hip": hip_depth,
        "ankle": ankle_depth,
        "body_angle": body_angle
    }

    # Existing fall_detector classification
    if basic_posture == "lying":
        return "lying", depth_info

    # Experimental rule:
    # A person leaning on one arm while lying.
    #
    # This does not prove that the person is
    # physically on the floor.
    #
    # Camera angle affects the result.

    if ENABLE_SUPPORTED_LYING:
        has_depth = (
            shoulder_depth is not None
            and hip_depth is not None
            and ankle_depth is not None
        )

        lower_body_visible = (
            ankles is not None
            and ankles[1] >= hips[1]
        )

        body_low_in_image = (
            hips[1] >= LOW_BODY_IMAGE_Y
        )

        body_is_tilted = (
            body_angle >= SUPPORTED_BODY_ANGLE
        )

        if (
            has_depth
            and lower_body_visible
            and body_low_in_image
            and body_is_tilted
        ):
            return "supported_lying", depth_info

    return basic_posture, depth_info

# ==========================================
# MAIN
# ==========================================

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

    config.enable_stream(
        rs.stream.depth,
        640,
        480,
        rs.format.z16,
        30
    )

    align = rs.align(rs.stream.color)

    pose_detector = None
    fall_detector = None

    pipeline_started = False

    lying_start = None
    assistance_alerted = False

    fall_detected = False
    last_status = None

    try:
        pipeline.start(config)
        pipeline_started = True

        pose_detector = PoseDetector()
        fall_detector = FallDetector()

        print("================================")
        print("RealSense Depth Demo Started")
        print("Q = Quit")
        print("R = Reset")
        print("No database or email notifications")
        print("================================")

        while True:
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(
                frames
            )

            color_frame = (
                aligned_frames.get_color_frame()
            )

            depth_frame = (
                aligned_frames.get_depth_frame()
            )

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(
                color_frame.get_data()
            ).copy()

            height, width = frame.shape[:2]

            now = time.monotonic()

            # ----------------------------------
            # Pose detection
            # ----------------------------------

            landmarks = pose_detector.detect_pose(
                frame
            )

            # ----------------------------------
            # Existing fall detection
            # ----------------------------------

            new_fall = fall_detector.update(
                landmarks,
                image_size=(width, height)
            )

            if new_fall:
                fall_detected = True
                start_alarm()
                print("FALL DETECTED")

            # ----------------------------------
            # Depth + posture estimation
            # ----------------------------------

            posture, depth_info = estimate_posture(
                landmarks,
                depth_frame,
                width,
                height
            )

            # ----------------------------------
            # Lying timer
            # ----------------------------------

            lying_state = posture in (
                "lying",
                "supported_lying"
            )

            if lying_state:
                if lying_start is None:
                    lying_start = now

                lying_duration = (
                    now - lying_start
                )

            else:
                lying_start = None
                lying_duration = 0.0

                if posture in (
                    "standing",
                    "sitting"
                ):
                    assistance_alerted = False

            # ----------------------------------
            # Display status
            # ----------------------------------

            status = f"Posture: {posture}"

            color = (0, 255, 0)

            if lying_state:
                if (
                    lying_duration
                    >= ASSISTANCE_SECONDS
                ):
                    status = "ASSISTANCE NEEDED"
                    color = (0, 0, 255)

                    if not assistance_alerted:
                        assistance_alerted = True
                        start_alarm()

                        print(
                            "DEMO ALERT: "
                            "Assistance needed"
                        )

                elif (
                    lying_duration
                    >= CHECKING_SECONDS
                ):
                    status = "CHECKING..."
                    color = (0, 165, 255)

                else:
                    status = (
                        f"Posture: {posture}"
                    )

                    color = (0, 255, 255)

            if fall_detected:
                status = "FALL DETECTED"
                color = (0, 0, 255)

            # ----------------------------------
            # Draw skeleton
            # ----------------------------------

            if landmarks is not None:
                pose_detector.draw_pose(
                    frame,
                    landmarks
                )

            # ----------------------------------
            # Draw text
            # ----------------------------------

            cv2.putText(
                frame,
                status,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Lying: {lying_duration:.1f}s",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if depth_info is not None:
                hip_depth = depth_info["hip"]

                if hip_depth is not None:
                    cv2.putText(
                        frame,
                        f"Hip distance: {hip_depth:.2f}m",
                        (15, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )

                body_angle = depth_info["body_angle"]

                cv2.putText(
                    frame,
                    f"Torso angle: {body_angle:.1f}",
                    (15, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "Q: Quit | R: Reset",
                (15, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "DEPTH DEMO - EXPERIMENTAL",
                (15, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # ----------------------------------
            # Terminal status
            # ----------------------------------

            if status != last_status:
                print(
                    "Pose:",
                    landmarks is not None,
                    "| Posture:",
                    posture,
                    "| Status:",
                    status
                )

                last_status = status

            # ----------------------------------
            # Keyboard
            # ----------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                fall_detector.reset()

                if hasattr(
                    pose_detector,
                    "previous_landmarks"
                ):
                    pose_detector.previous_landmarks.clear()

                if hasattr(
                    pose_detector,
                    "previous_points"
                ):
                    pose_detector.previous_points.clear()

                lying_start = None
                assistance_alerted = False
                fall_detected = False
                last_status = None

                print("Demo reset")

    except Exception as error:
        print("ERROR:", error)
        raise

    finally:
        cv2.destroyAllWindows()

        if pose_detector is not None:
            pose_detector.close()

        if pipeline_started:
            pipeline.stop()

        print("Demo stopped")

if __name__ == "__main__":
    main()