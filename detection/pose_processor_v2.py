import math


# ---------------------------------------
# 1. Calculate center
# ---------------------------------------

def get_center(left, right):
    x = (left[0] + right[0]) / 2
    y = (left[1] + right[1]) / 2

    return (x, y)


# ---------------------------------------
# 2. Convert normalized coordinates
#    to pixel coordinates
# ---------------------------------------

def to_pixel(point, frame_width, frame_height):
    return (
        point[0] * frame_width,
        point[1] * frame_height
    )


# ---------------------------------------
# 3. Calculate body angle
# ---------------------------------------

def calculate_body_angle(shoulder, hip):
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]

    angle = math.degrees(
        math.atan2(abs(dx), abs(dy))
    )

    return round(angle, 2)


# ---------------------------------------
# 4. Calculate joint angle
# ---------------------------------------

def calculate_joint_angle(a, b, c):
    angle1 = math.atan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle2 = math.atan2(
        c[1] - b[1],
        c[0] - b[0]
    )

    angle = abs(
        math.degrees(angle1 - angle2)
    )

    if angle > 180:
        angle = 360 - angle

    return round(angle, 2)


# ---------------------------------------
# 5. Calculate body height / width ratio
# ---------------------------------------

def calculate_body_ratio(
    points,
    frame_width,
    frame_height
):
    xs = [
        p[0] * frame_width
        for p in points
    ]

    ys = [
        p[1] * frame_height
        for p in points
    ]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if width == 0:
        return None

    return round(height / width, 2)


# ---------------------------------------
# 6. Process pose landmarks
# ---------------------------------------

def process_pose(
    landmarks,
    frame_width=640,
    frame_height=480
):

    # Check essential landmarks

    required = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip"
    ]

    missing = [
        key for key in required
        if landmarks.get(key) is None
    ]

    if missing:
        return {
            "shoulder_center": None,
            "hip_center": None,
            "body_angle": None,
            "left_knee_angle": None,
            "right_knee_angle": None,
            "body_ratio": None,
            "status": "insufficient_landmarks",
            "missing_landmarks": missing
        }

    # Get shoulder and hip landmarks

    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]

    left_hip = landmarks["left_hip"]
    right_hip = landmarks["right_hip"]

    # Calculate centers
    # Centers remain normalized coordinates

    shoulder = get_center(
        left_shoulder,
        right_shoulder
    )

    hip = get_center(
        left_hip,
        right_hip
    )

    # Convert centers to pixel coordinates

    shoulder_px = to_pixel(
        shoulder,
        frame_width,
        frame_height
    )

    hip_px = to_pixel(
        hip,
        frame_width,
        frame_height
    )

    # Calculate body angle

    body_angle = calculate_body_angle(
        shoulder_px,
        hip_px
    )

    left_knee_angle = None
    right_knee_angle = None
    body_ratio = None

    # Left knee

    if (
        landmarks.get("left_knee") is not None
        and landmarks.get("left_ankle") is not None
    ):

        left_knee_angle = calculate_joint_angle(
            to_pixel(
                left_hip,
                frame_width,
                frame_height
            ),
            to_pixel(
                landmarks["left_knee"],
                frame_width,
                frame_height
            ),
            to_pixel(
                landmarks["left_ankle"],
                frame_width,
                frame_height
            )
        )

    # Right knee

    if (
        landmarks.get("right_knee") is not None
        and landmarks.get("right_ankle") is not None
    ):

        right_knee_angle = calculate_joint_angle(
            to_pixel(
                right_hip,
                frame_width,
                frame_height
            ),
            to_pixel(
                landmarks["right_knee"],
                frame_width,
                frame_height
            ),
            to_pixel(
                landmarks["right_ankle"],
                frame_width,
                frame_height
            )
        )

    # Body ratio

    all_points = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle"
    ]

    missing_points = [
        key for key in all_points
        if landmarks.get(key) is None
    ]

    if not missing_points:

        points = [
            landmarks[key]
            for key in all_points
        ]

        body_ratio = calculate_body_ratio(
            points,
            frame_width,
            frame_height
        )

    # Return processed data

    return {
        "shoulder_center": shoulder,
        "hip_center": hip,
        "body_angle": body_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "body_ratio": body_ratio,
        "status": (
            "ok"
            if not missing_points
            else "partial_landmarks"
        ),
        "missing_landmarks": missing_points
    }


# ---------------------------------------
# 7. Convert MediaPipe landmarks
# ---------------------------------------

def convert_mediapipe_landmarks(person):

    landmark_ids = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28
    }

    landmarks = {}

    if person is None:
        return landmarks

    for name, index in landmark_ids.items():

        if index >= len(person):
            continue

        point = person[index]

        visibility = getattr(
            point,
            "visibility",
            None
        )

        if (
            visibility is not None
            and visibility < 0.5
        ):
            continue

        landmarks[name] = (
            point.x,
            point.y
        )

    return landmarks


# ---------------------------------------
# 8. MediaPipe Integration
# ---------------------------------------

def process_mediapipe_pose(
    person,
    frame_width=640,
    frame_height=480
):

    landmarks = convert_mediapipe_landmarks(
        person
    )

    return process_pose(
        landmarks,
        frame_width,
        frame_height
    )


# ---------------------------------------
# 9. Test
# ---------------------------------------

if __name__ == "__main__":

    test_landmarks = {
        "left_shoulder": (0.4, 0.2),
        "right_shoulder": (0.6, 0.2),
        "left_hip": (0.4, 0.5),
        "right_hip": (0.6, 0.5),
        "left_knee": (0.4, 0.7),
        "right_knee": (0.6, 0.7),
        "left_ankle": (0.4, 0.9),
        "right_ankle": (0.6, 0.9)
    }

    result = process_pose(
        test_landmarks
    )

    print("Pose Processing Result:")

    for key, value in result.items():
        print(f"{key}: {value}")