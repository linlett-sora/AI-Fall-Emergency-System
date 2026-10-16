
import math

def get_center(left, right):
    x = (left[0] + right[0]) / 2
    y = (left[1] + right[1]) / 2
    return (x, y)

def calculate_body_angle(shoulder, hip):
    dx = hip[0] - shoulder[0]
    dy = hip[1] - shoulder[1]

    angle = math.degrees(
        math.atan2(abs(dx), abs(dy))
    )

    return round(angle, 2)

def calculate_joint_angle(a, b, c):
    angle1 = math.atan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle2 = math.atan2(
        c[1] - b[1], c[0] - b[0]
    )

    angle = abs(math.degrees(angle1 - angle2))

    if angle > 180:
        angle = 360 - angle
    return round(angle, 2)

def calculate_body_ratio(points, frame_width, frame_height):
    xs = [p[0] * frame_width for p in points]
    ys = [p[1] * frame_height for p in points]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    if width == 0:
        return None

    return round(height / width, 2)

def process_pose(landmarks, frame_width=640, frame_height=480):
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    left_hip = landmarks["left_hip"]
    right_hip = landmarks["right_hip"]

    shoulder = get_center(left_shoulder, right_shoulder)
    hip = get_center(left_hip, right_hip)

    body_angle = calculate_body_angle(shoulder, hip)

    left_knee_angle = None
    right_knee_angle = None
    body_ratio = None

    required = [
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]

    if all(key in landmarks for key in required):
        left_knee_angle = calculate_joint_angle(
            left_hip,
            landmarks["left_knee"],
            landmarks["left_ankle"]
        )

        right_knee_angle = calculate_joint_angle(
            right_hip,
            landmarks["right_knee"],
            landmarks["right_ankle"]
        )

        points = [
            landmarks[key] for key in [
                "left_shoulder", "right_shoulder",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
                "left_ankle", "right_ankle"
            ]
        ]

        body_ratio = calculate_body_ratio(
            points, frame_width, frame_height
        )

    return {
        "shoulder_center": shoulder,
        "hip_center": hip,
        "body_angle": body_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "body_ratio": body_ratio
    }


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

    result = process_pose(test_landmarks)

    print("Pose Processing Result:")
    for key, value in result.items():
        print(f"{key}: {value}")