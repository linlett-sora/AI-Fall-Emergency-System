
from detection.pose_processor_v2 import process_pose

# Test 1: Standing
standing = {
    "left_shoulder": (0.4, 0.2),
    "right_shoulder": (0.6, 0.2),
    "left_hip": (0.4, 0.5),
    "right_hip": (0.6, 0.5),
    "left_knee": (0.4, 0.7),
    "right_knee": (0.6, 0.7),
    "left_ankle": (0.4, 0.9),
    "right_ankle": (0.6, 0.9)
}

# Test 2: Sitting
sitting = {
    "left_shoulder": (0.35, 0.2),
    "right_shoulder": (0.5, 0.2),
    "left_hip": (0.35, 0.5),
    "right_hip": (0.5, 0.5),
    "left_knee": (0.35, 0.75),
    "right_knee": (0.5, 0.75),
    "left_ankle": (0.6, 0.75),
    "right_ankle": (0.75, 0.75)
}

# Test 3: Lying
lying = {
    "left_shoulder": (0.2, 0.4),
    "right_shoulder": (0.2, 0.5),
    "left_hip": (0.5, 0.4),
    "right_hip": (0.5, 0.5),
    "left_knee": (0.7, 0.4),
    "right_knee": (0.7, 0.5),
    "left_ankle": (0.9, 0.4),
    "right_ankle": (0.9, 0.5)
}

test_cases = {
    "Standing": standing,
    "Sitting": sitting,
    "Lying": lying
}

for name, landmarks in test_cases.items():
    result = process_pose(landmarks)

    print(f"\n--- {name} ---")
    print("Body Angle:", result["body_angle"])
    print("Left Knee:", result["left_knee_angle"])
    print("Right Knee:", result["right_knee_angle"])
    print("Body Ratio:", result["body_ratio"])
    # -----------------------------------
# Lying + Partial Occlusion Test
# -----------------------------------

lying_partial = {
    "left_shoulder": (0.2, 0.4),
    "right_shoulder": (0.2, 0.5),

    "left_hip": (0.5, 0.4),
    "right_hip": (0.5, 0.5),

    "left_knee": (0.7, 0.4),
    "left_ankle": (0.9, 0.4)

    # Right knee and ankle are hidden
}

result = process_pose(lying_partial, 640, 480)

print("\n--- Lying + Partial Occlusion Test ---")

for key, value in result.items():
    print(f"{key}: {value}")

assert result["body_angle"] == 90.0
assert result["left_knee_angle"] == 180.0
assert result["right_knee_angle"] is None
assert result["status"] == "partial_landmarks"

print("Lying + Partial Occlusion Test: PASS")