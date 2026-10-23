
from types import SimpleNamespace
from detection.pose_processor_v2 import process_mediapipe_pose

# Create 33 dummy MediaPipe landmarks
person = [
    SimpleNamespace(x=0.0, y=0.0, visibility=0.0)
    for _ in range(33)
]

# Standing pose coordinates
points = {
    11: (0.4, 0.2),
    12: (0.6, 0.2),
    23: (0.4, 0.5),
    24: (0.6, 0.5),
    25: (0.4, 0.7),
    26: (0.6, 0.7),
    27: (0.4, 0.9),
    28: (0.6, 0.9)
}

for index, (x, y) in points.items():
    person[index] = SimpleNamespace(
        x=x, y=y, visibility=1.0
    )

result = process_mediapipe_pose(person, 640, 480)

print("MediaPipe Adapter Test:")
for key, value in result.items():
    print(f"{key}: {value}")
    
# Test: Right leg is not visible
partial_person = person.copy()

partial_person[26] = SimpleNamespace(
    x=0.0, y=0.0, visibility=0.0
)

partial_person[28] = SimpleNamespace(
    x=0.0, y=0.0, visibility=0.0
)

partial_result = process_mediapipe_pose(
    partial_person, 640, 480
)

print("\n--- Partial Landmarks Test ---")

for key, value in partial_result.items():
    print(f"{key}: {value}")
    
# Test: Left shoulder has low visibility
low_visibility_person = person.copy()

low_visibility_person[11] = SimpleNamespace(
    x=0.4,
    y=0.2,
    visibility=0.2
)

result_low = process_mediapipe_pose(
    low_visibility_person, 640, 480
)

print("\n--- Low Visibility Test ---")

for key, value in result_low.items():
    print(f"{key}: {value}")

assert result_low["status"] == "insufficient_landmarks"
assert "left_shoulder" in result_low["missing_landmarks"]

print("Low Visibility Test: PASS")

# Test: Body angle at different resolutions

tilted_person = person.copy()

# Move hip center slightly to the right
tilted_person[23] = SimpleNamespace(
    x=0.5, y=0.5, visibility=1.0
)

tilted_person[24] = SimpleNamespace(
    x=0.7, y=0.5, visibility=1.0
)

result1 = process_mediapipe_pose(
    tilted_person, 640, 480
)

result2 = process_mediapipe_pose(
    tilted_person, 1280, 960
)

print("\n--- Body Angle Accuracy Test ---")
print("640x480:", result1["body_angle"])
print("1280x960:", result2["body_angle"])

assert result1["body_angle"] == 23.96
assert result2["body_angle"] == 23.96

print("Body Angle Accuracy Test: PASS")