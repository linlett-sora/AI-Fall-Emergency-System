
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