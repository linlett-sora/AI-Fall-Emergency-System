
import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# RGB Camera
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

# Depth Camera
config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

try:
    pipeline.start(config)

    while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        depth_image = np.asanyarray(
            depth_frame.get_data()
        )

        # Measure center distance in meters
        distance = depth_frame.get_distance(320, 240)

        # Display distance on RGB image
        cv2.putText(
            color_image,
            f"Distance: {distance:.2f} m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Convert depth image to colors
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(
                depth_image,
                alpha=0.03
            ),
            cv2.COLORMAP_JET
        )

        # Display camera images
        cv2.imshow(
            "RealSense RGB",
            color_image
        )

        cv2.imshow(
            "RealSense Depth",
            depth_colormap
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()