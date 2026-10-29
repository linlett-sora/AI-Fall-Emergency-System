"""Experimental posture/risk demo. Does not modify team modules or send notifications.
Run from the project root: python test_realsense_posture_v2.py

IMPORTANT: Pose + depth cannot establish unconsciousness or true floor height.
"""
import math
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    import winsound
except ImportError:
    winsound = None

from detection.pose_detector import PoseDetector
from detection.fall_detector import FallDetector, classify_posture

CHECKING_SECONDS = 5.0
ASSISTANCE_SECONDS = 10.0
POSTURE_HOLD_SECONDS = 1.2
UNKNOWN_GRACE_SECONDS = 1.5
MIN_VISIBILITY = 0.5
MAX_DEPTH_METERS = 5.0
WINDOW_NAME = "RealSense Posture V2 - EXPERIMENTAL"


def alarm():
    if winsound is None:
        print("ALARM (audio unavailable)")
        return
    def worker():
        for _ in range(3):
            winsound.Beep(1800, 250)
            time.sleep(0.15)
    threading.Thread(target=worker, daemon=True).start()


def point(landmarks, index):
    if landmarks is None or len(landmarks) <= index:
        return None
    p = landmarks[index]
    if (getattr(p, "visibility", 1.0) < MIN_VISIBILITY
            or not math.isfinite(p.x) or not math.isfinite(p.y)
            or not (0 <= p.x <= 1 and 0 <= p.y <= 1)):
        return None
    return np.array([p.x, p.y], dtype=float)


def center(landmarks, a, b):
    x, y = point(landmarks, a), point(landmarks, b)
    if x is None and y is None:
        return None
    if x is None:
        return y
    if y is None:
        return x
    return (x + y) / 2


def depth_at(frame, pt, w, h):
    if pt is None:
        return None
    x = max(0, min(w - 1, int(pt[0] * w)))
    y = max(0, min(h - 1, int(pt[1] * h)))
    d = frame.get_distance(x, y)
    return d if math.isfinite(d) and 0 < d <= MAX_DEPTH_METERS else None


def estimate(landmarks, depth_frame, w, h):
    """Conservative heuristic: avoid confidently labeling incomplete poses standing."""
    if landmarks is None:
        return "no_person", {}
    shoulders = center(landmarks, 11, 12)
    hips = center(landmarks, 23, 24)
    knees = center(landmarks, 25, 26)
    ankles = center(landmarks, 27, 28)
    if shoulders is None or hips is None:
        return "uncertain", {}

    # Angle is measured in image coordinates, not 3D floor orientation.
    dx = (shoulders[0] - hips[0]) * w
    dy = (shoulders[1] - hips[1]) * h
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    hip_depth = depth_at(depth_frame, hips, w, h)
    shoulder_depth = depth_at(depth_frame, shoulders, w, h)
    knee_depth = depth_at(depth_frame, knees, w, h)
    ankle_depth = depth_at(depth_frame, ankles, w, h)
    basic = classify_posture(landmarks, (w, h))

    details = {"angle": angle, "hip_depth": hip_depth,
               "shoulder_depth": shoulder_depth, "knee_depth": knee_depth,
               "ankle_depth": ankle_depth, "basic": basic}
    if basic == "lying":
        return "lying", details

    # Clearly sideways trunk: plausible supported/reclined lying.
    # Requires visible hips and depth; not proof of contact with floor.
    if angle >= 45 and hip_depth is not None and shoulder_depth is not None:
        return "supported_lying", details

    # A horizontally extended leg and low hip may indicate reclining.
    # Require both knee and ankle geometry to avoid treating a bent knee as lying.
    if knees is not None and ankles is not None and hip_depth is not None:
        leg_dx = abs(ankles[0] - hips[0]) * w
        leg_dy = abs(ankles[1] - hips[1]) * h
        if (hips[1] >= 0.52 and leg_dx > 0.65 * max(leg_dy, 1.0)
                and leg_dx > 0.20 * w and angle >= 18):
            return "supported_lying", details

    # If lower-body evidence is absent, avoid claiming 'standing'.
    # This includes the user's propped-up pose where legs leave the frame.
    if knees is None or ankles is None:
        return "uncertain", details

    # A nearly upright torso alone does not prove standing.
    return basic if basic in ("standing", "sitting", "unknown") else "uncertain", details


class StablePosture:
    """Hold a label briefly before accepting; tolerate short unknown gaps."""
    def __init__(self):
        self.current = "uncertain"
        self.candidate = None
        self.since = None
        self.last_valid = None

    def update(self, raw, now):
        if raw in ("unknown", "uncertain", "no_person"):
            if self.last_valid is not None and now - self.last_valid <= UNKNOWN_GRACE_SECONDS:
                return self.current
            self.current = "no_person" if raw == "no_person" else "uncertain"
            self.candidate = None
            self.since = None
            return self.current
        self.last_valid = now
        if raw == self.current:
            self.candidate = None
            self.since = None
            return self.current
        if raw != self.candidate:
            self.candidate = raw
            self.since = now
        elif now - self.since >= POSTURE_HOLD_SECONDS:
            self.current = raw
            self.candidate = None
            self.since = None
        return self.current

    def reset(self):
        self.__init__()


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pose_detector = None
    fall_detector = None
    started = False
    posture_filter = StablePosture()
    risk_start = None
    alarm_sent = False
    fall_latched = False
    last_label = None

    try:
        pipeline.start(config)
        started = True
        pose_detector = PoseDetector()
        fall_detector = FallDetector()
        print("V2 EXPERIMENTAL: Q=quit, R=reset; no DB/email; not a medical detector")
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data()).copy()
            h, w = frame.shape[:2]
            now = time.monotonic()
            landmarks = pose_detector.detect_pose(frame)
            raw, info = estimate(landmarks, depth_frame, w, h)
            stable = posture_filter.update(raw, now)

            # Existing fall detector is used unchanged. Only accept its fall
            # signal when the experimental pose estimate is not absent/uncertain.
            fall = fall_detector.update(landmarks, image_size=(w, h))
            if fall and raw not in ("no_person", "uncertain", "unknown"):
                fall_latched = True
                alarm()
                print("FALL DETECTED (experimental)")

            risk_pose = (stable in ("lying", "supported_lying")
                         and raw in ("lying", "supported_lying"))
            if risk_pose:
                if risk_start is None:
                    risk_start = now
            elif stable in ("standing", "sitting", "no_person"):
                risk_start = None
                alarm_sent = False
            # Unknown frames do not advance the timer or trigger an alert.
            duration = (now - risk_start) if risk_start is not None and risk_pose else 0.0
            status = "Posture: " + stable.upper()
            color = (0, 255, 0)
            if risk_pose:
                color = (0, 255, 255)
                if duration >= ASSISTANCE_SECONDS:
                    status, color = "ASSISTANCE NEEDED (DEMO)", (0, 0, 255)
                    if not alarm_sent:
                        alarm_sent = True
                        alarm()
                elif duration >= CHECKING_SECONDS:
                    status, color = "CHECKING...", (0, 165, 255)
            if fall_latched:
                status, color = "FALL DETECTED (DEMO)", (0, 0, 255)

            if landmarks is not None:
                pose_detector.draw_pose(frame, landmarks)
            lines = [status, f"Raw: {raw} | Stable: {stable}",
                     f"Risk timer: {duration:.1f}s",
                     f"Torso angle (2D): {info.get('angle', 0):.1f} deg",
                     f"Hip camera distance: {info.get('hip_depth') or 0:.2f} m",
                     "Q: Quit | R: Reset", "EXPERIMENTAL - NOT MEDICAL"]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (12, 32 + 29 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.56, color if i == 0 else (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            label = (raw, stable, status)
            if label != last_label:
                print("raw:", raw, "stable:", stable, "status:", status)
                last_label = label
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                fall_detector.reset()
                posture_filter.reset()
                for name in ("previous_landmarks", "previous_points"):
                    value = getattr(pose_detector, name, None)
                    if hasattr(value, "clear"):
                        value.clear()
                risk_start = None
                alarm_sent = False
                fall_latched = False
                last_label = None
                print("RESET")
    finally:
        cv2.destroyAllWindows()
        if pose_detector is not None:
            pose_detector.close()
        if started:
            pipeline.stop()


if __name__ == "__main__":
    main()
