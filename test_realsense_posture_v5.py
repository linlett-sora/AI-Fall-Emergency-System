"""Experimental RealSense posture and potential-risk demo (V5).
Run from project root: python test_realsense_posture_v5.py
Does not modify team modules, send messages, or diagnose unconsciousness.
Optional person validation: python -m pip install ultralytics
When YOLO is absent, pose plausibility is used instead (weaker).
"""
import math
import time
import threading
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
try:
    import winsound
except ImportError:
    winsound = None

from detection.pose_detector import PoseDetector
from detection.fall_detector import classify_posture

CHECKING_SECONDS = 5.0
ASSISTANCE_SECONDS = 10.0
POSTURE_HOLD_SECONDS = 1.2
UNKNOWN_GRACE_SECONDS = 0.8
PERSON_HOLD_SECONDS = 0.5
PERSON_LOST_SECONDS = 0.65
MOVEMENT_THRESHOLD = 0.065  # normalized torso/hip displacement, over ~0.7 s
MOVEMENT_COOLDOWN = 1.5
MIN_VISIBILITY = 0.5
MAX_DEPTH_METERS = 5.0
USE_YOLO_IF_AVAILABLE = False  # Set True only after installing ultralytics and testing side-lying poses
YOLO_CONFIDENCE = 0.50
YOLO_EVERY_N_FRAMES = 5
WINDOW_NAME = 'RealSense Posture V5 - EXPERIMENTAL'


def alarm():
    def worker():
        if winsound is None:
            print('ALARM (audio unavailable)')
            return
        try:
            for _ in range(3):
                winsound.Beep(1800, 250)
                time.sleep(0.15)
        except Exception as exc:
            print('Alarm audio error:', exc)
    threading.Thread(target=worker, daemon=True).start()


def point(landmarks, index):
    if landmarks is None or len(landmarks) <= index:
        return None
    p = landmarks[index]
    if (getattr(p, 'visibility', 1.0) < MIN_VISIBILITY
            or not math.isfinite(p.x) or not math.isfinite(p.y)
            or not (0 <= p.x <= 1 and 0 <= p.y <= 1)):
        return None
    return np.array([p.x, p.y], dtype=float)


def center(landmarks, a, b):
    x, y = point(landmarks, a), point(landmarks, b)
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
    """V2 core preserved; adds conservative *candidate* slumped/reclined labels."""
    if landmarks is None:
        return 'no_person', {}
    shoulders = center(landmarks, 11, 12)
    hips = center(landmarks, 23, 24)
    knees = center(landmarks, 25, 26)
    ankles = center(landmarks, 27, 28)
    nose = point(landmarks, 0)
    if shoulders is None or hips is None:
        return 'uncertain', {}
    dx = (shoulders[0] - hips[0]) * w
    dy = (shoulders[1] - hips[1]) * h
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    hip_depth = depth_at(depth_frame, hips, w, h)
    shoulder_depth = depth_at(depth_frame, shoulders, w, h)
    knee_depth = depth_at(depth_frame, knees, w, h)
    ankle_depth = depth_at(depth_frame, ankles, w, h)
    basic = classify_posture(landmarks, (w, h))
    info = dict(angle=angle, hip_depth=hip_depth, shoulder_depth=shoulder_depth,
                knee_depth=knee_depth, ankle_depth=ankle_depth, basic=basic,
                shoulders=shoulders, hips=hips, knees=knees, ankles=ankles)

    # V2 posture rules, unchanged in order and thresholds.
    if basic == 'lying':
        posture = 'lying'
    elif angle >= 45 and hip_depth is not None and shoulder_depth is not None:
        posture = 'supported_lying'
    elif (knees is not None and ankles is not None and hip_depth is not None
          and hips[1] >= 0.52
          and abs(ankles[0] - hips[0]) * w > 0.65 * max(abs(ankles[1] - hips[1]) * h, 1.0)
          and abs(ankles[0] - hips[0]) * w > 0.20 * w and angle >= 18):
        posture = 'supported_lying'
    elif knees is None or ankles is None:
        posture = 'uncertain'
    else:
        posture = basic if basic in ('standing', 'sitting', 'unknown') else 'uncertain'

    # Experimental subtypes: not medical or definitive posture labels.
    # Preserve normal sitting unless several slumping cues co-occur.
    if posture == 'sitting' and nose is not None and knees is not None:
        head_low = nose[1] >= shoulders[1] - 0.025
        forward_tilt = angle >= 24
        if head_low and forward_tilt:
            posture = 'slumped_sitting'
    # If V2 calls an upright-supported pose sitting, use clear torso tilt + low hips.
    if posture == 'sitting' and angle >= 32 and hips[1] >= 0.55 and hip_depth is not None:
        posture = 'reclined_sitting'
    return posture, info


class StablePosture:
    def __init__(self):
        self.current = 'uncertain'
        self.candidate = None
        self.since = None
        self.last_valid = None

    def update(self, raw, now):
        if raw in ('unknown', 'uncertain', 'no_person'):
            if self.last_valid is not None and now - self.last_valid <= UNKNOWN_GRACE_SECONDS:
                return self.current
            self.current = 'no_person' if raw == 'no_person' else 'uncertain'
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


class MovementTracker:
    """Detects torso displacement; hand-only motion and subtle motion may be missed."""
    def __init__(self):
        self.history = deque()

    def update(self, info, now):
        hip = info.get('hips')
        shoulder = info.get('shoulders')
        if hip is None or shoulder is None:
            self.history.clear()
            return False
        self.history.append((now, (hip + shoulder) / 2))
        while len(self.history) > 1 and now - self.history[0][0] > 0.7:
            self.history.popleft()
        if len(self.history) < 2 or now - self.history[0][0] < 0.30:
            return False
        return np.linalg.norm(self.history[-1][1] - self.history[0][1]) >= MOVEMENT_THRESHOLD

    def reset(self):
        self.history.clear()


def plausible_person(landmarks, depth_frame, w, h):
    """Weak fallback, not a person detector. Rejects incomplete/implausible poses."""
    shoulders = center(landmarks, 11, 12)
    hips = center(landmarks, 23, 24)
    if shoulders is None or hips is None:
        return False
    torso = np.linalg.norm((shoulders - hips) * np.array([w, h]))
    if torso < 20 or torso > 330:
        return False
    visible = sum(point(landmarks, i) is not None for i in (11, 12, 23, 24, 25, 26, 27, 28))
    return visible >= 4 and depth_at(depth_frame, hips, w, h) is not None


def load_yolo():
    if not USE_YOLO_IF_AVAILABLE:
        return None
    try:
        from ultralytics import YOLO
        # Requires model weights available locally or downloadable by Ultralytics.
        model = YOLO('yolo11n.pt')
        print('YOLO person validation enabled')
        return model
    except Exception as exc:
        print('YOLO unavailable; using weaker pose validation:', exc)
        return None


def yolo_person(model, frame):
    results = model.predict(source=frame, conf=YOLO_CONFIDENCE, classes=[0],
                            verbose=False, imgsz=640)
    return any(len(r.boxes) > 0 for r in results)


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    detector = None
    started = False
    posture_filter = StablePosture()
    movement = MovementTracker()
    yolo = None
    frame_index = 0
    last_yolo = False
    person_since = None
    last_person_time = None
    risk_start = None
    alarm_sent = False
    motion_pause_until = 0.0
    last_label = None
    try:
        # YOLO load can download weights on first use. No changes to team files.
        yolo = load_yolo()
        pipeline.start(config)
        started = True
        detector = PoseDetector()
        print('V5 EXPERIMENTAL | Q=quit R=reset | No medical diagnosis / notifications')
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data()).copy()
            h, w = frame.shape[:2]
            now = time.monotonic()
            frame_index += 1
            landmarks = detector.detect_pose(frame)
            pose_ok = plausible_person(landmarks, depth_frame, w, h)
            if yolo is not None and frame_index % YOLO_EVERY_N_FRAMES == 1:
                try:
                    last_yolo = yolo_person(yolo, frame)
                except Exception as exc:
                    print('YOLO runtime error; rejecting person until recovered:', exc)
                    last_yolo = False
            candidate_person = pose_ok and (last_yolo if yolo is not None else True)
            if candidate_person:
                last_person_time = now
                if person_since is None:
                    person_since = now
            else:
                person_since = None
            confirmed_person = (candidate_person and person_since is not None
                                and now - person_since >= PERSON_HOLD_SECONDS)
            # Short dropouts do not force a new posture, but do not advance risk timer.
            recently_lost = (not confirmed_person and last_person_time is not None
                             and now - last_person_time <= PERSON_LOST_SECONDS)
            if confirmed_person:
                raw, info = estimate(landmarks, depth_frame, w, h)
            else:
                raw, info = ('uncertain' if recently_lost else 'no_person'), {}
            stable = posture_filter.update(raw, now)
            moving = confirmed_person and movement.update(info, now)
            if not confirmed_person:
                movement.reset()
            risk_labels = ('lying', 'supported_lying', 'slumped_sitting', 'reclined_sitting')
            safe_labels = ('standing', 'sitting')
            # Only accept a recovered posture after stabilization; transient raw noise
            # pauses the risk clock without declaring recovery.
            recovered = confirmed_person and stable in safe_labels and raw in safe_labels
            if recovered or (not confirmed_person and not recently_lost):
                risk_start = None
                alarm_sent = False
                motion_pause_until = 0.0
            # Motion alone is NOT proof of recovery: a collapsed person may move.
            # Only stable sitting/standing or a confirmed empty scene clears risk.
            risk_pose = (confirmed_person and raw in risk_labels and stable in risk_labels)
            if risk_pose:
                if risk_start is None:
                    risk_start = now
            else:
                # Uncertain frames restart the timer rather than falsely accumulate.
                risk_start = None
                alarm_sent = False
            duration = now - risk_start if risk_start is not None and risk_pose else 0.0
            status = 'Posture: ' + (stable if confirmed_person else 'no_person').upper()
            color = (0, 255, 0)
            if risk_pose:
                color = (0, 255, 255)
                if duration >= ASSISTANCE_SECONDS:
                    status = 'ASSISTANCE NEEDED (DEMO)'
                    color = (0, 0, 255)
                    if not alarm_sent:
                        alarm_sent = True
                        alarm()
                elif duration >= CHECKING_SECONDS:
                    status = 'CHECKING...'
                    color = (0, 165, 255)
            if confirmed_person and landmarks is not None:
                detector.draw_pose(frame, landmarks)
            lines = [status, f'Raw: {raw} | Stable: {stable}',
                     f'Risk timer: {duration:.1f}s',
                     f'Torso angle (2D): {info.get("angle", 0):.1f} deg',
                     f'Hip camera distance: {info.get("hip_depth") or 0:.2f} m',
                     'Person gate: ' + ('YOLO + Pose' if yolo is not None else 'Pose only (weak)'),
                     'Motion: ' + ('YES (not recovery)' if moving else 'NO'),
                     'Q: Quit | R: Reset', 'EXPERIMENTAL - NOT MEDICAL']
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, color if i == 0 else (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, frame)
            label = (raw, stable, status)
            if label != last_label:
                print('raw:', raw, 'stable:', stable, 'status:', status)
                last_label = label
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                posture_filter.reset()
                movement.reset()
                for name in ('previous_landmarks', 'previous_points'):
                    value = getattr(detector, name, None)
                    if hasattr(value, 'clear'):
                        value.clear()
                person_since = None
                last_person_time = None
                risk_start = None
                alarm_sent = False
                motion_pause_until = 0.0
                last_label = None
                print('RESET')
    finally:
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()
        if started:
            pipeline.stop()


if __name__ == '__main__':
    main()
