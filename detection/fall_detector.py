"""Posture-based fall detection for MediaPipe Pose landmarks."""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Tuple


_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_KNEE = 25
_RIGHT_KNEE = 26
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28

_MIN_VISIBILITY = 0.5
_LYING_ANGLE_DEGREES = 45.0
_SITTING_KNEE_ANGLE_DEGREES = 120.0
_FALL_WINDOW_SECONDS = 0.8
_LYING_CONFIRMATION_SECONDS = 1.0
_MIN_TORSO_ROTATION_DEGREES = 35.0


@dataclass(frozen=True)
class _PoseSample:
    timestamp: float
    posture: str
    torso_angle: float


def _coordinate(landmark: Any, name: str) -> Optional[float]:
    if isinstance(landmark, dict):
        value = landmark.get(name)
    else:
        value = getattr(landmark, name, None)

    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None

    return coordinate if math.isfinite(coordinate) else None


def _visible_point(landmarks: Any, index: int) -> Optional[Tuple[float, float]]:
    try:
        landmark = landmarks[index]
    except (IndexError, KeyError, TypeError):
        return None

    x = _coordinate(landmark, "x")
    y = _coordinate(landmark, "y")
    visibility = _coordinate(landmark, "visibility")
    if (
        x is None
        or y is None
        or visibility is None
        or visibility < _MIN_VISIBILITY
    ):
        return None

    return x, y


def _midpoint(
    landmarks: Any, first_index: int, second_index: int
) -> Optional[Tuple[float, float]]:
    first = _visible_point(landmarks, first_index)
    second = _visible_point(landmarks, second_index)
    if first is None and second is None:
        return None
    if first is None:
        return second
    if second is None:
        return first

    return (first[0] + second[0]) / 2, (first[1] + second[1]) / 2


def _knee_angle(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
    width: float,
    height: float,
) -> Optional[float]:
    hip_vector = ((hip[0] - knee[0]) * width, (hip[1] - knee[1]) * height)
    ankle_vector = (
        (ankle[0] - knee[0]) * width,
        (ankle[1] - knee[1]) * height,
    )
    hip_length = math.hypot(*hip_vector)
    ankle_length = math.hypot(*ankle_vector)
    if hip_length == 0 or ankle_length == 0:
        return None

    cosine = (
        hip_vector[0] * ankle_vector[0] + hip_vector[1] * ankle_vector[1]
    ) / (hip_length * ankle_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _pose_sample(
    pose_landmarks: Any,
    timestamp: float,
    width: float,
    height: float,
) -> Optional[_PoseSample]:
    landmarks = getattr(pose_landmarks, "landmark", pose_landmarks)
    shoulders = _midpoint(landmarks, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    hips = _midpoint(landmarks, _LEFT_HIP, _RIGHT_HIP)
    if shoulders is None or hips is None:
        return None

    dx = (shoulders[0] - hips[0]) * width
    dy = (shoulders[1] - hips[1]) * height
    return _PoseSample(
        timestamp=timestamp,
        posture=classify_posture(pose_landmarks, (width, height)),
        torso_angle=math.degrees(math.atan2(dx, dy)),
    )


def _angle_difference(first: float, second: float) -> float:
    return abs((first - second + 90) % 180 - 90)


def classify_posture(
    pose_landmarks: Any, image_size: Optional[Tuple[float, float]] = None
) -> str:
    """Classify a pose as ``lying``, ``sitting``, ``standing``, or ``unknown``.

    ``pose_landmarks`` may be a MediaPipe ``PoseLandmark`` sequence or the
    ``pose_landmarks`` result containing that sequence. ``image_size`` is an
    optional ``(width, height)`` pair used to account for frame aspect ratio.
    If knee landmarks are insufficient to distinguish sitting from standing,
    the result is ``unknown`` rather than an assumed standing posture.
    """
    if pose_landmarks is None:
        return "unknown"

    landmarks = getattr(pose_landmarks, "landmark", pose_landmarks)
    try:
        if image_size is None:
            width, height = 1.0, 1.0
        else:
            width, height = float(image_size[0]), float(image_size[1])
    except (IndexError, TypeError, ValueError):
        return "unknown"
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return "unknown"

    shoulders = _midpoint(landmarks, _LEFT_SHOULDER, _RIGHT_SHOULDER)
    hips = _midpoint(landmarks, _LEFT_HIP, _RIGHT_HIP)
    if shoulders is None or hips is None:
        return "unknown"

    dx = (shoulders[0] - hips[0]) * width
    dy = (shoulders[1] - hips[1]) * height
    torso_angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    if torso_angle >= _LYING_ANGLE_DEGREES:
        return "lying"

    has_leg_evidence = False
    for hip_index, knee_index, ankle_index in (
        (_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE),
        (_RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE),
    ):
        hip = _visible_point(landmarks, hip_index)
        knee = _visible_point(landmarks, knee_index)
        ankle = _visible_point(landmarks, ankle_index)
        if hip is None or knee is None or ankle is None:
            continue

        angle = _knee_angle(hip, knee, ankle, width, height)
        if angle is not None:
            has_leg_evidence = True
            if angle <= _SITTING_KNEE_ANGLE_DEGREES:
                return "sitting"

    return "standing" if has_leg_evidence else "unknown"


def detect_fall(
    pose_landmarks: Any,
    detector: "FallDetector",
    timestamp: Optional[float] = None,
    image_size: Optional[Tuple[float, float]] = None,
    person_id: Any = None,
) -> bool:
    """Update a person's detector with this frame and return a new fall event.

    Reuse the same ``FallDetector`` for every frame of one person's track.
    Pass ``person_id`` when available to reset history automatically when the
    tracked person changes.
    """
    return detector.update(pose_landmarks, timestamp, image_size, person_id)


class FallDetector:
    """Track one person's posture over time and detect a possible fall action."""

    def __init__(self) -> None:
        self._samples: deque[_PoseSample] = deque()
        self._fall_started_at: Optional[float] = None
        self._fall_reported = False
        self._last_update_at: Optional[float] = None
        self._person_id: Any = None
        self._interruption_started_at: Optional[float] = None

    def reset(self) -> None:
        """Clear tracking history so this detector can be reused."""
        self._samples.clear()
        self._fall_started_at = None
        self._fall_reported = False
        self._last_update_at = None
        self._person_id = None
        self._interruption_started_at = None

    def update(
        self,
        pose_landmarks: Any,
        timestamp: Optional[float] = None,
        image_size: Optional[Tuple[float, float]] = None,
        person_id: Any = None,
    ) -> bool:
        """Return ``True`` once when tracked movement resembles a fall.

        A possible fall requires a recent standing or sitting posture, a
        rapid torso rotation into a lying posture, and one full second of
        continued lying. Timestamps are seconds from a monotonic clock; if
        omitted, the current monotonic time is used. Missing poses preserve
        recent posture history and pause lying confirmation for short gaps.
        Pass ``person_id`` to reset automatically when the tracked person
        changes.
        """
        if person_id is not None:
            if self._last_update_at is not None and (
                self._person_id is None or person_id != self._person_id
            ):
                self.reset()
            self._person_id = person_id

        if self._fall_reported:
            return False

        now = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("timestamp must be a finite number of seconds")
        if self._last_update_at is not None and now <= self._last_update_at:
            raise ValueError("timestamp must increase for each frame")

        try:
            width, height = (
                (1.0, 1.0)
                if image_size is None
                else (float(image_size[0]), float(image_size[1]))
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("image_size must be a (width, height) pair") from error
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            raise ValueError("image_size dimensions must be positive and finite")

        self._last_update_at = now
        sample = (
            None
            if pose_landmarks is None
            else _pose_sample(pose_landmarks, now, width, height)
        )
        if sample is None or sample.posture == "unknown":
            if self._interruption_started_at is None:
                self._interruption_started_at = now
            return False

        if self._interruption_started_at is not None:
            interruption_duration = now - self._interruption_started_at
            if self._fall_started_at is not None:
                if interruption_duration <= _FALL_WINDOW_SECONDS:
                    self._fall_started_at += interruption_duration
                else:
                    self._fall_started_at = None
            self._interruption_started_at = None

        if self._fall_started_at is not None:
            if sample.posture == "lying":
                if now - self._fall_started_at >= _LYING_CONFIRMATION_SECONDS:
                    self._fall_reported = True
                    return True
            else:
                self._fall_started_at = None

        while (
            self._samples
            and now - self._samples[0].timestamp > _FALL_WINDOW_SECONDS
        ):
            self._samples.popleft()

        if sample.posture == "lying" and self._fall_started_at is None:
            previous = self._samples[-1] if self._samples else None
            if (
                previous is not None
                and previous.posture in ("standing", "sitting")
                and 0 < now - previous.timestamp <= _FALL_WINDOW_SECONDS
                and _angle_difference(sample.torso_angle, previous.torso_angle)
                >= _MIN_TORSO_ROTATION_DEGREES
            ):
                self._fall_started_at = now

        self._samples.append(sample)
        return False