import math
import unittest

from detection.fall_detector import FallDetector, classify_posture


def _pose(lying=False, include_legs=True, lean_degrees=0):
    landmarks = [None] * 33
    if lying:
        points = ((11, 0.15, 0.55), (12, 0.2, 0.55), (23, 0.5, 0.55), (24, 0.55, 0.55))
    else:
        angle = math.radians(lean_degrees)
        shoulder_x = 0.5 + math.sin(angle) * 0.35
        shoulder_y = 0.55 - math.cos(angle) * 0.35
        points = (
            (11, shoulder_x - 0.02, shoulder_y),
            (12, shoulder_x + 0.02, shoulder_y),
            (23, 0.48, 0.55),
            (24, 0.52, 0.55),
        )
    for index, x, y in points:
        landmarks[index] = {"x": x, "y": y, "visibility": 1.0}

    if include_legs and not lying:
        for index, x, y in (
            (25, 0.48, 0.72),
            (26, 0.52, 0.72),
            (27, 0.48, 0.9),
            (28, 0.52, 0.9),
        ):
            landmarks[index] = {"x": x, "y": y, "visibility": 1.0}
    return landmarks


def _sitting_pose():
    landmarks = _pose(include_legs=False)
    for index, x, y in (
        (25, 0.48, 0.7),
        (26, 0.52, 0.7),
        (27, 0.62, 0.7),
        (28, 0.58, 0.7),
    ):
        landmarks[index] = {"x": x, "y": y, "visibility": 1.0}
    return landmarks


class ClassifyPostureTests(unittest.TestCase):
    def test_normal_standing(self):
        self.assertEqual(classify_posture(_pose()), "standing")

    def test_sitting(self):
        self.assertEqual(classify_posture(_sitting_pose()), "sitting")

    def test_missing_leg_landmarks_do_not_claim_standing(self):
        self.assertEqual(classify_posture(_pose(include_legs=False)), "unknown")

    def test_missing_visibility_is_not_treated_as_reliable(self):
        landmarks = _pose()
        for index in (11, 12, 23, 24):
            landmarks[index]["visibility"] = None
        self.assertEqual(classify_posture(landmarks), "unknown")

    def test_low_visibility_is_not_treated_as_reliable(self):
        landmarks = _pose()
        for index in (11, 12, 23, 24):
            landmarks[index]["visibility"] = 0.1
        self.assertEqual(classify_posture(landmarks), "unknown")

    def test_single_visible_side_is_enough_to_classify(self):
        landmarks = _pose()
        for index in (12, 24, 26, 28):
            landmarks[index] = None
        self.assertEqual(classify_posture(landmarks), "standing")

    def test_lying_can_be_classified_without_leg_landmarks(self):
        self.assertEqual(classify_posture(_pose(lying=True)), "lying")


class FallDetectorTests(unittest.TestCase):
    def test_normal_sitting_does_not_trigger_a_fall(self):
        detector = FallDetector()
        self.assertFalse(detector.update(_pose(), 0.0))
        self.assertFalse(detector.update(_sitting_pose(), 0.2))
        self.assertFalse(detector.update(_sitting_pose(), 1.3))

    def test_rapid_transition_and_sustained_lying_reports_once(self):
        detector = FallDetector()
        self.assertFalse(detector.update(_pose(), 0.0))
        self.assertFalse(detector.update(_pose(lying=True), 0.2))
        self.assertFalse(detector.update(_pose(lying=True), 1.19))
        self.assertTrue(detector.update(_pose(lying=True), 1.21))
        self.assertFalse(detector.update(_pose(lying=True), 1.3))

    def test_gradual_reclining_does_not_use_stale_standing_history(self):
        detector = FallDetector()
        self.assertFalse(detector.update(_pose(), 0.0))
        for timestamp, angle in ((0.2, 20), (0.4, 40), (0.6, 45), (0.8, 65), (1.0, 85)):
            self.assertFalse(
                detector.update(_pose(lean_degrees=angle), timestamp)
            )
        self.assertFalse(detector.update(_pose(lying=True), 1.2))
        self.assertFalse(detector.update(_pose(lying=True), 2.3))

    def test_short_tracking_gap_pauses_confirmation_and_keeps_history(self):
        detector = FallDetector()
        self.assertFalse(detector.update(_pose(), 0.0))
        self.assertFalse(detector.update(_pose(lying=True), 0.2))
        self.assertFalse(detector.update(None, 0.5))
        self.assertFalse(detector.update(_pose(lying=True), 0.6))
        self.assertFalse(detector.update(_pose(lying=True), 1.1))
        self.assertTrue(detector.update(_pose(lying=True), 1.61))

    def test_long_tracking_gap_does_not_confirm_a_stale_fall(self):
        detector = FallDetector()
        self.assertFalse(detector.update(_pose(), 0.0))
        self.assertFalse(detector.update(_pose(lying=True), 0.2))
        self.assertFalse(detector.update(None, 0.3))
        self.assertFalse(detector.update(None, 1.2))
        self.assertFalse(detector.update(_pose(lying=True), 1.3))
        self.assertFalse(detector.update(_pose(lying=True), 2.4))

    def test_unknown_or_unreliable_pose_does_not_start_a_fall(self):
        detector = FallDetector()
        landmarks = _pose(lying=True)
        for index in (11, 12, 23, 24):
            landmarks[index]["visibility"] = 0.1

        self.assertFalse(detector.update(landmarks, 0.0))
        self.assertFalse(detector.update(None, 0.2))
        self.assertFalse(detector.update(landmarks, 1.3))

    def test_new_person_id_resets_history_and_reported_state(self):
        detector = FallDetector()
        self.assertFalse(detector.update(_pose(), 0.0, person_id="first"))
        self.assertFalse(detector.update(_pose(lying=True), 0.2, person_id="first"))
        self.assertTrue(detector.update(_pose(lying=True), 1.21, person_id="first"))

        self.assertFalse(detector.update(_pose(lying=True), 1.3, person_id="second"))
        self.assertFalse(detector.update(_pose(), 1.4, person_id="second"))
        self.assertFalse(detector.update(_pose(lying=True), 1.5, person_id="second"))

    def test_first_person_id_resets_unidentified_history(self):
        detector = FallDetector()
        self.assertFalse(detector.update(_pose(), 0.0))
        self.assertFalse(detector.update(_pose(lying=True), 0.2, person_id="new"))
        self.assertFalse(detector.update(_pose(lying=True), 1.21, person_id="new"))

    def test_timestamps_must_increase_across_tracking_interruptions(self):
        detector = FallDetector()
        self.assertFalse(detector.update(None, 1.0))
        with self.assertRaises(ValueError):
            detector.update(None, 1.0)


if __name__ == "__main__":
    unittest.main()
