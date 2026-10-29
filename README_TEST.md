# RealSense Fall Detection & Posture Tests

## Overview

This folder contains experimental detection tests for the **AI Fall Detection & Emergency Support System**, developed for Kamata Festival 2026.

These tests are separate from the team's main application (`main.py`).

**Git Branch:** `feature/realsense-detection-tests`

## 1. Requirements

### Hardware

* Windows PC
* Intel RealSense depth camera (D435i used for development)
* USB connection compatible with the camera

### Software

* Python (tested locally with Python 3.14.4)
* MediaPipe
* OpenCV
* NumPy
* pyrealsense2

The package versions were tested in the development environment. Installation on other Python versions or operating systems has not been verified.

## 2. Installation

Clone the repository:

```bash
git clone <repository-url>
cd AI_Fall_Emergency_System
```

Switch to the testing branch:

```bash
git fetch origin
git switch feature/realsense-detection-tests
```

If the branch does not exist locally, use:

```bash
git switch --track origin/feature/realsense-detection-tests
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

**Note:** `winsound` is included with Python on Windows and does not require separate installation.

## 3. MediaPipe Model

The following model file is required:

```text
models/pose_landmarker_lite.task
```

Make sure the model exists before running the tests.

## 4. Running the Tests

### RealSense Camera Test

```bash
python test_realsense.py
```

### RealSense Depth Test

```bash
python test_realsense_depth_demo.py
```

### RealSense Fall Detection

```bash
python test_realsense_fall.py
```

### Posture Detection V2

```bash
python test_realsense_posture_v2.py
```

### Posture Detection V5

```bash
python test_realsense_posture_v5.py
```

### Posture Detection V5 Fix

```bash
python test_realsense_posture_v5_fix.py
```

**V5 Fix is the current experimental version being evaluated.**

These tests are intended for development and comparison. They are not production-ready safety systems.

## 5. Optional YOLO Person Validation

V5 and V5 Fix include optional YOLO person validation.

By default:

```python
USE_YOLO_IF_AVAILABLE = False
```

Therefore, Ultralytics is not required for the default configuration.

To experiment with YOLO:

```bash
python -m pip install ultralytics
```

Then change the setting to `True` in the relevant test file.

**Important:** Test side-lying and partially obscured poses before enabling YOLO. YOLO may fail to recognize people in unusual postures.

The YOLO model weights may need to be downloaded separately.

## 6. Alarm and Safety Testing

Test the following situations:

1. Normal standing
2. Normal sitting
3. Bending to pick up an object
4. Lying down slowly
5. Side-lying
6. Reclining or supported lying
7. Possible fall
8. Recovery after a fall
9. No person in the camera view
10. Objects that may be mistaken for people

Check for both missed detections and false alarms.

## 7. Known Limitations

* MediaPipe may misclassify lying or crouching as standing.
* Pose landmarks may be missing when the person is partially obscured.
* A person lying down does not necessarily indicate an emergency.
* The system cannot determine whether someone is unconscious.
* RealSense depth information alone does not guarantee accurate floor-height detection.
* Without YOLO, person validation relies on pose plausibility and may produce false positives.
* Detection thresholds and alarm behavior require further testing.

## 8. Future Improvements

Planned research and development:

* Combine YOLO person detection with MediaPipe Pose.
* Use Intel RealSense depth data to improve posture and risk evaluation.
* Improve detection of side-lying, back-facing, and supported-lying poses.
* Reduce false positives caused by clothing, furniture, and other objects.
* Distinguish posture classification from emergency risk assessment.
* Evaluate detection accuracy using multiple test scenarios.

These improvements are **future work**, not confirmed completed features.

## 9. Important Notes

* Do not modify `main.py` when running these standalone tests.
* Use fictional data for testing.
* Do not upload private medical information.
* This system is an educational prototype and must not be relied upon for real emergency monitoring.
