AI Fall Detection & Emergency Support System

Main Features:
- Fall Detection
- No-Movement Timer
- Emergency Alert
- Resident ID
- Emergency Profile
- Rescue Guidance

Future Features:
- Danger Zone Detection
- Helmet Detection
- Staff Notification
- Dashboard
- Heart Rate Sensor


# AI Fall Detection & Emergency Support System

## Project Overview
An AI-based system designed to detect possible falls, monitor movement, and provide emergency assistance.

Project: Kamata Festival 2026

## Planned Features
- Human Pose Detection
- Fall Detection
- No-Movement Timer
- Emergency Alarm
- Resident ID Management
- Emergency Medical Profile
- Rescue Guidance

## Future Features
- Danger Zone Detection
- Helmet Detection
- Staff Notification
- Safety Dashboard
- Heart Rate Sensor

## Technologies
- Python
- OpenCV
- MediaPipe
- NumPy
- SQLite

## Team Roles


## Team Roles (4 People)

### Role 1 - Pose Detection (2 People)

#### Person 1A - Pose Detection Core
- detection/pose_detector.py
- Video and camera processing
- Body landmark detection
- Skeleton visualization

#### Person 1B - Pose Data Processing
- detection/pose_processor.py
- Landmark coordinate processing
- Body angle calculation
- Posture data preparation
- Send processed data to Fall Detection

### Role 2 - Fall Detection (1 Person)
- detection/fall_detector.py
- Standing / Sitting / Lying analysis
- Fall detection logic
- Fall timer and movement analysis

### Role 3 - Emergency & Database (1 Person)
- alert/alarm.py
- emergency/emergency_guide.py
- database/database.py
- resident/resident_manager.py
- Warning and alarm system
- Emergency rescue guidance
- Resident information management
- Event history storage

### Role 4 - Main System (All 4 Members)
- main.py
- config.py
- User interface
- Module integration
- Testing and debugging
- Final demonstration
## Installation

1. Clone the GitHub repository.
2. Open the project folder in VS Code.
3. Install required libraries:

    python -m pip install -r requirements.txt

4. Initialize the database:

    python database/database.py

5. Run the basic application:

    python main.py

Note: Video detection features are still under development.

## Current Progress
- [x] Project folder created
- [x] GitHub repository created
- [x] Python libraries installed on initial PC
- [x] Database table created
- [ ] Pose Detection
- [ ] Fall Detection
- [ ] Emergency Alert
- [ ] User Interface
- [ ] Final Testing

## Important Notes
- Use fictional resident data for testing.
- Do not upload private medical data.
- Develop features using separate Git branches.