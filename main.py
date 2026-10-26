#import os
#video_path = os.path.join("videos", "test_fall.mp4")
#print("AI Fall Emergency Detection System")
#print(f"Video Path: {video_path}")

#if os.path.exists(video_path):
#    print("Video file found. Proceeding with analysis...")
#else:
#    print("Video file not found.")
#    print ("Please ensure the video file is located in the 'videos' directory and named 'test_fall.mp4'.")

# main.py (ဗဟိုထိန်းချုပ်မှုဖိုင်)

import cv2
from database.database import create_database
from detection.pose_detector import detect_pose       # Role 1 ဖိုင်
from detection.fall_detector import check_fall        # Role 2 ဖိုင်
from alert.alarm import trigger_emergency_alert        # Role 3 (သင့်ဖိုင်)
from emergency.emergency_guide import display_emergency_screen # Role 3 (သင့်ဖိုင်)

def main():
    create_database() # အရင်ဆုံး DB ဆောက်မယ်
    cap = cv2.VideoCapture(0) # Live Camera ဖွင့်မယ်
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # ၁။ Role 1 ရဲ့ Pose ကို စစ်ဆေးမယ်
        landmarks = detect_pose(frame)
        
        # ၂။ Role 2 ရဲ့ လဲကျမှု ရှိမရှိ စစ်ဆေးမယ်
        is_fallen = check_fall(landmarks)
        
        # 🚨 ၃။ လဲကျသွားလျှင် သင့်ရဲ့ Role 3 အရေးပေါ်စနစ်ကို လှမ်းနှိုးမယ်
        if is_fallen:
            trigger_emergency_alert("R001", "FALL_DETECTED")
            display_emergency_screen({"name": "Kamata Taro"}, "FALL_DETECTED")
            break
            
    cap.release()

if __name__ == "__main__":
    main()
