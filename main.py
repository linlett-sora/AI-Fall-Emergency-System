import os
video_path = os.path.join("videos", "test_fall.mp4")
print("AI Fall Emergency Detection System")
print(f"Video Path: {video_path}")

if os.path.exists(video_path):
    print("Video file found. Proceeding with analysis...")
else:
    print("Video file not found.")
    print ("Please ensure the video file is located in the 'videos' directory and named 'test_fall.mp4'.")