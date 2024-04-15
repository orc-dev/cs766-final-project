"""
File Name: process_video.py
Date Created: Mar.20 2024
Python Version: 3.11

Description:
This program reads the original video file, add pose/hand landmarks and 
save the marked video. Meanwhile, it stores the frame id, timestamp and
the coordinates of each landmark-point to a csv file.

Dependencies:
- OpenCV
- MediaPipe (0.10.9)

Usage
- Frome `code_base` directory
- Run `python3.11 src/process_video.py`
"""
import time
import re
import os
import cv2
import mediapipe as mp
import pandas as pd
import landmark_name

# Initialize body landmarks detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

# Initialize hand landmarks detector
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands()

# For displaying the landmarks on the video frames
mp_drawing = mp.solutions.drawing_utils 


def highlight_landmark(frame, results_pose, style):
    for i, color in style.items():
        point = results_pose.pose_landmarks.landmark[i]
        x = int(point.x * frame.shape[1])
        y = int(point.y * frame.shape[0])
        
        cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 5, color,           -1)


def init_row(timestamp):
    frame_data = {'timestamp': timestamp}

    # Pose landmarks ('PS')
    for i in range(33):
        frame_data[f'PS_{i}'] = (0, 0)

    # Left hand landmarks ('LH')
    for i in range(21):
        frame_data[f'LH_{i}'] = (0, 0)

    # Right hand landmarks ('RH')
    for i in range(21):
        frame_data[f'RH_{i}'] = (0, 0)

    return frame_data


def process_frame(frame, frame_id, timestamp, landmarks_list):
    # Convert the frame to RGB for MediaPipe and process the RGB frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb_frame)
    results_hands = hands.process(rgb_frame)
    
    # Initialize data for the current frame
    frame_data = init_row(timestamp)
    
    # Draw body landmarks and collect their data if they exist
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for i, point in enumerate(results_pose.pose_landmarks.landmark):
            frame_data[f'PS_{i}'] = (point.x, point.y)
    
    # Draw hand landmarks and collect their data if they exist
    if results_hands.multi_hand_landmarks:
        for hid, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Classify the hand as left or right
            temp = results_hands.multi_handedness[hid].classification[0].label
            hand_label = 'LH' if temp == 'Left' else 'RH'
            
            for i, point in enumerate(hand_landmarks.landmark):
                frame_data[f'{hand_label}_{i}'] = (point.x, point.y)
    
    # Append the data for the current frame to the list.
    landmarks_list.append(frame_data)
    return frame


def process_video(dat_path, out_path):
    # Start to process
    start_time = time.time()
    print('\nprocess_video() starts running...')
    print(f'  dat_path: {dat_path}')
    print(f'  out_path: {out_path}')

    # Read input video
    cap = cv2.VideoCapture(dat_path)

    # Set parameters for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    print(f'frame_size: {frame_size}')

    out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    
    # Init parameter for building dataframe
    landmarks_list = []
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print('Reached the end or unrecoverable error.')
            break 
        
        timestamp = frame_id / fps
        marked_frame = process_frame(frame, frame_id, timestamp, landmarks_list)
        out.write(marked_frame)
        frame_id += 1
        
    # Release the video capture object
    cap.release()
    
    # Compute and display runtime
    run_time = (time.time() - start_time) / 60
    print(f'Program exits. Runtime: {run_time:.2f} min')
    return landmarks_list


def regex_video_name(video_path):
    # Extract substring inbetween the last '/' and the last '.'
    pattern = r'/([^/]+)\.[^\.]+$'
    match = re.search(pattern, video_path)
    video_name = match.group(1) if match else 'No match found'
    return video_name

video_list = [
    "MAP014.mov",
    "MAP019.mov",
    "MAP037_AB.mp4",
    "MAP040_A'B.mov",
    "MAP046_A'B.mov",
    "MAP061_AB'.mov",
    "MAP063_A'B'.mov",
    "MAP064_A'B'.mov",
    "MAP066_AB'.mov",
    "MAP108_AB'.mov",
]

def main():
    dat_path = f'data/{video_list[8]}'

    # Build out path and process the data video
    video_name = regex_video_name(dat_path)
    out_path = f'output/videos/{video_name}_marked.mp4'
    landmarks_list = process_video(dat_path, out_path)

    # Write and save csv file
    df = pd.DataFrame(landmarks_list)
    csv_file_path = f'output/tables/{video_name}_raw.csv'
    df.to_csv(csv_file_path, index=False)


if __name__ == '__main__':
    main()
