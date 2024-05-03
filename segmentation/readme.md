configuration:
pip install pandas
pip install matplotlib
pip install -U numpy scipy scikit-learn
pip install cv2


Gesture temporal segmentation:

segment.py implemented several temporal segmentation techniques from various published computer vision papers with proper asjustments to fit our data.

Static_elbow: finds the preliminary static time periods. 
G_unit: uses velocities and accelations as feature vector and applies hierachical/kmeans clustering.
segment_qom: thresholds the local velocity to distinguish between rest time periods and active time periods.
elbow_traj: thresholds distance from the elbow position to the static position to distinguish between rest and action.
smooth_ind: smooth the rest time indicator vector, making it jumps between 0 and 1 less frequently.
regex_video_name, rgb, visualize_elbow_wrist_data, visualize_rest_pose_elbow_wrist, show_clip_video: visualize the segmentation results.

Usage: choose one dataset from the datalist, prespecify the parameter L, run segment.py, the program returns rest_ind-- the rest time indicator vector and prints the plots showing the segmentation.
