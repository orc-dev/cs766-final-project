import numpy as np
import pandas as pd
import re
import my_plot
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import cv2


def static_elbow(std_data): 
    #divide the frames into grids cell_numer*cell_number
    cell_number = 15
    cell_h = 1600//cell_number
    cell_w = 2500//cell_number
    #get the cell index of the left elbow
    w_loc_0 = [int(w//cell_w) for w in std_data[:, 27]] #left elbow x
    h_loc_0 = [int(h//cell_h) for h in std_data[:, 28]] #left elbow y
    #print(std_data[:,26])
    w_loc = [int(w//cell_w) for w in std_data[:, 27] if w != 0]
    h_loc = [int(h//cell_h) for h in std_data[:, 28] if h != 0]
    #print(w_loc)
    # std_data[:, 26], std_data[:,27] #left elbow
    cells = list(product(h_loc, w_loc))
    counter = Counter(cells)
    #find the cell that appears most frequently
    static_cell = counter.most_common(1)[0][0]
    h_comparisons = [static_cell[0] == h for h in h_loc_0]
    w_comparisons = [static_cell[1] == w for w in w_loc_0]
    #check whether the left elbow is in the staic cell
    indicator = [a and b for a, b in zip(h_comparisons, w_comparisons)]
    left_wrist = std_data[:, 32]
    left_elbow = std_data[:, 28]
    right_wrist = std_data[:, 34]
    right_elbow = std_data[:, 30]
    ind_wrist = [(left_wrist[i] >= left_elbow[i]) and (right_wrist[i] >= right_elbow[i]) for i in range(len(left_wrist))]
    #print(sum(indicator))
    return [p and q for p,q in zip(indicator, ind_wrist)] #returns an indicator array that corresponds to the time points of static positions
                     # 1: is static position 0: not static

def G_unit(std_data, d):
    
    #27: 'left_elbow', w
    #28: 'left_elbow', h
    #29: 'right_elbow', w
    #30: 'right_elbow', h
    #31: 'left_wrist', w
    #32: 'left_wrist', h
    #33: 'right_wrist', w
    #34: 'right_wrist', h

    (n, l) = std_data.shape
    #The velocity only represents from t_d to t_n-1
    l_elbow_v = ((std_data[d:, 27] - std_data[:(n-d), 27])**2 + (std_data[d:, 28] - std_data[:(n-d), 28])**2)/(std_data[d:, 0] - std_data[:(n-d), 0])
    r_elbow_v = ((std_data[d:, 29] - std_data[:(n-d), 29])**2 + (std_data[d:, 30] - std_data[:(n-d), 30])**2)/(std_data[d:, 0] - std_data[:(n-d), 0])
    l_wrist_v = ((std_data[d:, 31] - std_data[:(n-d), 31])**2 + (std_data[d:, 32] - std_data[:(n-d), 32])**2)/(std_data[d:, 0] - std_data[:(n-d), 0])
    r_wrist_v = ((std_data[d:, 33] - std_data[:(n-d), 33])**2 + (std_data[d:, 34] - std_data[:(n-d), 34])**2)/(std_data[d:, 0] - std_data[:(n-d), 0])
    #The acceleration only represents from t_d+1 to t_n-1
    l_elbow_a = (l_elbow_v[1:] - l_elbow_v[:-1])/(std_data[(d+1):, 0] - std_data[d:(n-1), 0])
    r_elbow_a = (r_elbow_v[1:] - r_elbow_v[:-1])/(std_data[(d+1):, 0] - std_data[d:(n-1), 0])
    l_wrist_a = (l_wrist_v[1:] - l_wrist_v[:-1])/(std_data[(d+1):, 0] - std_data[d:(n-1), 0])
    r_wrist_a = (r_wrist_v[1:] - r_wrist_v[:-1])/(std_data[(d+1):, 0] - std_data[d:(n-1), 0])
    #feature_df is a (n-d-1)*8 np.array, each row is a representation feature vector of a frame from t_d+1 to t_n-1
    feature_df = np.column_stack((l_elbow_v[1:], r_elbow_v[1:], l_wrist_v[1:], r_wrist_v[1:], l_elbow_a, r_elbow_a, l_wrist_a, r_wrist_a))
    #cluster the feature vectors into 2 groups

    #kmeans = KMeans(n_clusters=2)
    #kmeans.fit(feature_df)
    #clusters = kmeans.labels_

    # Z = linkage(feature_df, method='ward')
    # plt.figure(figsize=(10, 5))
    # dendrogram(Z, labels=np.arange(1, feature_df.shape[0]+1), color_threshold=0)
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample index')
    # plt.ylabel('Distance')
    # plt.show()

    clustering = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    clusters = clustering.fit_predict(feature_df)
    ###NEED LABEL!!!
    
    #Now try to find the cluster that aligns better with the preliminary static positions
    static_indicator = static_elbow(std_data)
    alignment_0 = np.sum(np.logical_and(static_indicator[(d+1):], clusters == 0))
    alignment_1 = np.sum(np.logical_and(static_indicator[(d+1):], clusters == 1))
    print(sum(clusters == 0))
    print(sum(clusters == 1))
    if alignment_0 > alignment_1:
        return clusters == 0
    else:
        return clusters == 1
    #The algorithm returns the indicator array when the frame is at rest unit
    #The method excluded the first d time points. only t_d to t_n-1


def segment_qom(std_data): #std_data is the numpy array of the std csv
                              #8L is the average number of frames for all gestures
    m = std_data.shape[0]
    #p = 35 #only consider the first p features
    #we consider the local quantity of movement, i.e., the Euclidean distances of the body junctions across frames
    qom_global = np.linalg.norm(std_data[1:(m) , 27:35] - std_data[0:(m - 1), 27:35], axis=1)/(std_data[1:(m) , 0] - std_data[0:(m - 1), 0])
    #qom_global is from t_1 to t_n-1
    static_indicator = static_elbow(std_data)
    #find the mean and std of the quantities of movement during static position
    me = np.mean(qom_global[static_indicator[1:]])
    sd = np.std(qom_global[static_indicator[1:]])
    #use mean + 2*sd as the threshold for checking rest positions
    threshold_inter = me + 0.5*sd
    #indices = np.where(qom_global < threshold_inter)[0]
    return qom_global < threshold_inter # the indicator vec from t_1 to t_n-1
    
    # indicator = np.ones(len(indices), dtype=bool)
    # for i in range(1, len(indices) - 1):
    #     # Check if both neighbors (numerically) are in the array
    #     if indices[i-1] == indices[i] - 1 and indices[i+1] == indices[i] + 1:
    #         indicator[i] = False
    # # plt.plot(std_data[1:,0], qom_global)
    # # plt.show()
    #return std_data[:,0][indices][indicator]

def elbow_traj(std_data):
    static_time = static_elbow(std_data)
    #print(sum(static_time))
    mean_w = np.mean(std_data[static_time, 26])
    mean_h = np.mean(std_data[static_time, 27])
    eu = (std_data[:, 27] - mean_w)**2 + (std_data[:, 28] - mean_h)**2 
    #Euclidean distance between the location of the current elbow and that of the static state
    #print(eu)
    #find the average distance fluctruation during the static time
    eu_static = (std_data[static_time, 27] - mean_w)**2 + (std_data[static_time, 28] - mean_h)**2 
    #use mean+2*sd as threshold
    tau = np.mean(eu_static) + np.std(eu_static)
    #print(tau)
    return eu < tau #return a indicator vec of static positions, the vec is from t_0 to t_n-1


def smooth_ind(vec, L, r):
    '''input: an indicator vec of static positions
       output: an smoothed indicator vec'''
    #L is the pre-specified half moving window width
    #r is the threshold of ratio of near-neighbor 1s: 
    #  if above r portion of the nearest 2L points have 1s, the current point is 1
    n = len(vec)
    smoothed_vec = np.zeros(n)
    for i in range(L, n-L):
        smoothed_vec[i] = np.mean(vec[(i-L): (i+L)])
    return smoothed_vec > r
    

def regex_video_name(video_path):
    pattern = r'/([^/]+)\.[^\.]+$'
    match = re.search(pattern, video_path)
    video_name = match.group(1) if match else 'No match found'
    return video_name

def rgb(*args):
    return np.array(args) / 255.0

def visualize_elbow_wrist_data(df, base_name):
    x = [df[:,29],df[:,33],df[:,27],df[:,31]]
    #    right elbow: right wrist; left elbow: left wrist
    y = [df[:,30],df[:,34],df[:,28],df[:,32]]

    colors = [
        rgb(240, 35, 70),
        rgb(230, 109, 137),
        rgb(70, 35, 240),
        rgb(140, 167, 230)
    ]
    my_plot.make_scatter_plot_multiple(x, y, colors, 
        'Elbow and Wrist Distribution', 
        f'{base_name}_elbow_wrist_landmarks.png')

    timeline = df[:,0]
    my_plot.make_line_segment_plot_multiple(timeline, y[2:4], colors, 
        'Elbow and Wrist Height',
        f'{base_name}_elbow_wrist_height.png')
    
def visualize_rest_pose_elbow_wrist(df, vec, d, title = '', file_path = None):
    #    right elbow: right wrist; left elbow: left wrist
    y = [df[:,30],df[:,34],df[:,28],df[:,32]]
    colors = [
        rgb(240, 35, 70),
        rgb(230, 109, 137),
        rgb(70, 35, 240),
        rgb(140, 167, 230)
    ]
    timeline = df[:,0]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(0,2):
        #ax.plot(x, y[i], linestyle='-', color=rgb(220,220,220), alpha=0.4, linewidth=0.5)
        ax.plot(timeline, y[i], marker='o', linestyle='None', color=colors[i + 2], markersize=1, alpha=0.4)
    for i in range(d, len(timeline)-d-1):  # Loop over the data points
        if vec[i] == 1:  # Condition to highlight
            ax.axvspan(timeline[i], timeline[i + 1], color='yellow', alpha=0.5)
    # Shoulder height
    SHOULDER_HEIGHT = 1000
    ax.axhline(y=SHOULDER_HEIGHT, color=rgb(200, 100, 100), linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Timeline (second)')
    ax.set_ylabel('Height of Elbow and Wrist Landmarks')
    ax.set_xlim([0, max(timeline)]) 
    ax.set_ylim([1600, 0])
    ax.grid(True)
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, dpi=1000)
        plt.close(fig)

def show_clip_video(video_path = 'MAP108_AB.mov', time_points = [(5, 10), (20, 25)] ):
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Get frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Process each segment
    for i, (start, end) in enumerate(time_points):
        # Calculate the start and end frames
        start_frame = int(start * fps)
        end_frame = int(end * fps)
    
        # Set the current frame position of the video file to 'start_frame'
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # or use 'XVID' for .avi format
        out = cv2.VideoWriter(f'output_segment_{i+1}.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        
        # Read and write frames from 'start_frame' to 'end_frame'
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
    
        # Release the output file
        out.release()

    # Release the VideoCapture object
    cap.release()





def main():
    data_list = [
    "MAP014_std.csv",
    "MAP037_std.csv", 
    "MAP040_std.csv", 
    "MAP046_std.csv", 
    "MAP061_std.csv", 
    "MAP063_std.csv", 
    "MAP064_std.csv", 
    "MAP108_AB__std.csv"
    ]
    std_csv_path = f'tables/{data_list[7]}'
    base_name = regex_video_name(std_csv_path)[:-4]
    #with open(std_csv_path, 'r') as file:
    #    column_names = np.array(file.readline().strip().split(','))
    df = np.loadtxt(std_csv_path, delimiter=',', skiprows=1)

    #check static elbow time period
    # static_time = static_elbow(df)
    # visualize_rest_pose_elbow_wrist(df, static_time, 1, title = 'preliminary static time',
    #                                  file_path = 'preliminary_visualization.png')

    #segment qom

    # L = 10
    # qom_ind = segment_qom(df)
    # print(sum(qom_ind))
    # rest_qom = smooth_ind(qom_ind, L, 0.6)
    # visualize_rest_pose_elbow_wrist(df, rest_qom, 1, title = 'Quantity of movement method',
    #                                 file_path = 'qom_visualization.png')

   # elbow traj
    # L = 10
    # elbow_ind = smooth_ind(elbow_traj(df), L, 0.8)
    # visualize_rest_pose_elbow_wrist(df, elbow_ind, 1, title = 'Elbow Trajectory method',
    #                                 file_path = 'traj_visualization.png')

    #G_unit
    # L = 10
    # gunit_ind = smooth_ind(G_unit(df,3), L, 0.8)
    # visualize_rest_pose_elbow_wrist(df, gunit_ind, 3, title = 'Quantity of movement method',
    #                                  file_path = 'gunit_visualization.png')

    #combine elbow_traj and segment qom
    L = 10
    qom_ind = smooth_ind(segment_qom(df), L, 0.6)
    elbow_ind = smooth_ind(elbow_traj(df), L, 0.8)
    rest_ind = [a and b for a,b in zip(qom_ind, elbow_ind[1:])] #from t_1 to t_n-1
    visualize_rest_pose_elbow_wrist(df, rest_ind, 3, title = 'QOM + ELBOW method',
                                      file_path = 'compound_method_visualization.png')

    # #visualize_elbow_wrist_data(df, base_name)
    # indicator = rest_ind #from t_1 to t_n-1
    # for i in range(1, len(indicator) - 1):
    #     # Check if both neighbors (numerically) are in the array
    #     if rest_ind[i-1] == True and rest_ind[i+1] == True:
    #         indicator[i] = False
    # print(indicator)
    # index_ind = np.where(indicator == True)
    # print(index_ind[1:(len(index_ind) - 1)])
    # time_points = df[1:, 0][index_ind[1:(len(index_ind) - 1)]]
    # print(len(time_points))







if __name__ == '__main__':
    main()
