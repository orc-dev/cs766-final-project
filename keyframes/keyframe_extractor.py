"""
File Name: keyframe_extractor.py
Date Created: Apr.20 2024
Python Version: 3.11.2

Description:
This program contains the `extract_keyframe()` method, which extract keyframes
from an input gesture chunk. Meanwhile, it contains methods to make video from
these extracted keyframe with simple linear interpolation.
"""
import cv2
import pandas as pd
import numpy as np

CANVAS_HEIGHT = 1600
CANVAS_WIDTH  = 2560

# Prefix of column names (landmarks) which we want to display
vertex_key = [f'PS_{i}' for i in range(11, 25)]
edge_key = [
    [11, 12], [12, 24], [23, 24], [23, 11],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19]
]


def out_of_bound(x, y, threshold):
    return x <= threshold or y <= threshold


def make_video(chunk, duration, output_name, write_image=False):
    frame_num = len(chunk)
    fps = frame_num / duration

    # print(f'start_time: {start_time} sec, ({start_time // 60}:{start_time % 60})')
    # print(f'end_time: {end_time} sec, ({end_time // 60}:{end_time % 60})')
    # print(f'duration: {duration} sec')
    # print(f'frame num: {frame_num} rows')
    # print(vertex_key)

    # # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'example/{output_name}.mp4', fourcc, fps, 
        (CANVAS_WIDTH, CANVAS_HEIGHT))

    # Make each frame by drawing circles and line segments
    for row_idx, row in chunk.iterrows():
        # Create a blank image
        frame = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        frame[:] = [60, 60, 60]

        # High light wrist node
        for key in vertex_key:
            x = int(row[f'{key}_x'])
            y = int(row[f'{key}_y'])

            # skip outliers
            if out_of_bound(x, y, 500):
                continue

            if key == 'PS_15' or key == 'PS_16':
                cv2.circle(frame, (x, y), 12, (250, 250, 15), -1)

        # Draw connections
        for e in edge_key:
            x1 = int(row[f'PS_{e[0]}_x'])
            y1 = int(row[f'PS_{e[0]}_y'])
            x2 = int(row[f'PS_{e[1]}_x'])
            y2 = int(row[f'PS_{e[1]}_y'])
            
            if out_of_bound(x1, y1, 500) or out_of_bound(x2, y2, 500):
                continue
            else:
                cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), 2)

        # Draw each landmark 
        for key in vertex_key:
            x = int(row[f'{key}_x'])
            y = int(row[f'{key}_y'])

            # skip outliers
            if out_of_bound(x, y, 500):
                continue

            cv2.circle(frame, (x, y), 7, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 4, (255, 15, 15),   -1)
        
        # Write the frame to the video
        video.write(frame)

        # (Optional debug) Save the frame as an image
        if write_image:
            frame_filename = f'example/video_images/frame_{row_idx - 3117}.png'
            cv2.imwrite(frame_filename, frame)
            
    # Release the video writer
    video.release()


def extract_keyframe(chunk):
    
    def compute_velocity(px, py, t):
        # Compute raw velocity
        dx = np.diff(px)
        dy = np.diff(py)
        velo = np.sqrt(dx ** 2 + dy ** 2) / np.diff(t)

        # Remove noise
        threshold = 175
        for i in range(1, len(velo) - 1):
            if (velo[i] >= threshold): 
                continue
            
            prev = velo[i - 1]
            next = velo[i + 1]
            
            d1 = prev - threshold
            d2 = next - threshold
            
            if (d1 * d2 > 0):
                velo[i] = 0
        return velo
    
    def get_oneside_keyframe(velo):
        bin_velo = (velo > 0).astype(int)
        key_flag = np.ones(len(bin_velo)).astype(int)

        for i in range(1, len(key_flag) - 1):
            prev = bin_velo[i - 1]
            curr = bin_velo[i]
            next = bin_velo[i + 1]
            key_flag[i] = (curr == 0 and (prev != 0 or next != 0))

        keyframes = np.where(key_flag > 0)[0]
        return keyframes

    def combine_closed_points(xa, ya, xb, yb, kid):
        result = [kid[0]]
        
        a1 = np.array([xa[kid[0]], ya[kid[0]]])
        b1 = np.array([xb[kid[0]], yb[kid[0]]])
        
        def far_enough(p2, p1):
            return np.sum((p2 - p1) ** 2) > 4000
        
        for i in range(1, len(kid)):
            k = kid[i]
            
            a2 = np.array([xa[k], ya[k]])
            b2 = np.array([xb[k], yb[k]])
            
            if k - kid[i - 1] > 5 or far_enough(a2, a1) or far_enough(b2, b1):
                result.append(k)
                a1 = a2
                b1 = b2 

        # Handle last frame
        z = len(xa) - 1
        y = result[-1]

        a2 = np.array([xa[z], ya[z]])
        b2 = np.array([xb[z], yb[z]])

        if z - y > 5 or far_enough(a2, a1) or far_enough(b2, b1):
            result.append(z)
        else:
            result[-1] = z

        return np.array(result)

    # Wrist data
    x_lo = chunk['PS_16_x'].to_numpy()
    y_lo = chunk['PS_16_y'].to_numpy()
    x_hi = chunk['PS_15_x'].to_numpy()
    y_hi = chunk['PS_15_y'].to_numpy()
    t = chunk['timestamp'].to_numpy()
   
    # Extract keyframe from lo-side
    velocity_lo = compute_velocity(x_lo, y_lo, t)
    keyframe_lo = get_oneside_keyframe(velocity_lo)

    # Extract keyframe from hi-side
    velocity_hi = compute_velocity(x_hi, y_hi, t)
    keyframe_hi = get_oneside_keyframe(velocity_hi)

    union = np.union1d(keyframe_lo, keyframe_hi)
    return combine_closed_points(x_lo, y_lo, x_hi, y_hi, union)


def create_keyframe_based_data(chunk, keyframe):
    # Create a new copy of dataframe from input chunk
    df = pd.DataFrame(chunk).copy(deep=True)
    df.reset_index(drop=True, inplace=True)  # reset index
    print(keyframe)
    
    # Linear interpolate values inbetween keyframe
    for i in range(len(keyframe) - 1):
        row_from = keyframe[i]
        row_end  = keyframe[i + 1]

        t0 = df.iloc[row_from, 0]
        t1 = df.iloc[row_end,  0]

        head = df.loc[row_from, 'PS_11_x':'PS_24_y']
        tail = df.loc[row_end,  'PS_11_x':'PS_24_y']
        
        # Remove outliers
        head_zeros = (np.array(head) <= 500)
        tail_zeros = (np.array(tail) <= 500)
        either_zeros = head_zeros | tail_zeros
        indices = np.where(either_zeros)[0]
        
        for rid in range(row_from + 1, row_end):
            t = df.iloc[rid, 0]
            theta = (t - t0) / (t1 - t0)
            new_data = np.array(head + (tail - head) * theta)
            new_data[indices] = 0
            df.loc[rid, 'PS_11_x':'PS_24_y'] = new_data

    # Create a new column
    keyframe_flag = np.zeros(len(df)).astype(int)
    keyframe_flag[keyframe] = 1
    df['keyframe_flag'] = keyframe_flag

    return df


def main():
    # Read csv data and extract selected section
    base_name = 'MAP014'
    cln_file = f'example/{base_name}_clean.csv'
    std_file = f'example/{base_name}_std.csv'
    
    # Show skeleton animation of the original data
    df = pd.read_csv(cln_file)
    start_time = 740
    end_time = 770
    chunk = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time + 1)]
    #make_video(chunk, end_time - start_time, output_name='PRE')

    # Show skeleton animation of the Fixed-shoulder data
    df = pd.read_csv(std_file)
    start_time = 750
    end_time = 780
    chunk = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time + 1)]
    #make_video(chunk, end_time - start_time, output_name='FST')
    
    # Extract keyframe from a gesture chunk
    keyframe = extract_keyframe(chunk)
    keyframe_df = create_keyframe_based_data(chunk, keyframe)
    #make_video(keyframe_df, end_time - start_time, output_name='KEY')


# python3.11 example/keyframe_extractor.py
if __name__ == '__main__':
    main()