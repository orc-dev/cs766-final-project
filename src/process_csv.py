"""
File Name: process_csv.py
Date Created: Mar.20 2024
Python Version: 3.11.2

Description:
...

Usage
- Frome `code_base` directory
- Run `python3.11 src/process_csv.py`
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import my_plot

CANVAS_HEIGHT = 1600
CANVAS_WIDTH  = 2560
SHOULDER_WIDTH = 320
SHOULDER_HEIGHT = 1000

def regex_video_name(video_path):
    # Extract substring inbetween the last '/' and the last '.'
    pattern = r'/([^/]+)\.[^\.]+$'
    match = re.search(pattern, video_path)
    video_name = match.group(1) if match else 'No match found'
    return video_name


def split_df(df):
    tuple_columns = [col for col in df.columns if col != 'timestamp']
    new_cols_dfs = []

    for col in tuple_columns:
        # Split (x,y) into two columns
        new_col_df = pd.DataFrame(df[col].tolist(), 
            columns=[f'{col}_x', f'{col}_y'], index=df.index)

        # Append the new DataFrame to the list
        new_cols_dfs.append(new_col_df)

    # Concatenate all the new column DataFrames horizontally (axis=1)
    df = pd.concat([df.drop(tuple_columns, axis=1)] + new_cols_dfs, axis=1)
    return df


def clean_data(raw_csv_path):
    # Define the function to process each point
    def process_point(point):
        # Convert the string representation of a tuple to an actual tuple
        if isinstance(point, str):
            point = eval(point)
        # Check if the point is valid
        if 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
            # Rescale the (x,y) point
            return (point[0] * CANVAS_WIDTH, point[1] * CANVAS_HEIGHT)
        else:
            # Replace with (0,0) if out of bounds
            return (0, 0)
    
    # Read raw input data
    df = pd.read_csv(raw_csv_path)

    # Initialize a new DataFrame with the same columns
    new_df = pd.DataFrame(columns=df.columns)

    # Iterate over the columns
    for column in df.columns:
        # Check if the column contains landmark data ('PS_', 'LH_', or 'RH_')
        if (column.startswith('PS_') or 
            column.startswith('LH_') or 
            column.startswith('RH_')):
            # Process each cell in the column
            new_df[column] = df[column].apply(lambda x: process_point(x))
        else:
            new_df[column] = df[column]

    return split_df(new_df)



def visualize_shoulder_data(df, base_name):
    # 1. Visualize shoulder landmarks
    x = pd.concat([df['PS_11_x'], df['PS_12_x']], axis=0).to_numpy()
    y = pd.concat([df['PS_11_y'], df['PS_12_y']], axis=0).to_numpy()
    my_plot.make_scatter_plot(x, y, 
        f'output/plots/{base_name}_shoulder_landmarks.png')
    
    # 2. Visualize shoulder distance
    timeline = df['timestamp'].to_numpy()
    shoulder_dx = df['PS_11_x'].to_numpy() - df['PS_12_x'].to_numpy()
    my_plot.make_line_segment_plot(timeline, shoulder_dx,
        f'output/plots/{base_name}_shoulder_distance.png')


def visualize_elbow_wrist_data(df, base_name):
    # 1. Visualize shoulder landmarks
    x = [
        df['PS_14_x'].to_numpy(),
        df['PS_16_x'].to_numpy(),
        df['PS_13_x'].to_numpy(),
        df['PS_15_x'].to_numpy()
    ]

    y = [
        df['PS_14_y'].to_numpy(),
        df['PS_16_y'].to_numpy(),
        df['PS_13_y'].to_numpy(),
        df['PS_15_y'].to_numpy()
    ]

    colors = [
        rgb(240, 35, 70),
        rgb(230, 109, 137),
        rgb(70, 35, 240),
        rgb(140, 167, 230)
    ]
    my_plot.make_scatter_plot_multiple(x, y, colors, 
        'Elbow and Wrist Distribution', 
        f'output/plots/{base_name}_elbow_wrist_landmarks.png')

    # 2. Visualize shoulder distance
    timeline = df['timestamp'].to_numpy()
    my_plot.make_line_segment_plot_multiple(timeline, y[2:4], colors, 
        'Elbow and Wrist Height',
        f'output/plots/{base_name}_elbow_wrist_height.png')


def rgb(*args):
    return np.array(args) / 255.0

def quick_plot(x, y):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create the line plot using the axes object
    ax.plot(x, y, linestyle='-', color=(0.7,0.7,0.7), alpha=0.5, linewidth=1)
    ax.plot(x, y, marker='o', linestyle='None', color=rgb(20, 20, 200), markersize=1, alpha=0.4)
    ax.axhline(y=320, color=rgb(200, 100, 100), linestyle='--', linewidth=1)

    ax.set_title('Title')
    ax.set_xlabel('Timeline (second)')
    ax.set_ylabel('Y_label')

    ax.set_xlim([0, max(x)]) 
    ax.set_ylim([0, max(y) + 1])
    ax.grid(True)

    plt.show()

def transform_to_shoulder_coordinate(df):
    # shoulder landmarks id: 12  11

    # Create a new copy of data
    new_df = df.copy()

    # Shoulder width
    shoulder_dx = df['PS_11_x'].to_numpy() - df['PS_12_x'].to_numpy()
    temp = np.zeros(len(shoulder_dx), dtype=int)

    temp[(shoulder_dx == 0)] = 0                           # missing
    temp[(shoulder_dx > 0) & (shoulder_dx < 200)] = 1      # turn around
    temp[(shoulder_dx >= 200) & (shoulder_dx <= 400)] = 2  # standard
    temp[(shoulder_dx > 400)] = 3                          # near sit

    new_df['stat_0'] = temp
    indices = new_df.index[new_df['stat_0'] != 2].tolist()
    
    # Reference points for fixed shoulder
    lo_x = (CANVAS_WIDTH - SHOULDER_WIDTH) / 2
    hi_x = (CANVAS_WIDTH + SHOULDER_WIDTH) / 2
    
    # Translation vectors
    lo_dx = df['PS_12_x'].to_numpy() - lo_x
    lo_dy = df['PS_12_y'].to_numpy() - SHOULDER_HEIGHT

    hi_dx = df['PS_11_x'].to_numpy() - hi_x
    hi_dy = df['PS_11_y'].to_numpy() - SHOULDER_HEIGHT

    # Update lower section
    lo_x_cols = [f'PS_{i}_x' for i in range(12, 33, 2)] + [f'RH_{i}_x' for i in range(21)]
    lo_y_cols = [f'PS_{i}_y' for i in range(12, 33, 2)] + [f'RH_{i}_y' for i in range(21)]
    
    for col in lo_x_cols: 
        new_df[col] = new_df[col].to_numpy() - lo_dx
        new_df.loc[indices, col] = 0

    for col in lo_y_cols:
        new_df[col] = new_df[col].to_numpy() - lo_dy
        new_df.loc[indices, col] = 0

    # Update higher section
    hi_x_cols = [f'PS_{i}_x' for i in range(11, 33, 2)] + [f'LH_{i}_x' for i in range(21)]
    hi_y_cols = [f'PS_{i}_y' for i in range(11, 33, 2)] + [f'LH_{i}_y' for i in range(21)]
    
    for col in hi_x_cols:
        new_df[col] = new_df[col].to_numpy() - hi_dx
        new_df.loc[indices, col] = 0

    for col in hi_y_cols:
        new_df[col] = new_df[col].to_numpy() - hi_dy
        new_df.loc[indices, col] = 0

    return new_df


data_list = [
    "MAP014_raw.csv",
    "MAP108_AB'_raw.csv",
    'trim_3_6_raw.csv',
    'trim_tiny_raw.csv'
]

def main():
    # Extract base name from input files with the pattern {base_name}_raw.csv
    raw_csv_path = f'output/tables/{data_list[0]}'
    base_name = regex_video_name(raw_csv_path)[:-4]

    # Clean out-of-bound landmarks with (0,0) and split data tuple
    df = clean_data(raw_csv_path)
    df.to_csv(f'output/tables/{base_name}_clean.csv', index=False)
    
    # Visualize shoulder data
    visualize_shoulder_data(df, base_name)
    
    # Transform landmarks on 'Fixed-Shoulder Coordinate System'
    # Use median of shoulder-width as a scalar to normalize data
    std_df = transform_to_shoulder_coordinate(df)
    #visualize_shoulder_data(std_df, f'{base_name}_std')
    std_df.to_csv(f'output/tables/{base_name}_std.csv', index=False)

    # Visualize elbow-wrist data
    visualize_elbow_wrist_data(std_df, base_name)

if __name__ == '__main__':
    main()



