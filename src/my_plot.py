import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CANVAS_HEIGHT = 1600
CANVAS_WIDTH  = 2560
SHOULDER_WIDTH = 320
SHOULDER_HEIGHT = 1000

def rgb(*args):
    return np.array(args) / 255.0


def make_scatter_plot(x, y, file_path):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Scatter plot
    ax.scatter(x, y, s=2, color=rgb(20, 20, 200), alpha=0.4)
    ax.set_xlim([0, 2560])
    ax.set_ylim([1600, 0])
    ax.set_title(f'Shoulder Landmarks in Screen Coordinate')
    ax.grid()

    # Save plot
    plt.savefig(file_path, dpi=1000)
    plt.close(fig)


def make_line_segment_plot(x, y, file_path):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create the line plot using the axes object
    ax.plot(x, y, linestyle='-', color=(0.7,0.7,0.7), alpha=0.5, linewidth=1)
    ax.plot(x, y, marker='o', linestyle='None', color=rgb(20, 20, 200), markersize=1, alpha=0.4)
    ax.axhline(y=320, color=rgb(200, 100, 100), linestyle='--', linewidth=1)

    ax.set_title('Shoulder Distance Change Over Time')
    ax.set_xlabel('Timeline (second)')
    ax.set_ylabel('Horizontal distance between should landmarks')

    ax.set_xlim([0, max(x)]) 
    ax.set_ylim([0, 1500])
    ax.grid(True)
    
    # Save plot
    plt.savefig(file_path, dpi=1000)
    plt.close(fig)


def make_scatter_plot_multiple(x, y, colors, title='', file_path=None):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Scatter plot
    for i in range(len(x)):
        ax.scatter(x[i], y[i], s=2, color=colors[i], alpha=0.4)
    
    # Shoulders
    x_s = np.array([CANVAS_WIDTH - SHOULDER_WIDTH, 
                    CANVAS_WIDTH + SHOULDER_WIDTH]) / 2
    y_s = np.array([SHOULDER_HEIGHT, SHOULDER_HEIGHT])
    ax.scatter(x_s, y_s, s=50, color=rgb(245, 185, 20), alpha=1)

    # Axes settings
    ax.set_xlim([0, 2560])
    ax.set_ylim([1600, 0])
    ax.set_title(title)
    ax.grid()

    # Show or save plot
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, dpi=1000)
        plt.close(fig)


def make_line_segment_plot_multiple(x, y, colors, title='', file_path=None):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create the line plot using the axes object
    for i in range(len(y)):
        ax.plot(x, y[i], linestyle='-', color=rgb(220,220,220), alpha=0.4, linewidth=0.5)
        ax.plot(x, y[i], marker='o', linestyle='None', color=colors[i + 2], markersize=1, alpha=0.4)

    # Shoulder height
    ax.axhline(y=SHOULDER_HEIGHT, color=rgb(200, 100, 100), linestyle='--', linewidth=1)

    ax.set_title(title)
    ax.set_xlabel('Timeline (second)')
    ax.set_ylabel('Height of Elbow and Wrist Landmarks')

    ax.set_xlim([0, max(x)]) 
    ax.set_ylim([1600, 0])
    ax.grid(True)
    
    # Show or save plot
    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path, dpi=1000)
        plt.close(fig)