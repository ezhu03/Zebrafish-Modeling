'''
This code takes in a given file and video and displays the video with a reflection visualization of the fish side by side with the original video.
'''
import os
from pprint import pprint
import pathlib
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats
import seaborn as sns
import pandas as pd
import trajectorytools as tt
import trajectorytools.plot as ttplot
import trajectorytools.socialcontext as ttsocial
import matplotlib.animation as animation
from IPython import display
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter

plt.rcParams['animation.ffmpeg_path'] = '/Users/ezhu/Documents/GitHub/Zebrafish-Modeling/ffmpeg'

# Render/export controls
OUTPUT_DPI = 200  # adjust higher (e.g., 150–200) for even sharper output

'''
SET THESE VALUES BEFORE RUNNING THE CODE
radius: radius of the tank (for reflection calculation)
file: path to the numpy file containing the position and velocity data
video: path to the tracked video file to be displayed
'''

radius = 5

file = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-half1/trajectories/validated.npy"
video = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-half1/1fish-1fps-15min-21dpf-half1_2024-07-16-122129-0000_tracked.avi"

#file = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/trajectories/validated.npy"
#video = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/1fish15min1fps-half-1-21dpf_tracked.avi"

# Save the merged array to a new .npy file
#np.save("merged_file.npy", merged_array)



# Step 1: Install Required Packages
# pip install matplotlib ffmpeg

# Step 2: Import Libraries

# Step 3: Create a Function to Update the Frame
import cv2
'''
    opens the provided numpy file to be processed by trajectorytools
'''
def openfile(file, sigma = 1):
    tr = tt.Trajectories.from_idtrackerai(file, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': sigma})
    return tr
tr = openfile(file)


'''
uses the trajectorytools package to process the data and print out the positions, velocities, and accelerations of the data
returns a useable trajectory object tr
'''
def processtr(tr):
    center, radiusr = tr.estimate_center_and_radius_from_locations(in_px=True)
    tr.origin_to(center)
    tr.new_length_unit(tr.params['body_length_px'], 'BL')
    tr.new_time_unit(tr.params['frame_rate'], 's')
    print('Positions:')
    print('X range:', np.nanmin(tr.s[...,0]), np.nanmax(tr.s[...,0]), 'BL')
    print('Y range:', np.nanmin(tr.s[...,1]), np.nanmax(tr.s[...,1]), 'BL')
    print('Velcities:')
    print('X range:', np.nanmin(tr.v[...,0]), np.nanmax(tr.v[...,0]), 'BL/s')
    print('Y range:', np.nanmin(tr.v[...,1]), np.nanmax(tr.v[...,1]), 'BL/s')
    print('Accelerations:')
    print('X range:', np.nanmin(tr.a[...,0]), np.nanmax(tr.a[...,0]), 'BL/s^2')
    print('Y range:', np.nanmin(tr.a[...,1]), np.nanmax(tr.a[...,1]), 'BL/s^2')
    pprint(tr.params)
    return tr

tr = processtr(tr)

positions = tr.s*(radius/tr.params['radius'])

velocities = tr.v



'''
function that calculates where a fish can see its reflection given a position and velocity
'''
def plotReflection(xposition, yposition, xvelocity, yvelocity, axis):
    mag = np.sqrt(xposition **2 + yposition**2)
    magv = np.sqrt(xvelocity **2 + yvelocity**2)
    distance = 2*(10 - mag)

    reflection = 0.85

    angles = np.arange(0,6.28,0.01)
    xbound = radius*np.cos(angles) 
    ybound = radius*np.sin(angles) 
    labels=np.zeros(len(angles))
    for i in range(len(angles)):
        magd = np.sqrt((xbound[i]-xposition)**2+(ybound[i]-yposition)**2)
        theta = np.arccos((xbound[i]*(xbound[i]-xposition)+ybound[i]*(ybound[i]-yposition))/(radius*magd))
        phi = np.arccos((xvelocity*(xbound[i]-xposition)+yvelocity*(ybound[i]-yposition))/(magv*magd))
        if theta > 0.85 and theta < 2.29 and phi < 2.958:
            labels[i]=1
    colors = []
    for label in labels:
        if label == 0:
            colors.append('No Reflection')
        else:
            colors.append('Reflection')
    sns.scatterplot(x=xbound, y=ybound, hue=colors, palette={'No Reflection': 'dimgrey', 'Reflection': 'lightgrey'}, ax=axis)
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlim(-radius, radius); axis.set_ylim(-radius, radius)
    axis.legend(loc='upper right')
    axis.quiver(xposition, yposition, xvelocity/magv, yvelocity/magv)
# Function to update the frame
"""def update(frame):
    # Clear the current axis
    plt.clf()
    # Read the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  # Set the frame position
    ret, frame_img = cap.read()
    if ret:
        # Convert from BGR to RGB (OpenCV uses BGR)
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        # Display the frame
        plt.imshow(frame_img_rgb)
        plt.axis('off')  # Turn off axis
        plt.title(f'Frame {frame}')  # Add a title with frame number

# Open the video file
video_path = '/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/1fish15min1fps-half-2_2024-02-28-143116-0000_tracked.avi'
cap = cv2.VideoCapture(video_path)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create the animation
fig = plt.figure()
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50)  # Adjust interval as needed

plt.show()

# Release the video capture object
cap.release()"""

from matplotlib.gridspec import GridSpec

'''
plot the given frame with the reflection visualization and video frame
'''
def update(frame):
    # Clear the current axis
    ax1.clear()
    # Read the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame_img = cap.read()
    if ret:
        # Convert from BGR to RGB (OpenCV uses BGR)
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        # Display the frame
        ax1.imshow(frame_img_rgb, interpolation='nearest')
        ax1.axis('off')  # Turn off axis
        ax1.set_title(f'Frame {frame}')  # Add a title with frame number
    # Clear the current axis
    ax2.clear()
    # Example: plot a sine wave
    x = positions[frame-1][0][0]
    y = -1*positions[frame-1][0][1]
    vx = velocities[frame-1][0][0]
    vy = -1*velocities[frame-1][0][1]
    plotReflection(x, y, vx, vy, ax2)
    ax2.set_title('Reflection Visualization')
        


'''
setup the plot to display the video and reflection visualization
'''
# Open the video file
video_path = video
cap = cv2.VideoCapture(video_path)

# Derive source video properties for optimal export
fps_src = cap.get(cv2.CAP_PROP_FPS)
fps = int(round(fps_src)) if fps_src and fps_src > 0 else 15
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1024

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create the figure sized to the native pixel dimensions (two panels side-by-side)
fig = plt.figure()
fig.set_dpi(OUTPUT_DPI)
fig.set_size_inches((2 * frame_w) / OUTPUT_DPI, frame_h / OUTPUT_DPI)
gs = GridSpec(1, 2, width_ratios=[1, 1])  # 1 row, 2 columns, video takes 3/4 of the width

# Subplot for the video
ax1 = fig.add_subplot(gs[0, 0])

# Subplot for the additional plot
ax2 = fig.add_subplot(gs[0, 1])
plt.tight_layout(pad=0)

# Create the animation for the video
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=500)

# Create the animation for the additional plot
#ani_plot = animation.FuncAnimation(fig, update_plot, frames=total_frames, interval=100)

#print(plt)
#plt.show()

# Release the video capture object


writer = FFMpegWriter(
    fps=fps,
    codec='libx264',
    metadata=dict(artist='Me'),
    bitrate=None,  # let CRF control quality
    extra_args=[
        '-crf', '14',           # lower = higher quality; 14~18 is visually lossless
        '-preset', 'slow',      # better compression at the cost of compute
        '-pix_fmt', 'yuv420p',  # broad compatibility
        '-profile:v', 'high',
        '-movflags', '+faststart'
    ]
)

# Assuming `ani` is your animation object
output_path = 'reflection-visualization-clear-21dpf-1.mp4'
ani.save(output_path, writer=writer, dpi=OUTPUT_DPI)

cap.release()
