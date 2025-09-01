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

#file = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
#video = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/1fish15min1fps-half-2_tracked.avi"

file = "data/07.16.24/session_1fish-1fps-15min-21dpf-half1/trajectories/validated.npy"
video = "data/07.16.24/session_1fish-1fps-15min-21dpf-half1/1fish-1fps-15min-21dpf-half1_2024-07-16-122129-0000_tracked.avi"

# Save the merged array to a new .npy file
#np.save("merged_file.npy", merged_array)
radius = 5


# Step 1: Install Required Packages
# pip install matplotlib ffmpeg

# Step 2: Import Libraries

# Step 3: Create a Function to Update the Frame
import cv2

def openfile(file, sigma = 1):
    tr = tt.Trajectories.from_idtrackerai(file, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': sigma})
    return tr
tr = openfile(file)



def processtr(tr):
    center, radius = tr.estimate_center_and_radius_from_locations(in_px=True)
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



def plotReflection(xposition, yposition, xvelocity, yvelocity, axis):
    mag = np.sqrt(xposition **2 + yposition**2)
    magv = np.sqrt(xvelocity **2 + yvelocity**2)
    distance = 2*(radius - mag)

    reflection = 0.85

    angles = np.arange(0,6.28,0.01)
    xbound = radius*np.cos(angles) 
    ybound = radius*np.sin(angles) 
    labels=np.zeros(len(angles))
    for i in range(len(angles)):
        magd = np.sqrt((xbound[i]-xposition)**2+(ybound[i]-yposition)**2)
        theta = np.arccos((xbound[i]*(xbound[i]-xposition)+ybound[i]*(ybound[i]-yposition))/(radius*magd))
        phi = np.arccos((xvelocity*(xbound[i]-xposition)+yvelocity*(ybound[i]-yposition))/(magv*magd))
        if angles[i] > 1.57 and angles[i] < 4.71 and theta > 0.85 and theta < 2.29 and phi < 2.958:
            labels[i]=1
        elif angles[i] < 1.57 or angles[i] > 4.71:
            labels[i]=2
    colors = []
    for label in labels:
        if label == 0:
            colors.append('No Reflection')
        elif label == 1:
            colors.append('Reflection')
        elif label == 2:
            colors.append('Sanded')
    sns.scatterplot(x=xbound, y=ybound, hue=colors, palette={'No Reflection': 'dimgrey', 'Reflection': 'lightgrey', 'Sanded': 'darkred'}, ax=axis)
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    axis.quiver(xposition, yposition, xvelocity/1024, yvelocity/1024, angles='xy', scale_units='xy', scale=1, width=0.004)

# Function to update the frame
def update(frame):
    # Clear the axis but preserve limits/aspect after we draw
    ax1.clear()

    # Seek to requested frame and read it
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame_img = cap.read()
    if ret:
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        # Map image extent to tank coords so overlay aligns with arena ([-radius, radius])
        ax1.imshow(frame_img_rgb, extent=[-radius, radius, -radius, radius])
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Frame {frame}')

    # Get fish state for this frame (flip y to match existing convention)
    x = positions[frame-1][0][0]
    y = -1 * positions[frame-1][0][1]
    vx = velocities[frame-1][0][0]
    vy = -1 * velocities[frame-1][0][1]

    # Overlay reflection directly on top of the video frame (reuse existing plotReflection)
    plotReflection(x, y, vx, vy, ax1)

    # Keep limits fixed so overlay doesn't rescale between frames
    ax1.set_xlim(-radius, radius)
    ax1.set_ylim(-radius, radius)


# Open the video file
video_path = video
cap = cv2.VideoCapture(video_path)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a single square figure; overlay reflection directly on the video
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(1, 1, 1)

# Create the animation for the video
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=500)

# Create the animation for the additional plot
#ani_plot = animation.FuncAnimation(fig, update_plot, frames=total_frames, interval=100)

#print(plt)
#plt.show()

# Release the video capture object


writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Assuming `ani` is your animation object
ani.save('data_analysis/trajectorytools/1fish/reflectionvisualization/reflection-visualization-overlay-half-21dpf-1.mp4', writer=writer)

cap.release()
