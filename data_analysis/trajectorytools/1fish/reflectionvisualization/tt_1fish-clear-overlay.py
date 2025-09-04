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
OUTPUT_DPI = 200  # high DPI for sharper output
'''
SET THESE VALUES BEFORE RUNNING THE CODE
radius: radius of the tank (for reflection calculation)
file: path to the numpy file containing the position and velocity data
video: path to the tracked video file to be displayed
'''

radius = 5

file = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
video = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear1/1fish-1fps-15min-21dpf-clear1_2024-05-21-134623-0000_tracked.avi"

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

positions = tr.s*(radius*tr.params['body_length_px']/1024)

velocities = tr.v*radius*tr.params['body_length_px']/(1024*tr.params['frame_rate'])



'''
function that calculates where a fish can see its reflection given a position and velocity
'''
def plotReflection(xposition, yposition, xvelocity, yvelocity, axis):
    # Compute magnitudes with small epsilon to avoid division by zero
    mag = np.sqrt(xposition ** 2 + yposition ** 2)
    magv = np.sqrt(xvelocity ** 2 + yvelocity ** 2) + 1e-9

    # Precompute circular boundary points
    angles = np.arange(0, 2*np.pi, 0.01)
    xbound = radius * np.cos(angles)
    ybound = radius * np.sin(angles)

    # Label boundary points that satisfy reflection criteria
    labels = np.zeros(len(angles))
    for i in range(len(angles)):
        magd = np.sqrt((xbound[i]-xposition)**2 + (ybound[i]-yposition)**2) + 1e-9
        # Angle between boundary normal (radial) and line-of-sight to fish
        theta = np.arccos(
            (xbound[i]*(xbound[i]-xposition) + ybound[i]*(ybound[i]-yposition)) / (radius * magd)
        )
        # Angle between fish velocity and line-of-sight to boundary point
        phi = np.arccos(
            (xvelocity*(xbound[i]-xposition) + yvelocity*(ybound[i]-yposition)) / (magv * magd)
        )
        if 0.85 < theta < 2.29 and phi < 2.958:
            labels[i] = 1

    # Draw boundary as a thin outline on top of the frame
    axis.plot(xbound, ybound, linewidth=1, zorder=2)

    # Overlay the reflection mask directly on the video (no separate subplot)
    # Non-reflective = darker, reflective = lighter, with transparency so the video is visible
    non_reflect_idx = labels == 0
    reflect_idx = labels == 1
    axis.scatter(xbound[non_reflect_idx], ybound[non_reflect_idx], s=4, alpha=0.35, c='dimgrey', label='No Reflection', zorder=3)
    axis.scatter(xbound[reflect_idx], ybound[reflect_idx], s=8, alpha=0.7, c='lightgrey', label='Reflection', zorder=4)
    axis.legend(loc='upper right', frameon=True)

    # Draw velocity direction vector (length reflects velocity magnitude) at fish location
    axis.quiver(xposition, yposition, xvelocity, yvelocity, angles='xy', scale_units='xy', scale=1, width=0.004, zorder=5)


'''
plot the given frame with the reflection visualization and video frame
'''
def update(frame):
    # Clear the axis but preserve limits/aspect after we draw
    ax1.clear()

    # Set video to requested frame and read it
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, frame_img = cap.read()
    if ret:
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        # Draw the frame with coordinates matching tank coordinates so overlays align
        # Map image extent to [-radius, radius] in both x and y; ensure equal aspect
        ax1.imshow(frame_img_rgb, extent=[-radius, radius, -radius, radius], interpolation='nearest')
        ax1.set_aspect('equal')
        ax1.axis('off')
        #ax1.set_title(f'Frame {frame}')

    # Get fish state for this frame (flip y to match earlier convention)
    x = positions[frame-1][0][0]
    y = -1 * positions[frame-1][0][1]
    vx = velocities[frame-1][0][0]  # Convert to BL/s
    vy = -1 * velocities[frame-1][0][1]  # Convert to BL/s

    # Overlay reflection directly on top of video
    plotReflection(x, y, vx, vy, ax1)

    # Keep limits fixed to tank coords so overlay does not rescale
    ax1.set_xlim(-radius, radius)
    ax1.set_ylim(-radius, radius)


'''
setup the plot to display the video and reflection visualization
'''
# Open the video file
video_path = video
cap = cv2.VideoCapture(video_path)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Derive source video properties for optimal export
fps_src = cap.get(cv2.CAP_PROP_FPS)
fps = int(round(fps_src)) if fps_src and fps_src > 0 else 15
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1024

# Create a single-axes figure sized to the video dimensions
fig = plt.figure()
fig.set_dpi(OUTPUT_DPI)
fig.set_size_inches(frame_w / OUTPUT_DPI, frame_h / OUTPUT_DPI)
ax1 = fig.add_subplot(1, 1, 1)
plt.tight_layout(pad=0)

# Create the animation for the video
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=500)

# Release the video capture object


writer = FFMpegWriter(
    fps=fps,
    codec='libx264',
    metadata=dict(artist='Me'),
    bitrate=None,  # let CRF control quality
    extra_args=[
        '-crf', '14',           # lower = higher quality; 14–18 is visually lossless
        '-preset', 'slow',      # better compression at the cost of compute
        '-pix_fmt', 'yuv420p',  # broad compatibility
        '-profile:v', 'high',
        '-movflags', '+faststart'
    ]
)

# Assuming `ani` is your animation object
output_path = 'data_analysis/trajectorytools/1fish/reflectionvisualization/reflection-visualization-overlay-clear-21dpf-1.mp4'
ani.save(output_path, writer=writer, dpi=OUTPUT_DPI)
output2_path = '/Users/ezhu/Downloads/reflection-visualization-overlay-clear-21dpf-1.mp4'
ani.save(output2_path, writer=writer, dpi=OUTPUT_DPI)


cap.release()
