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

file = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
video = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/1fish15min1fps-half-2_2024-02-28-143116-0000_tracked.avi"

file = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/trajectories/validated.npy"
video = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/1fish15min1fps-half-1-21dpf_2024-03-13-142822-0000_tracked.avi"

# Save the merged array to a new .npy file
#np.save("merged_file.npy", merged_array)



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

positions = tr.s*(10/tr.params['radius'])

velocities = tr.v

radius = 10

def plotReflection(xposition, yposition, xvelocity, yvelocity, axis):
    mag = np.sqrt(xposition **2 + yposition**2)
    magv = np.sqrt(xvelocity **2 + yvelocity**2)
    distance = 2*(10 - mag)

    reflection = 0.85

    angles = np.arange(0,6.28,0.01)
    xbound = 10*np.cos(angles) 
    ybound = 10*np.sin(angles) 
    labels=np.zeros(len(angles))
    for i in range(len(angles)):
        magd = np.sqrt((xbound[i]-xposition)**2+(ybound[i]-yposition)**2)
        theta = np.arccos((xbound[i]*(xbound[i]-xposition)+ybound[i]*(ybound[i]-yposition))/(radius*magd))
        phi = np.arccos((xvelocity*(xbound[i]-xposition)+yvelocity*(ybound[i]-yposition))/(magv*magd))
        if angles[i] > 1.57 and angles[i] < 4.71 and theta > 0.85 and theta < 2.29 and phi < 2.958:
            labels[i]=1
    sns.scatterplot(x=xbound, y=ybound, hue=labels, ax=axis)
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

# Function to update the frame
def update(frame):
    # Clear the current axis
    ax1.clear()
    # Read the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  # Set the frame position
    ret, frame_img = cap.read()
    if ret:
        # Convert from BGR to RGB (OpenCV uses BGR)
        frame_img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        # Display the frame
        ax1.imshow(frame_img_rgb)
        ax1.axis('off')  # Turn off axis
        ax1.set_title(f'Frame {frame}')  # Add a title with frame number

# Function to update the additional plot
def update_plot(frame):
    # Clear the current axis
    ax2.clear()
    # Example: plot a sine wave
    x = positions[frame-1][0][0]
    y = -1*positions[frame-1][0][1]
    vx = velocities[frame-1][0][0]
    vy = -1*velocities[frame-1][0][1]
    plotReflection(x, y, vx, vy, ax2)
    ax2.set_title('Additional Plot')

# Open the video file
video_path = video
cap = cv2.VideoCapture(video_path)

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create the figure and gridspec
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(1, 2, width_ratios=[1, 1])  # 1 row, 2 columns, video takes 3/4 of the width

# Subplot for the video
ax1 = fig.add_subplot(gs[0, 0])

# Subplot for the additional plot
ax2 = fig.add_subplot(gs[0, 1])

# Create the animation for the video
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100)

# Create the animation for the additional plot
ani_plot = animation.FuncAnimation(fig, update_plot, frames=total_frames, interval=100)

plt.show()

# Release the video capture object
cap.release()






"""def openfile(file, sigma = 1):
    tr = tt.Trajectories.from_idtrackerai(file, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': sigma})
    return tr
tr1 = openfile(file1)
tr2 = openfile(file2)
tr3 = openfile(file3)
tr4 = openfile(file4)
tr5 = openfile(file5)



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

tr1 = processtr(tr1)
tr2 = processtr(tr2)
tr3 = processtr(tr3)
tr4 = processtr(tr4)
tr5 = processtr(tr5)

print(tr1.params['radius'])
phalf = np.concatenate([tr1.s*(10/tr1.params['radius']), tr2.s*(10/tr2.params['radius']), tr3.s*(10/tr3.params['radius']), tr4.s*(10/tr4.params['radius']), tr5.s*(10/tr5.params['radius'])],axis=0)
print(phalf.shape)
phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])


'''vhalf = np.array([tr1.v, tr2.v, tr3.v, tr4.v, tr5.v])
print(vhalf.shape)
vhalf = np.reshape(vhalf, [vhalf.shape[0]*vhalf.shape[1]*vhalf.shape[2], 2])'''




plt.hist2d(phalf[:, 0], phalf[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.01)
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 1 Fish Half Tank')
plt.colorbar(label='Frequency')
plt.show()


phalf= pd.DataFrame(phalf)
phalf.rename(columns={0: 'x', 1: 'y'})
phalf['center'] = np.sqrt(phalf[0]**2 + phalf[1]**2)"""
