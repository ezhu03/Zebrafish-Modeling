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
file11 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish1-1/trajectories/validated.npy"
file22 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish2-2/trajectories/validated.npy"
file33 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish3-3/trajectories/validated.npy"
file12 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish1-2/trajectories/validated.npy"
file23 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish2-3/trajectories/validated.npy"
file31 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish3-1/trajectories/validated.npy"
file13 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish1-3/trajectories/validated.npy"
file21 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish2-1/trajectories/validated.npy"
file32 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish3-2/trajectories/validated.npy"

def openfile(file, sigma = 1):
    tr = tt.Trajectories.from_idtrackerai(file, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': sigma})
    return tr
tr11 = openfile(file11)
tr22 = openfile(file22)
tr12 = openfile(file12)
tr23 = openfile(file23)
tr31 = openfile(file31)
tr13 = openfile(file13)
tr21 = openfile(file21)
tr32 = openfile(file32)


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

tr11 = processtr(tr11)
tr22 = processtr(tr22)
tr12 = processtr(tr12)
tr23 = processtr(tr23)
tr31 = processtr(tr31)
tr13 = processtr(tr13)
tr21 = processtr(tr21)
tr32 = processtr(tr32)

pclear = np.array([tr11.s, tr22.s])
print(pclear.shape)
pclear = np.reshape(pclear, [pclear.shape[0]*pclear.shape[1]*pclear.shape[2], 2])

psand = np.array([tr12.s, tr23.s, tr31.s])
print(psand.shape)
psand = np.reshape(psand, [psand.shape[0]*psand.shape[1]*psand.shape[2], 2])

phalf = np.array([tr13.s, tr21.s, tr32.s])
print(phalf.shape)
phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1]*phalf.shape[2], 2])

vclear = np.array([tr11.v, tr22.v])
print(vclear)
print(vclear.shape)
vclear = np.reshape(vclear, [vclear.shape[0]*vclear.shape[1]*vclear.shape[2], 2])

vsand = np.array([tr12.v, tr23.v, tr31.v])
print(vsand.shape)
vsand = np.reshape(vsand, [vsand.shape[0]*vsand.shape[1]*vsand.shape[2], 2])

vhalf = np.array([tr13.v, tr21.v, tr32.v])
print(vhalf.shape)
vhalf = np.reshape(vhalf, [vhalf.shape[0]*vhalf.shape[1]*vhalf.shape[2], 2])

# Create a 2D histogram
print(vclear)
vclear = pd.DataFrame(vclear)
vclear['norm'] = np.sqrt(vclear[0]**2 + vclear[1]**2)

vsand = pd.DataFrame(vsand)
vsand['norm'] = np.sqrt(vsand[0]**2 + vsand[1]**2)

vhalf = pd.DataFrame(vhalf)
vhalf['norm'] = np.sqrt(vhalf[0]**2 + vhalf[1]**2)
plt.hist2d(pclear[:, 0], pclear[: , 1], bins=(20, 20), cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.02)
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 1Fish Clear Tank')
plt.colorbar(label='Frequency')
plt.show()

plt.hist2d(psand[:, 0], psand[: , 1], bins=(20, 20), cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.02)
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 1 Fish Sanded Tank')
plt.colorbar(label='Frequency')
plt.show()

plt.hist2d(phalf[:, 0], phalf[: , 1], bins=(20, 20), cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.02)
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 1 Fish Half Tank')
plt.colorbar(label='Frequency')
plt.show()

pclear= pd.DataFrame(pclear)
pclear.rename(columns={0: 'x', 1: 'y'})
pclear['center'] = np.sqrt(pclear[0]**2 + pclear[1]**2)
psand= pd.DataFrame(psand)
psand.rename(columns={0: 'x', 1: 'y'})
psand['center'] = np.sqrt(psand[0]**2 + psand[1]**2)
phalf= pd.DataFrame(phalf)
phalf.rename(columns={0: 'x', 1: 'y'})
phalf['center'] = np.sqrt(phalf[0]**2 + phalf[1]**2)

#sns.histplot(data = vclear, x='norm', bins = 40, binrange = [0,120])
#plt.show()


# Plot the first histplot in the first subplot
sns.histplot(data = pclear, x='center', stat = "density", bins = 40, binrange = [0,16], alpha = 0.5, label="Clear", color = "#bfd37a")
sns.histplot(data = phalf, x='center', stat="density", bins = 40, binrange = [0,16], alpha = 0.5, label = "Half", color = "#89b2ae")
# Plot the second histplot in the second subplot
sns.histplot(data = psand, x='center', stat="density", bins = 40, binrange = [0,16], alpha = 0.5, label = "Sanded", color = "#5b818e")


plt.legend()
# Show the plots
plt.show()

# Plot the first histplot in the first subplot
sns.histplot(data = vclear, x='norm', stat = "density", bins = 40, binrange = [0,100], alpha = 0.5, label="Clear", color = "#bfd37a")

# Plot the second histplot in the second subplot
sns.histplot(data = vsand, x='norm', stat="density", bins = 40, binrange = [0,100], alpha = 0.5, label = "Sanded", color = "#5b818e")
plt.legend()
# Show the plots
plt.show()

min_y = 0
max_y = 1
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot the first histplot in the first subplot
sns.histplot(data=pclear, x='center', stat="density", bins=40, binrange=[0, 16], alpha=0.5, label="Clear", color="#bfd37a", ax=axes[0])
axes[0].set_title('Clear')

# Plot the second histplot in the second subplot
sns.histplot(data=psand, x='center', stat="density", bins=40, binrange=[0, 16], alpha=0.5, label="Sanded", color="#5b818e", ax=axes[2])
axes[2].set_title('Sanded')
sns.histplot(data=phalf, x='center', stat="density", bins=40, binrange=[0, 16], alpha=0.5, label="Half", color="#89b2ae", ax=axes[1])
axes[1].set_title('Half')

axes[0].set_ylim(min_y, max_y)
axes[1].set_ylim(min_y, max_y)
axes[2].set_ylim(min_y, max_y)

# Add a common legend
for ax in axes:
    ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

min_y = 0
max_y = 0.1
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot the first histplot in the first subplot
sns.histplot(data=vclear, x='norm', stat="density", bins=40, binrange=[0, 100], alpha=0.5, label="Clear", color="#bfd37a", ax=axes[0])
axes[0].set_title('Clear')

# Plot the second histplot in the second subplot
sns.histplot(data=vsand, x='norm', stat="density", bins=40, binrange=[0, 100], alpha=0.5, label="Sanded", color="#89b2ae", ax=axes[2])
axes[2].set_title('Sanded')

sns.histplot(data=vhalf, x='norm', stat="density", bins=40, binrange=[0, 100], alpha=0.5, label="Half", color="#89b2ae", ax=axes[1])
axes[1].set_title('Half')
axes[0].set_ylim(min_y, max_y)
axes[1].set_ylim(min_y, max_y)
axes[2].set_ylim(min_y, max_y)

# Add a common legend
for ax in axes:
    ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()