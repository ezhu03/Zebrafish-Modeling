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
x = int(input('dpf: '))

file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"

if x==21:
    file1 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/trajectories/validated.npy"
    file2 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-2-21dpf/trajectories/validated.npy"
    file3 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-3-21dpf/trajectories/validated.npy"

# Save the merged array to a new .npy file
#np.save("merged_file.npy", merged_array)

def openfile(file, sigma = 1):
    tr = tt.Trajectories.from_idtrackerai(file, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': sigma})
    return tr
tr1 = openfile(file1)
tr2 = openfile(file2)
tr3 = openfile(file3)
if x==7:
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
if x==7:
    tr4 = processtr(tr4)
    tr5 = processtr(tr5)


def count_ones(array):
    count = 0
    for num in array:
        if num == 1:
            count += 1
    return count
radius = 10
def plotReflection(xposition, yposition, xvelocity, yvelocity):
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
    return count_ones(labels)/len(labels)


refl_prop = []
correlations = []
pos_arr = []
def border_turning(tr):
#phalf = np.concatenate([tr1.s*(10/tr1.params['radius']), tr2.s*(10/tr2.params['radius']), tr3.s*(10/tr3.params['radius']), tr4.s*(10/tr4.params['radius']), tr5.s*(10/tr5.params['radius'])],axis=0)
#phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])
    pos1= tr.s*(10/tr.params['radius'])

    pos1 = np.array(pos1.reshape(pos1.shape[0],2))

    for pos in pos1:
        pos[1]*=(-1)

    v1 = np.array(tr.v).reshape(tr.v.shape[0],2)

    for vel in v1:
        vel[1]*=(-1)

    norms = np.linalg.norm(v1, axis=1)

    v1 = v1 / norms[:, np.newaxis]

    i=0
    while i<len(pos1)-1:
        dp = np.dot(v1[i],v1[i+1])
        pos_mag = np.sqrt(pos1[i][0]**2+pos1[i][1]**2)
        prop = plotReflection(pos1[i][0],pos1[i][1],v1[i][0],v1[i][1])
        if prop>0 and pos_mag>0:
            refl_prop.append(prop)
            correlations.append(dp)
            pos_arr.append(pos_mag)
        i+=1
        

border_turning(tr1)
border_turning(tr2)
border_turning(tr3)
if x==7:
    border_turning(tr4)
    border_turning(tr5)

data = {'x': refl_prop, 'y':correlations}

df = pd.DataFrame(data)
plt.scatter(x=refl_prop, y=correlations, s=1)
# Scatter plot

bin_edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Bin data by 'Category' using pd.cut() and calculate mean
df['bins'] = pd.cut(df['x'], bins=bin_edges)
print(df)
mean_areas = df.groupby('bins')['x'].agg(['mean', 'std']).reset_index()
mean_values = df.groupby('bins')['y'].agg(['mean', 'std']).reset_index()
mean_values['bin'] = [0.05,0.15,0.25,0.35,0.45]
plt.errorbar(mean_areas['mean'], mean_values['mean'], yerr=mean_values['std'], xerr=mean_areas['std'], fmt='o', capsize=5, label='Mean with Error Bars', color='black', mec='black')
plt.show()
print(mean_values)
print(mean_areas)


