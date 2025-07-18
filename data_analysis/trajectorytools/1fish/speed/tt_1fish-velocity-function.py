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
from scipy.optimize import curve_fit

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
    file4 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-half-4/trajectories/validated.npy"


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
tr4 = openfile(file4)
if x==7:
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
if x==7:
    tr5 = processtr(tr5)

v1 = np.array(tr1.v*tr1.params['length_unit']*20/(2048*tr1.params['time_unit']))
v1 = np.reshape(v1,(v1.shape[0],v1.shape[2]))
v2 = np.array(tr2.v*tr2.params['length_unit']*20/(2048*tr2.params['time_unit']))
v2 = np.reshape(v2,(v2.shape[0],v2.shape[2]))
v3 = np.array(tr3.v*tr3.params['length_unit']*20/(2048*tr3.params['time_unit']))
v3 = np.reshape(v3,(v3.shape[0],v3.shape[2]))
v4 = np.array(tr4.v*tr4.params['length_unit']*20/(2048*tr4.params['time_unit']))
v4 = np.reshape(v4,(v4.shape[0],v4.shape[2]))
if x==7:
    v5 = np.array(tr5.v*tr5.params['length_unit']*20/(2048*tr5.params['time_unit']))
    v5 = np.reshape(v5,(v5.shape[0],v5.shape[2]))
velocities = np.vstack((v1,v2,v3,v4))
speeds = []
for velo in velocities:
    speeds.append(np.sqrt(velo[0]**2+velo[1]**2))
plt.hist(speeds, bins = 20)
plt.show()
print(np.mean(speeds),np.std(speeds))
