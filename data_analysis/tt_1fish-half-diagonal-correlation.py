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
file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"

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
phalf['center'] = np.sqrt(phalf[0]**2 + phalf[1]**2)
