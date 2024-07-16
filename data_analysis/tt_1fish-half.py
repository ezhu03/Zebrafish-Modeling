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
if x==7:
    file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
    file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
    file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
    file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
    file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"
    file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half1/trajectories/validated.npy"
    file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half2/trajectories/validated.npy"
    file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half3/trajectories/validated.npy"
    files = [file1,file2,file3,file4,file5,file6,file7,file8]
elif x==14: 
    file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half1/trajectories/validated.npy"
    file2= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half2/trajectories/validated.npy"
    file3= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half3/trajectories/validated.npy"
    file4= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half4/trajectories/validated.npy"
    file5= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half5/trajectories/validated.npy"
    files = [file1,file2,file3,file4, file5]
elif x==21:
    file1 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/trajectories/validated.npy"
    file2 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-2-21dpf/trajectories/validated.npy"
    file3 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-3-21dpf/trajectories/validated.npy"
    file4 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-half-4/trajectories/validated.npy"
    files = [file1,file2,file3,file4]


# Save the merged array to a new .npy file
#np.save("merged_file.npy", merged_array)

def openfile(file, sigma = 1):
    tr = tt.Trajectories.from_idtrackerai(file, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': sigma})
    return tr
trs = []
for file in files:
    tr_temp = openfile(file)
    trs.append(tr_temp)



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

processedpos = []
for temp in trs:
    processed_temp = processtr(temp)
    temppos = processed_temp.s*processed_temp.params['length_unit']*(20/2048)
    processedpos.append(temppos)
    




phalf = np.concatenate(processedpos,axis=0)
print(phalf.shape)
phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])


'''vhalf = np.array([tr1.v, tr2.v, tr3.v, tr4.v, tr5.v])
print(vhalf.shape)
vhalf = np.reshape(vhalf, [vhalf.shape[0]*vhalf.shape[1]*vhalf.shape[2], 2])'''




plt.hist2d(phalf[:, 0], phalf[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015)
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 1 Fish Half Tank')
plt.colorbar(label='Frequency')
plt.show()


phalf= pd.DataFrame(phalf)
phalf.rename(columns={0: 'x', 1: 'y'})
phalf['center'] = np.sqrt(phalf[0]**2 + phalf[1]**2)
