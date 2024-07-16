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
elif x==14: 
    file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half1/trajectories/validated.npy"
    file2= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half2/trajectories/validated.npy"
elif x==21:
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

if x == 7:
    tr1 = openfile(file1)
    tr2 = openfile(file2)
    tr3 = openfile(file3)
    tr4 = openfile(file4)
    tr5 = openfile(file5)
    tr6 = openfile(file6)
    tr7 = openfile(file7)
    tr8 = openfile(file8)
elif x == 14:
    tr1 = openfile(file1)
    tr2 = openfile(file2)
elif x == 21:
    tr1 = openfile(file1)
    tr2 = openfile(file2)
    tr3 = openfile(file3)


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


if x==7:
    tr1 = processtr(tr1)
    tr2 = processtr(tr2)
    tr3 = processtr(tr3)
    tr4 = processtr(tr4)
    tr5 = processtr(tr5)
    tr6 = processtr(tr6)
    tr7 = processtr(tr7)
    tr8 = processtr(tr8)
elif x==14:
    tr1 = processtr(tr1)
    tr2 = processtr(tr2)
elif x==21:
    tr1 = processtr(tr1)
    tr2 = processtr(tr2)
    tr3 = processtr(tr3)

count_left = []
count_right = []

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
        count = 0
        dp = np.dot(v1[i],v1[i+1])
        pos_mag = np.sqrt(pos1[i][0]**2+pos1[i][1]**2)
        if i<len(pos1)-1 and pos1[i][0]< 0 and pos_mag>8 and dp>0.707:
            while i<len(pos1)-2 and dp>0.707:
                i+=1
                dp = np.dot(v1[i],v1[i+1])
                count+=1
            count_left.append(count)
        if i<len(pos1)-1 and pos1[i][0]> 0 and pos_mag>8 and dp>0.707:
            while i<len(pos1)-2 and pos_mag>8 and dp>0.707:
                i+=1
                dp = np.dot(v1[i],v1[i+1])
                count+=1
            count_right.append(count)
            #print('d')
        i+=1


if x==7:
    border_turning(tr1)
    border_turning(tr2)
    border_turning(tr3)
    border_turning(tr4)
    border_turning(tr5)
    border_turning(tr6)
    border_turning(tr7)
    border_turning(tr8)
elif x==14:
    border_turning(tr1)
    border_turning(tr2)
elif x==21:
    border_turning(tr1)
    border_turning(tr2)
    border_turning(tr3)

print(len(count_left), np.mean(count_left), np.std(count_left))
print(len(count_right), np.mean(count_right), np.std(count_right))






