'''
Iterate through the dpf values and produce a histogram of the distance from the center of the tank for each fish
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

# Iterating through dpf values and setting up file paths
arr = [7,14,21]
arr = [70,140,210]
outputs = []
for x in arr:
    '''
    We go by convention that 7,14,21 is clear and 70,140,210 is sanded, and 700,1400,2100 is half sanded
    This code is designed to be used with clear data (7,14,21), the sanded and half sanded data is provided but not meant to be used for this code
    '''
    if x==0:
        break
    if x == 7:
        file1 = "data/2.28.24/session_1fish15min1fps-clear-1/trajectories/validated.npy"
        file2 = "data/2.28.24/session_1fish15min1fps-clear-2/trajectories/validated.npy"
        file3 = "data/2.28.24/session_1fish15min1fps-clear-3/trajectories/validated.npy"
        file4 = "data/2.28.24/session_1fish15min1fps-clear-4/trajectories/validated.npy"
        file5 = "data/2.28.24/session_1fish15min1fps-clear-5/trajectories/validated.npy"
        file6 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy"
        file7 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear2/trajectories/validated.npy"
        file8 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]

    if x == 14:
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]


    if x==21:
        file1 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file2 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file3 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file4 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file5 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file6 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file7 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear4/trajectories/validated.npy"
        file8 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]

    if x==70:
        file1 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file2 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file3= "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file4 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded1/trajectories/validated.npy"
        file5 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded2/trajectories/validated.npy"
        file6 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6]

    if x==140:
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]


    if x == 210:
        file1 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file2 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file3 = "data/5.21.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file4 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file5 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file6 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file7 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded4/trajectories/validated.npy"
        file8 = "data/07.16.24/session_1fish-1fps-15min-21dpf-sanded5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]

    # Opening trajectory files
    def openfile(file, sigma = 1):
        tr = tt.Trajectories.from_idtrackerai(file, 
                                        interpolate_nans=True,
                                        smooth_params={'sigma': sigma})
        return tr

    trs = []
    for file in files:
        tr_temp = openfile(file)
        trs.append(tr_temp)

    # Processing trajectory objects
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

    # Extracting positions and velocities
    processedpos = []
    processedvel = []
    for temp in trs:
        processed_temp = processtr(temp)
        temppos = processed_temp.s*processed_temp.params['length_unit']*(20/2048)
        tempvel = processed_temp.v*processed_temp.params['length_unit']*(20/2048)
        processedpos.append(temppos)
        processedvel.append(tempvel)
        
    phalf = np.concatenate(processedpos,axis=0)
    print(phalf.shape)
    phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])

    dist = []

    # Create a dataframe to store position data
    phalf= pd.DataFrame(phalf,columns=['x','y'])
    phalf.rename(columns={'A': 'x', 'B': 'y'})
    phalf['center'] = np.sqrt(phalf['x']**2 + phalf['y']**2)
    phalf['y'] = -1 * phalf['y']
    print(phalf)
    outputs.append(phalf)

# Building combined DataFrame
for i in range(len(outputs)):
    if arr[i] % 10 == 0:
        outputs[i]['Age'] = str(arr[i]/10)+'dpf'
    else:
        outputs[i]['Age'] = str(arr[i])+'dpf'
combined_df = pd.concat(outputs)

# Plotting histogram of distances from center
sns.histplot(data=combined_df, x='center',stat='percent',hue='Age',bins=10,binrange=[0,10],palette=sns.color_palette(palette='YlGnBu_r'),alpha=0.75,multiple='dodge',common_norm=False)
x = arr[0]
if x % 10 == 0:
    
    plt.title('Distance From Center for 1 Fish Sanded Tank Over Time')
else:
    plt.title('Distance From Center for 1 Fish Clear Tank Over Time')
plt.show()
