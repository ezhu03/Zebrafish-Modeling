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
file1 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish1-1/trajectories/validated.npy"
file2 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish2-2/trajectories/validated.npy"
file3 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish3-3/trajectories/validated.npy"

#file1 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish1-2/trajectories/validated.npy"
#file2 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish2-3/trajectories/validated.npy"
#file3 = "/Volumes/Hamilton/Zebrafish/AVI/9.13/session_fish3-1/trajectories/validated.npy"


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
#tr4 = openfile(file4)
#tr5 = openfile(file5)

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
#tr4 = processtr(tr4)
#tr5 = processtr(tr5)

correlations = []
positions = []

def diagonal_correlation(tr):
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



    for i in range(len(pos1)-1):
        dp = np.dot(v1[i],v1[i+1])
        correlations.append(dp)
        positions.append(np.sqrt(pos1[i][0]**2+pos1[i][1]**2))

diagonal_correlation(tr1)
diagonal_correlation(tr2)
diagonal_correlation(tr3)
#diagonal_correlation(tr4)
#diagonal_correlation(tr5)

print(len(correlations), np.mean(correlations), np.std(correlations))

def radial_binning(pos, arr):

    df = pd.DataFrame(np.vstack((pos, arr)).T, columns=['radius','correlation'])

    # Define bin edges
    bin_edges = [0, 2, 4, 8, 10]

    # Bin data by 'Category' using pd.cut() and calculate mean
    df['bins'] = pd.cut(df['radius'], bins=bin_edges)
    mean_values = df.groupby('bins')['correlation'].agg(['mean', 'std'])

    print(mean_values)

radial_binning(positions,correlations)





