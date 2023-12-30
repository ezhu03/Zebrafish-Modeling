import os
from pprint import pprint
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats
import seaborn as sns
import pandas as pd
# trajectorytools needs to be installed. To install, 
# pip install trajectorytools or follow the instructions at
# http://www.github.com/fjhheras/trajectorytools
import trajectorytools as tt
import trajectorytools.plot as ttplot
import trajectorytools.socialcontext as ttsocial
def openfile(file, sigma = 1):
    tr = tt.Trajectories.from_idtrackerai(file, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': sigma})
    return tr


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
    print(radius)
    return tr, radius
circle = "/Users/ezhu/Documents/session_25fish_15min_10fps/trajectories/validated.npy"
annulus = "/Users/ezhu/Documents/session_50fish_15minacc_10fps_annulus/trajectories/validated.npy"
trc= openfile(circle)
tra= openfile(annulus)
trc, rc = processtr(trc)
tra, ra  = processtr(tra)
sc = np.delete(trc.s, [3,6,21], axis = 1)
vc = np.delete(trc.v, [3,6,21], axis =1)
ac = np.delete(trc.a, [3,6,21], axis = 1)

sa = np.delete(tra.s, [2], axis = 1)
va = np.delete(tra.v, [2], axis =1)
aa = np.delete(tra.a, [2], axis = 1)

msdc = []
meanc = []
varc = []
for i in range(sc.shape[0]):
    msd = (sc[i,:,0]-sc[0,:,0])**2+(sc[i,:,1]-sc[0,:,1])**2
    msdc.append(msd)
    meanc.append(np.mean(msd))
    varc.append(1.96*np.std(msd)/np.sqrt(sc.shape[1]))
    
#sns.lineplot(x=range(len(meanc)), y=meanc, errorbar = ("ci",95))
#plt.show()
df = pd.DataFrame(msdc)

# Calculate mean and confidence interval for each row
row_means = df.mean(axis=1)
confidence_intervals = df.sem(axis=1)  # Assuming normal distribution

# Create a line plot with 95% confidence interval
plt.figure(figsize=(10, 6))
sns.lineplot(x=df.index, y=row_means, label='Row Mean')
plt.fill_between(df.index, row_means - confidence_intervals, row_means + confidence_intervals, alpha=0.2)

plt.show()

sa = (rc/ra)*sa
msda = []
meana = []
for i in range(sa.shape[0]):
    msd = (sa[i,:,0]-sa[0,:,0])**2+(sa[i,:,1]-sa[0,:,1])**2
    msda.append(msd)
    meana.append(np.mean(msd))

df = pd.DataFrame(msda)

# Calculate mean and confidence interval for each row
row_means = df.mean(axis=1)
confidence_intervals = df.sem(axis=1)  # Assuming normal distribution

# Create a line plot with 95% confidence interval
plt.figure(figsize=(10, 6))
sns.lineplot(x=df.index, y=row_means, label='Row Mean')
plt.fill_between(df.index, row_means - confidence_intervals, row_means + confidence_intervals, alpha=0.2)
plt.show()