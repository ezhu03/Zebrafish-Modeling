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

trajectories_path = "/Users/ezhu/Documents/session_25fish_15min_10fps/trajectories/validated.npy"
#trajectories_path = "/Users/eric/Documents/session_25fish_annulus_10fps/trajectories/validated.npy"

tr = tt.Trajectories.from_idtrackerai(trajectories_path, 
                                      interpolate_nans=True,
                                      smooth_params={'sigma': 1})
# Since the arena of the setup was circular and the fish visited the borders of the arena
# we use the estimate_center_and_radius_from_locations to center the trajectories
# in the arena
center, radius = tr.estimate_center_and_radius_from_locations(in_px=True)
tr.origin_to(center)
# In our case we know that the body_length_px is a good estimate for the body length
# since we loaded the trajectories with the method from_idtrackerai this value is 
# stored in the tr.params disctionary
tr.new_length_unit(tr.params['body_length_px'], 'BL')
# Since we loaded the trajectories with the method from_idtrackerai we can 
# use the frame_rate variable stored in the tr.params disctioanry to
# to set the time units to seconds
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

positions = np.reshape(tr.s, (tr.s.shape[0]*tr.s.shape[1] , 2))
velocities = np.reshape(tr.v, (tr.v.shape[0]*tr.v.shape[1] , 2))
acc = np.reshape(tr.a, (tr.a.shape[0]*tr.a.shape[1] , 2))
velocities =np.sqrt( velocities[: , 0]**2 + velocities[: , 1]**2)
acc =np.sqrt( acc[: , 0]**2 + acc[: , 1]**2)
print(velocities)
print(acc)
#sns.histplot(data = velocities, bins = 20, binwidth = 0.2)
#plt.xlim(0,5)
#plt.show()
positions = pd.DataFrame(positions)
velocities = pd.DataFrame(velocities)
acc = pd.DataFrame(acc)
positions.rename(columns={0: 'x', 1: 'y'})
positions['center'] = np.sqrt(positions[0]**2 + positions[1]**2)
print(positions)
sns.histplot(data = positions, x=0, y=1, bins = [20,20], common_norm=True)
plt.show()
sns.histplot(data = positions, x='center', bins = 40, binrange = [0,16])
plt.show()
sns.histplot(data = velocities, x=0, bins = 40, binrange = [0,10])
plt.show()
sns.histplot(data = acc, x=0, bins = 40, binrange = [0,20])
plt.show()