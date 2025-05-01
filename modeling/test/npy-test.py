import numpy as np
import trajectorytools as tt
import matplotlib.pyplot as plt
from pprint import pprint

# Load the .npy file
file = 'data/7dpf/clear/01.25.25/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy'
data = np.load(file,allow_pickle=True)
print(data)
file = 'modeling/data/randomwalk/positions7dpf0.npy'
def openfile(file, sigma = 1):
        tr = tt.Trajectories.from_idtrackerai(file, 
                                        interpolate_nans=True,
                                        smooth_params={'sigma': sigma})
        return tr
tr = openfile(file)
print(tr)
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
processtr(tr)
# Plot the histogram
'''plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of 1D Array Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()'''