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
annulus = "/Users/ezhu/Documents/session_25fish_annulus_10fps/trajectories/validated.npy"
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

pcircle = np.reshape(sc, (sc.shape[0]*sc.shape[1] , 2))
vcircle = np.reshape(vc, (vc.shape[0]*vc.shape[1] , 2))
acircle = np.reshape(ac, (ac.shape[0]*ac.shape[1] , 2))
vcircle =np.sqrt( vcircle[: , 0]**2 + vcircle[: , 1]**2)
acircle =np.sqrt( acircle[: , 0]**2 + acircle[: , 1]**2)

pann = np.reshape(sa, (sa.shape[0]*sa.shape[1] , 2))
vann = np.reshape(va, (va.shape[0]*va.shape[1] , 2))
aann = np.reshape(aa, (aa.shape[0]*aa.shape[1] , 2))
vann =np.sqrt( vann[: , 0]**2 + vann[: , 1]**2)
aann =np.sqrt( aann[: , 0]**2 + aann[: , 1]**2)

#sns.histplot(data = velocities, bins = 20, binwidth = 0.2)
#plt.xlim(0,5)
#plt.show()
poscircle = pd.DataFrame(pcircle)
vcircle = pd.DataFrame(vcircle)
acircle = pd.DataFrame(acircle)
poscircle.rename(columns={0: 'x', 1: 'y'})
poscircle['center'] = np.sqrt(poscircle[0]**2 + poscircle[1]**2)

posann = pd.DataFrame(pann)
vann = pd.DataFrame(vann)
aann = pd.DataFrame(aann)
posann.rename(columns={0: 'x', 1: 'y'})
posann['center'] = (trc.params['radius']/tra.params['radius']) * np.sqrt(posann[0]**2 + posann[1]**2)

#sns.histplot(data = poscircle, x=0, y=1, stat="density", bins = [20,20], common_norm=True, palette = sns.color_palette("RdBu", 10))
#plt.show()
plt.hist2d(pcircle[:, 0], pcircle[: , 1], bins=(20, 20), cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.010)
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 25 Fish Clear Circular Tank')
plt.colorbar(label='Frequency')
plt.show()
plt.hist2d(pann[:, 0], pann[: , 1], bins=(20, 20), cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.010)
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 25 Fish Sanded Annulus Tank')
plt.colorbar(label='Frequency')
plt.show()
sns.histplot(data = poscircle, x='center', bins = 40, binrange = [0,17],label = "circle",stat="density",alpha = 0.5, color="#bfd37a")
sns.histplot(data = posann, x='center', bins = 40, binrange = [0,17],label ="annulus",stat="density",alpha=0.5, color="#89b2ae")
plt.title('Position')
plt.legend()
plt.show()
sns.histplot(data = vcircle, x=0, bins = 40, binrange = [0,10],label="circle",stat="density",alpha=0.5, color="#bfd37a")
sns.histplot(data = vann, x=0, bins = 40, binrange = [0,10], label = "annulus",stat="density",alpha=0.5, color="#89b2ae")
plt.legend()
plt.title('Velocity')
plt.show()
sns.histplot(data = acircle, x=0, bins = 40, binrange = [0,20], label = "circle",stat="density",alpha=0.5, color="#bfd37a")
sns.histplot(data = aann, x=0, bins = 40, binrange = [0,20], label="annulus",stat="density",alpha=0.5, color="#89b2ae")
plt.legend()
plt.title('Acceleration')
plt.show()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

# Plot 1
sns.histplot(data=poscircle, x='center', bins=40, binrange=[0, 17], label="circle", stat="density", alpha=0.5, color="#bfd37a", ax=axes[0, 0])
sns.histplot(data=posann, x='center', bins=40, binrange=[0, 17], label="annulus", stat="density", alpha=0.5, color="#89b2ae", ax=axes[0, 1])
axes[0,0].set_ylabel('Position')
axes[0, 0].set_ylim([0,0.6])
axes[0, 1].set_ylim([0,0.6])

# Plot 2
sns.histplot(data=vcircle, x=0, bins=40, binrange=[0, 10], label="circle", stat="density", alpha=0.5, color="#bfd37a", ax=axes[1, 0])
sns.histplot(data=vann, x=0, bins=40, binrange=[0, 10], label="annulus", stat="density", alpha=0.5, color="#89b2ae", ax=axes[1, 1])
axes[1,0].set_ylabel('Velocity')
axes[1, 0].set_ylim([0,1])
axes[1, 1].set_ylim([0,1])

# Plot 3
sns.histplot(data=acircle, x=0, bins=40, binrange=[0, 20], label="circle", stat="density", alpha=0.5, color="#bfd37a", ax=axes[2, 0])
sns.histplot(data=aann, x=0, bins=40, binrange=[0, 20], label="annulus", stat="density", alpha=0.5, color="#89b2ae", ax=axes[2, 1])
axes[2,0].set_ylabel('Acceleration')
axes[2, 0].set_ylim([0,0.5])
axes[2, 1].set_ylim([0,0.5])

# Add a common legend
for ax in axes.flat:
    ax.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()
print(tra.params['radius'])