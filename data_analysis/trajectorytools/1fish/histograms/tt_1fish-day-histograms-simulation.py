'''
Given clear or sanded tank data, this will iterate through the data and produce histograms for the radial speed, 
radial position, angle to the wall, and the angle to the wall given the initial reflection area.
'''
import os
import sys
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# iterate through the dpf values
x = int(input('dpf : '))
arrnames = ['clear', 'sanded', 'half']

if x==7 or x==14 or x==21:
    blind = input('Blind fish? (Y/N) : ')
    indiv = input('Individual plots? (Y/N) : ')
    if x == 7:
        file1 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear1/trajectories/trajectories.npy"
        file2 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear2/trajectories/trajectories.npy"
        file3 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear3/trajectories/trajectories.npy"
        file4 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear4/trajectories/trajectories.npy"
        file5 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear5/trajectories/trajectories.npy"
        file6 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy"
        file7 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear2/trajectories/validated.npy"
        file8 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear3/trajectories/validated.npy"
        clear = [file1,file2,file3,file4,file5,file6,file7,file8]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file8 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            clear = [file1,file2,file3,file4,file5,file6,file7,file8]
        file1 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded1/trajectories/validated.npy"
        file2 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded2/trajectories/validated.npy"
        file3 = "data/07.02.24/session_1fish-1fps-15min-7dpf-sanded3/trajectories/validated.npy"
        file4 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded1/trajectories/trajectories.npy"
        file5 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded2/trajectories/trajectories.npy"
        file6 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded3/trajectories/trajectories.npy"
        file7 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded4/trajectories/trajectories.npy"
        file8 = "data/06.18.25/session_1fish-1fps-15min-7dpf-sanded5/trajectories/trajectories.npy"
        sanded = [file1,file2,file3,file4,file5,file6,file7,file8]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            sanded = [file1,file2,file3,file5,file6,file7]
        file1 = "data/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
        file2 = "data/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
        file3 = "data/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
        file4 = "data/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
        file5 = "data/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"
        file6 = "data/07.02.24/session_1fish-1fps-15min-7dpf-half1/trajectories/validated.npy"
        file7 = "data/07.02.24/session_1fish-1fps-15min-7dpf-half2/trajectories/validated.npy"
        file8 = "data/07.02.24/session_1fish-1fps-15min-7dpf-half3/trajectories/validated.npy"
        half = [file1, file2, file3, file4, file5, file6, file7, file8]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-half5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-half1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-half2-crispr/trajectories/validated.npy"
            file8 = "data/10.17.24/session_1fish-1fps-15min-7dpf-half3-crispr/trajectories/validated.npy"
            half = [file1, file2, file3, file4, file5, file6, file7, file8]
    elif x == 14:
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear5/trajectories/validated.npy"
        clear = [file1,file2,file3,file4,file5]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            clear = [file1,file2,file3,file5,file6, file7]
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-sanded5/trajectories/validated.npy"
        sanded = [file1,file2,file3,file4,file5]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            sanded = [file1,file2,file3,file4,file5, file6, file7]
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-half5/trajectories/validated.npy"
        half = [file1, file2, file3, file4, file5]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-half5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-half1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-half2-crispr/trajectories/validated.npy"
            half = [file1, file2, file3, file4, file5, file6, file7]
    elif x==21:
        clear = ["modeling/data/boundary/const10radius0boxradius5iter10fish1_15min_21dpf_clear.npy"]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear3-crispr/trajectories/validated.npy"
            clear = [file1,file2,file3]
        sanded = ["modeling/data/boundary/const10radius0boxradius5iter10fish1_15min_21dpf_sanded.npy"]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded3-crispr/trajectories/validated.npy"
            sanded = [file1,file2,file3]
        
        half = ["modeling/data/boundary/const10radius0boxradius5iter10fish1_15min_21dpf_half.npy"]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-half1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-half2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-half3-crispr/trajectories/validated.npy"
            half = [file1, file2, file3]
else:
    sys.exit('No data for ' + str(x) + ' dpf')

# Create empty lists to store outputs
outputs = []
voutputs = []

days = ['7dpf', '14dpf', '21dpf']
arr = [clear, sanded, half]
tt_avg = []
tt_std = []
for files in arr:
    '''
    SET THESE VALUES BEFORE RUNNING THE CODE
    radius: radius of the tank (for reflection calculation)
    times: number of time points to calculate the turning time
    '''
    radius=5
    times=10
    '''
    We go by convention that 7,14,21 is clear and 70,140,210 is sanded, and 700,1400,2100 is half sanded
    This code is designed to be used with clear data (7,14,21), the sanded and half sanded data is provided but not meant to be used for this code
    '''
    

    '''
    function that calculates where a fish can see its reflection given a position and velocity
    '''
    def plotReflection(xposition, yposition, xvelocity, yvelocity):
        mag = np.sqrt(xposition **2 + yposition**2)
        magv = np.sqrt(xvelocity **2 + yvelocity**2)
        distance = 2*(radius - mag)

        reflection = 0.85

        angles = np.arange(0,6.28,0.01)
        xbound = radius*np.cos(angles) 
        ybound = radius*np.sin(angles) 
        labels=np.zeros(len(angles))
        for i in range(len(angles)):
            magd = np.sqrt((xbound[i]-xposition)**2+(ybound[i]-yposition)**2)
            theta = np.arccos((xbound[i]*(xbound[i]-xposition)+ybound[i]*(ybound[i]-yposition))/(radius*magd))
            phi = np.arccos((xvelocity*(xbound[i]-xposition)+yvelocity*(ybound[i]-yposition))/(magv*magd))
            if theta > 0.85 and theta < 2.29 and phi < 2.958:
                labels[i]=1
        return count_ones(labels)/len(labels)
    
    refl_prop = []
    correlations = []
    pos_arr = []

    '''
    converts the position and velocity data into a useable format for analysis
    takes each time point to find the border correlation given the initial border reflection area
    '''
    pos = []
    for file in files:
        data = np.load(file)
        pos.append(data)
    pos = np.concatenate(pos, axis=0)
    print(pos)
    #pos = np.reshape(pos, [pos.shape[0]*pos.shape[1], 2])
    phalf = pd.DataFrame(pos,columns=['x','y'])
    phalf.rename(columns={'A': 'x', 'B': 'y'})
    phalf['r'] = np.sqrt(phalf['x']**2 + phalf['y']**2)
    phalf['theta'] = np.arctan2(-1*phalf['x'],phalf['y'])
    print(phalf)
    outputs.append(phalf)
        
        




    '''
    convert arrays into dataframes and then merge the dataframes together into one large dataframe
    '''
    '''
    plt.hist2d(phalf[:, 0], phalf[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015)
    plt.xlabel('X-bins')
    plt.ylabel('Y-bins')
    plt.title('Heatmap for 1 Fish Half Sanded Tank ' + str(x)+'dpf')
    plt.colorbar(label='Frequency')
    plt.show()
    '''        
    '''
    with the dataframe, we can now plot the data, referencing the title of the plot
    '''

    if x%10 ==0:
        color = 'red'
    else:
        color = 'blue'
    half_df = phalf.copy()
    if(indiv == 'Y'):

        f, ax = plt.subplots(figsize=(10, 8))
        corr = half_df.corr()
        sns.heatmap(corr,cmap=sns.diverging_palette(220, 10, as_cmap=True),vmin=-1.0, vmax=1.0,square=True, ax=ax)
        plt.show()

        plt.figure(figsize=(9, 6))
        #ax = sns.histplot(half_df, x="x", y="y",bins=(10, 10), binrange=[[-10,10],[-10,10]],cmap = sns.color_palette("light:b",as_cmap=True),cbar=True)
        #ax.set_aspect('equal')

        nearwall_df = half_df[half_df['r'] > 8]
        sns.histplot(data=nearwall_df, x='phi',stat='percent',bins=10,binrange=[0,np.pi/2],color=color,alpha=0.5)
        plt.xlabel('Phi')
        plt.ylabel('Percent')
        plt.ylim(0,30)
        if x % 10 == 0:
            n = int(x/10)
            plt.title('Phi Histogram for 1 Fish Sanded Tank ' +str(n) +'dpf')
        else:
            plt.title('Phi Histogram for 1 Fish Clear Tank ' + str(x)+'dpf')
        #plt.colorbar(label='Frequency')
        plt.show()
        sns.histplot(data=half_df, x='theta',stat='percent',bins=20,binrange=[-np.pi,np.pi],color=color,alpha=0.5)
        plt.xlabel('Theta')
        plt.ylabel('Percent')
        plt.ylim(0,12.5)
        if x % 10 == 0:
            n = int(x/10)
            plt.title('Theta Histogram for 1 Fish Sanded Tank ' +str(n) +'dpf')
        else:
            plt.title('Theta Histogram for 1 Fish Clear Tank ' + str(x)+'dpf')
        #plt.colorbar(label='Frequency')
        plt.show()
        sns.histplot(data=half_df, x='spd_r',stat='percent',bins=10,binrange=[0,2.5],color=color,alpha=0.5)
        
        plt.xlabel('Radial Speed')
        plt.ylabel('Percent')
        plt.ylim(0,100)
        if x % 10 == 0:
            n = int(x/10)
            plt.title('Radial Speed Histogram for 1 Fish Sanded Tank ' +str(n) +'dpf')
        else:
            plt.title('Radial Speed Histogram for 1 Fish Clear Tank ' + str(x)+'dpf')
        #plt.colorbar(label='Frequency')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='spd_r', s=5, color=color,alpha=0.5)
        if x % 10 == 0:
            n = int(x/10)
            plt.title('Relationship between Radial Speed and Radial Position for Sanded '+str(n) +'dpf')
        else:
            plt.title('Relationship between Radial Speed and Radial Position for Clear' + str(x)+'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Radial Speed')
        plt.xlim(0,5)
        plt.ylim(0,3)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='spd', s=5, color=color,alpha=0.5)
        if x % 10 == 0:
            n = int(x/10)
            plt.title('Relationship between Speed and Radial Position for Sanded '+str(n) +'dpf')
        else:
            plt.title('Relationship between Speed and Radial Position for Clear' + str(x)+'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Radial Speed')
        plt.xlim(0,5)
        plt.ylim(0,3)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='vr', s=5, color=color,alpha=0.5)
        if x % 10 == 0:
            n = int(x/10)
            plt.title('Relationship between Radial Velocity and Radial Position for Sanded '+str(n) +'dpf')
        else:
            plt.title('Relationship between Radial Velocity and Radial Position for Clear' + str(x)+'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Radial Velocity')
        plt.xlim(0,5)
        plt.ylim(-3,3)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=half_df, x='r', y='phi', s=5,color=color,alpha=0.5)
        if x % 10 == 0:
            n = int(x/10)
            plt.title('Relationship between Wall Angle and Radial Position for Sanded '+str(n) +'dpf')
        else:
            plt.title('Relationship between Wall Angle and Radial Position for Clear' + str(x)+'dpf')
        plt.xlabel('Radial Position')
        plt.ylabel('Angle to Wall')
        plt.xlim(0,10)
        plt.grid(True)
        plt.show()
    #turns = np.array(turns)
    '''plt.hist2d(turns[:, 0], turns[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.04)
    plt.xlabel('X-bins')
    plt.ylabel('Y-bins')
    if x % 10 == 0:
        n = int(x/10)
        plt.title('Heatmap for Turning Location 1 Fish Sanded Tank ' +str(n) +'dpf')
    else:
        plt.title('Heatmap for Turning Location 1 Fish Clear Tank ' + str(x)+'dpf')
    plt.colorbar(label='Frequency')
    plt.show()'''


#plt.rcParams['figure.dpi'] = 300
# KDE overlay (kept as separate)
# KDE overlay as percentage (area = 100%)
from scipy.stats import gaussian_kde

plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'purple']
theta_grid = np.linspace(-np.pi, np.pi, 512)

for output, label, color in zip(outputs, arrnames, colors):
    data = output['theta'].dropna().values
    # circular wrapping for continuity
    wrapped = np.concatenate([data - 2*np.pi, data, data + 2*np.pi])
    kde = gaussian_kde(wrapped, bw_method=0.05)
    vals = kde(theta_grid)
    # normalize to percentage (area under curve = 100)
    area = np.trapz(vals, theta_grid)
    vals_pct = (vals / area) * 100.0
    plt.plot(theta_grid, vals_pct, label=label, alpha=0.4, linewidth=5., color=color)

plt.xlabel('Angular Distribution (θ, radians)')
plt.ylabel('Probability (% per radian)')
plt.xlim([-np.pi, np.pi])
plt.ylim([0,50])
plt.legend()


#plt.savefig('/Users/ezhu/Downloads/angular_overlay.png', dpi=3000, bbox_inches='tight')
plt.show()

# Smoothed Circular plot overlay using KDE, scaled to percentage
from scipy.stats import gaussian_kde

plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

n_points = 360
theta_grid = np.linspace(-np.pi, np.pi, n_points)

for output, label, color in zip(outputs, arrnames, colors):
    data = output['theta'].dropna().values
    wrapped = np.concatenate([data - 2*np.pi, data, data + 2*np.pi])
    kde = gaussian_kde(wrapped, bw_method=0.05)
    vals = kde(theta_grid)
    area = np.trapz(vals, theta_grid)
    vals_pct = (vals / area) * 100.0
    ax.plot(theta_grid, vals_pct, label=label, linewidth=5, color=color, alpha=0.4)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_yticks([])  # keep clean
ax.set_ylabel('Probability (% per radian)', labelpad=20)
ax.set_ylim([0, 50])
plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.05))
plt.tight_layout()

#plt.savefig('/Users/ezhu/Downloads/angular_overlay_circular.png', dpi=3000)
plt.show()

# Polar histogram (rose plot) showing percent per bin
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)

# choose number of angular bins
n_bins = 24
edges = np.linspace(-np.pi, np.pi, n_bins + 1)
width = edges[1] - edges[0]
print(outputs[0]['theta'].dropna().values)
for i, output in enumerate(outputs):
    vals = output['theta'].dropna().values
    print(vals)
    if vals.size == 0:
        continue
    # Use weights so the histogram reports percentage per bin (summing to 100%)
    weights = np.full_like(vals, 100.0 / vals.size, dtype=float)
    ax.hist(
        vals,
        bins=edges,
        weights=weights,
        histtype='bar',
        linewidth=2,
        edgecolor=colors[i],
        facecolor='none',
        fill=False,
        label=arrnames[i],
        alpha=0.4)

ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
# radial labels as percentages
ax.set_rlabel_position(225)
ax.set_ylim([0, 10])  # set radial limits
ax.set_yticks([5, 10])
ax.set_yticklabels(['5%', '10%'])
ax.set_ylabel('Percentage (%)', labelpad=40)

plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1.05))
plt.tight_layout()
plt.savefig('/Users/ezhu/Downloads/angular_histogram_circular.png', dpi=3000)
plt.show()