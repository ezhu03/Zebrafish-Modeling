'''
Given clear or sanded tank data, this will iterate through the data and produce histograms for the radial speed, 
radial position, angle to the wall, and the angle to the wall given the initial reflection area. To produce the plots
we bootstraps the data to get a more accurate representation of the data.
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
# iterate through the dpf values
val = input('Sanded or Clear? : ')
if val == 'Sanded':
    arr = [70,140,210]
elif val == 'Clear':
    arr = [7,14,21]
indiv = input('Individual plots? (Y/N) : ')
blind = input('Blind fish? (Y/N) : ')

outputs = []
voutputs = []

days = ['7dpf', '14dpf', '21dpf']

tt_avg = []
tt_std = []
for x in arr:
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
    if x==0:
        break
    if x == 7:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy"
        file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-clear2/trajectories/validated.npy"
        file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-clear3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file8 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file4,file5,file6,file7,file8]

    if x == 14:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "/Volumes/Hamilton/Zebrafish/AVI/11.13.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "/Volumes/Hamilton/Zebrafish/AVI/11.13.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file5,file6, file7]


    if x==21:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-clear4/trajectories/validated.npy"
        file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-clear5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-clear3-crispr/trajectories/validated.npy"
            files = [file1,file2,file3]

    if x==70:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file3= "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded1/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded2/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6]
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            file8 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-sanded3-crispr/trajectories/validated.npy"

            files = [file1,file2,file3,file5,file6,file7]

    if x==140:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "/Volumes/Hamilton/Zebrafish/AVI/11.13.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "/Volumes/Hamilton/Zebrafish/AVI/11.13.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file4,file5, file6, file7]

    if x == 210:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-sanded4/trajectories/validated.npy"
        file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-sanded5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-sanded3-crispr/trajectories/validated.npy"
            files = [file1,file2,file3]



    
    def count_ones(array):
        count = 0
        for num in array:
            if num == 1:
                count += 1
        return count
    # Save the merged array to a new .npy file
    #np.save("merged_file.npy", merged_array)
    '''
    opens the provided numpy file to be processed by trajectorytools
    '''
    def openfile(file, sigma = 1):
        tr = tt.Trajectories.from_idtrackerai(file, 
                                        interpolate_nans=True,
                                        smooth_params={'sigma': sigma})
        return tr
    '''
    opens each provided file and processes the data to be used for the analysis
    '''
    trs = []
    for file in files:
        tr_temp = openfile(file)
        trs.append(tr_temp)

    '''
    uses the trajectorytools package to process the data and print out the positions, velocities, and accelerations of the data
    returns a useable trajectory object tr
    '''
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
    
    '''
    function that calculates where a fish can see its reflection given a position and velocity
    '''
    def plotReflection(xposition, yposition, xvelocity, yvelocity):
        mag = np.sqrt(xposition **2 + yposition**2)
        magv = np.sqrt(xvelocity **2 + yvelocity**2)
        distance = 2*(radius - mag)

        reflection = 0.85

        angles = np.arange(0,6.28,0.01)
        xbound = 10*np.cos(angles) 
        ybound = 10*np.sin(angles) 
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
    #times=10

    '''
    converts the position and velocity data into a useable format for analysis
    takes each time point to find the border correlation given the initial border reflection area
    '''
    def border_turning(tr):
    #phalf = np.concatenate([tr1.s*(10/tr1.params['radius']), tr2.s*(10/tr2.params['radius']), tr3.s*(10/tr3.params['radius']), tr4.s*(10/tr4.params['radius']), tr5.s*(10/tr5.params['radius'])],axis=0)
    #phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])
        pos1= tr.s*tr.params['length_unit']*(2*radius/2048)

        pos1 = np.array(pos1.reshape(pos1.shape[0],2))

        for pos in pos1:
            pos[1]*=(-1)

        v1 = np.array(tr.v).reshape(tr.v.shape[0],2)

        for vel in v1:
            vel[1]*=(-1)

        norms = np.linalg.norm(v1, axis=1)

        v1 = v1 / norms[:, np.newaxis]

        i=0
        #times = 20
        while i<len(pos1):
            prop = plotReflection(pos1[i][0],pos1[i][1],v1[i][0],v1[i][1])

            #if prop>0 and pos_mag>0:
            refl_prop.append(prop)
            i+=1

    '''
    produce arrays with the positions and velocities of the fish
    '''
    processedpos = []
    processedvel = []
    for temp in trs:
        processed_temp = processtr(temp)
        border_turning(processed_temp)
        temppos = processed_temp.s*processed_temp.params['length_unit']*(2*radius/2048)
        tempvel = processed_temp.v*(processed_temp.params['length_unit']/processed_temp.params['time_unit'])*(2*radius/2048)
        processedpos.append(temppos)
        processedvel.append(tempvel)
    
    phalf = np.concatenate(processedpos,axis=0)
    print(phalf.shape)
    phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])


    vhalf = np.concatenate(processedvel,axis=0)
    print(vhalf.shape)
    vhalf = np.reshape(vhalf, [vhalf.shape[0]*vhalf.shape[1], 2])
    dist = []
    



    '''plt.hist2d(phalf[:, 0], phalf[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015
            )
    plt.xlabel('X-bins')
    plt.ylabel('Y-bins')
    if x % 10 == 0:
        n = int(x/10)
        plt.title('Heatmap for 1 Fish Sanded Tank ' +str(n) +'dpf')
    else:
        plt.title('Heatmap for 1 Fish Clear Tank ' + str(x)+'dpf')
    plt.colorbar(label='Frequency')
    plt.show()'''
    center = (0, 0)

    # Create circle
    theta = np.linspace(0, 2 * np.pi, 300)
    xc = center[0] + radius * np.cos(theta)
    yc = center[1] + radius * np.sin(theta)
    plt.figure(figsize=(3, 3))
    plt.plot(xc, yc, label=f'Circle with radius {radius}')
    plt.hist2d(phalf[:, 0], -1*phalf[: , 1], bins=(10, 10), range=[[-1*radius,radius],[-1*radius,radius]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.05
            )
    #plt.xlabel('X-bins')
    #plt.ylabel('Y-bins')
    '''if x % 10 == 0:
        n = int(x/10)
        plt.title('Heatmap for 1 Fish Sanded Tank ' +str(n) +'dpf')
    else:
        plt.title('Heatmap for 1 Fish Clear Tank ' + str(x)+'dpf')'''
    #plt.colorbar(label='Frequency')
    plt.show()



    '''
    convert arrays into dataframes and then merge the dataframes together into one large dataframe
    '''
    phalf= pd.DataFrame(phalf,columns=['x','y'])
    phalf.rename(columns={'A': 'x', 'B': 'y'})
    phalf['r'] = np.sqrt(phalf['x']**2 + phalf['y']**2)
    phalf['y'] = -1 * phalf['y']
    phalf['theta'] = np.arctan2(-1*phalf['x'],phalf['y'])
    print(phalf)
    #outputs.append(phalf)

    vhalf= pd.DataFrame(vhalf,columns=['vx','vy'])
    vhalf.rename(columns={'A': 'vx', 'B': 'vy'})
    vhalf['spd'] = np.sqrt(vhalf['vx']**2 + vhalf['vy']**2)
    vhalf['vy'] = -1 * vhalf['vy']
    vhalf['vtheta'] = np.arctan2(-1*vhalf['vx'],vhalf['vy'])
    
    print(vhalf)
    #voutputs.append(vhalf)

    half_df =  pd.concat([phalf, vhalf], axis=1)
    print(half_df)
    half_df['vrx'] = half_df['vx']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    half_df['vry'] = half_df['vy']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    half_df['vr'] = half_df['spd']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    half_df['spd_r'] = np.abs(half_df['vr'])
    half_df['vtx'] = half_df['vx']-half_df['vrx']
    half_df['vty'] = half_df['vy']-half_df['vry']
    half_df['spd_t'] = np.sqrt(half_df['vtx']**2+half_df['vty']**2)
    phi_temp = np.arccos((-np.cos(half_df['theta'])*half_df['vx']-np.sin(half_df['theta']*half_df['vy']))/half_df['spd'])
    print(phi_temp)
    for i in range(len(phi_temp)):
        if phi_temp[i] > np.pi/2:
            phi_temp[i] = np.pi-phi_temp[i]
    half_df['phi'] = phi_temp
    half_df['refl_prop'] = refl_prop
    print(half_df['refl_prop'])
    half_df = pd.DataFrame(half_df.sample(n=10000, replace=True, random_state=0))
    half_df.reset_index(drop=False, inplace=True)
    print(half_df)

    '''
    with the dataframe, we can now plot the data, referencing the title of the plot
    '''
    if x%10 ==0:
        color = 'red'
    else:
        color = 'blue'
    
    center = (0, 0)

    # Create circle
    theta = np.linspace(0, 2 * np.pi, 300)
    xc = center[0] + radius * np.cos(theta)
    yc = center[1] + radius * np.sin(theta)
    plt.figure(figsize=(3, 3))
    plt.plot(xc, yc, label=f'Circle with radius {radius}')
    plt.hist2d(half_df['x'], half_df['y'], bins=(10, 10), range=[[-1*radius,radius],[-1*radius,radius]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.05
            )
    #plt.xlabel('X-bins')
    #plt.ylabel('Y-bins')
    '''if x % 10 == 0:
        n = int(x/10)
        plt.title('Heatmap for 1 Fish Sanded Tank ' +str(n) +'dpf')
    else:
        plt.title('Heatmap for 1 Fish Clear Tank ' + str(x)+'dpf')'''
    #plt.colorbar(label='Frequency')
    plt.show()


    turn_times = []
    turns = []
    temp_counter = 0
    min = 0
    max = np.pi
    for index, row in half_df.iterrows():
        if temp_counter == 0 and row['r'] > 0.8*radius and row['phi'] < 0.4:
            max = row['phi'] + np.pi/2
            temp_counter+=1
        elif temp_counter == 0 and row['r'] > 0.8*radius and row['phi'] > np.pi - 0.4:
            min = row['phi'] - np.pi/2
            temp_counter+=1
        elif temp_counter > 0 and row['phi'] >= min and row['phi'] <= max:
            temp_counter+=1
        elif temp_counter > 0:
            turns.append([row['x'],row['y']])
            turn_times.append(temp_counter)
            temp_counter = 0
            min = 0
            max = np.pi
    tt_avg.append(np.mean(turn_times))
    tt_std.append(np.std(turn_times))

    
             
    '''
    with the dataframe, we can now plot the data, referencing the title of the plot
    '''

    if x%10 ==0:
        color = 'red'
    else:
        color = 'blue'

    if(indiv == 'Y'):

        f, ax = plt.subplots(figsize=(10, 8))
        corr = half_df.corr()
        sns.heatmap(corr,cmap=sns.diverging_palette(220, 10, as_cmap=True),vmin=-1.0, vmax=1.0,square=True, ax=ax)
        plt.show()

        plt.figure(figsize=(9, 6))
        #ax = sns.histplot(half_df, x="x", y="y",bins=(10, 10), binrange=[[-10,10],[-10,10]],cmap = sns.color_palette("light:b",as_cmap=True),cbar=True)
        #ax.set_aspect('equal')

        nearwall_df = half_df[half_df['r'] > 0.8*radius]
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
        sns.histplot(data=half_df, x='spd_r',stat='percent',bins=20,binrange=[0,2.5],color=color,alpha=0.5)
        
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
        plt.xlim(0,radius)
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
        plt.xlim(0,radius)
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
        plt.xlim(0,radius)
        plt.grid(True)
        plt.show()
    turns = np.array(turns)
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

    outputs.append(half_df)
dpf = [7,14,21]
plt.errorbar(x=dpf,y=tt_avg,yerr = tt_std,fmt='o',color=color)
plt.xlabel('Days Post Fertilization')
plt.ylabel('Turning Time Along Wall')
if x % 10 == 0:
    n = int(x/10)
    plt.title('Mean Turning Time over dpf for Sanded')
else:
    plt.title('Mean Turning Time over dpf for Clear')
plt.show()
for i in range(len(outputs)):
    if arr[i] % 10 == 0:
        outputs[i]['Age'] = str(arr[i]/10)+'dpf'
    else:
        outputs[i]['Age'] = str(arr[i])+'dpf'
combined_df = pd.concat(outputs)
'''colors=sns.color_palette(palette='YlGnBu_r')
for i in range(len(outputs)):
    #plt.hist(data=outputs[i],x='center',density=True,bins=10,range=[0,10],color=(colors[i], 0.3))
    if arr[0]%10==0:
        sns.histplot(data=outputs[i], x='center',stat='percent',bins=20,binrange=[0,10],color=colors[2*i+1],alpha=1-i/3, label = str(arr[i]/10) + 'dpf')
    else:
        sns.histplot(data=outputs[i], x='center',stat='percent',bins=20,binrange=[0,10],color=colors[2*i+1],alpha=1-i/3, label = str(arr[i]) + 'dpf')'''
sns.histplot(data=combined_df, x='r',stat='percent',hue='Age',bins=10,binrange=[0,10],palette=sns.color_palette(palette='YlGnBu_r'),alpha=0.75,multiple='dodge',common_norm=False)
x = arr[0]
if x % 10 == 0:
    
    plt.title('Distance From Center for 1 Fish Sanded Tank Over Time')
else:
    plt.title('Distance From Center for 1 Fish Clear Tank Over Time')
plt.show()

sns.histplot(data=combined_df, x='spd',stat='percent',hue='Age',bins=10,binrange=[0,5],palette=sns.color_palette(palette='YlGnBu_r'),alpha=0.75,multiple='dodge',common_norm=False)
x = arr[0]
if x % 10 == 0:
    
    plt.title('Speed for 1 Fish Sanded Tank Over Time')
else:
    plt.title('Speed for 1 Fish Clear Tank Over Time')
plt.show()

sns.histplot(data=combined_df, x='spd_r',stat='percent',hue='Age',bins=10,binrange=[0,2.5],palette=sns.color_palette(palette='YlGnBu_r'),alpha=0.75,multiple='dodge',common_norm=False)
x = arr[0]
if x % 10 == 0:
    
    plt.title('Radial Speed for 1 Fish Sanded Tank Over Time')
else:
    plt.title('Radial Speed for 1 Fish Clear Tank Over Time')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_df, x='r', y='spd_r', s=5, hue='Age',palette=sns.color_palette(palette='YlGnBu_r'),alpha=0.25)
plt.title('Relationship between Radial Speed and Radial Position')
plt.xlabel('Radial Position')
plt.ylabel('Radial Speed')
plt.grid(True)
plt.show()