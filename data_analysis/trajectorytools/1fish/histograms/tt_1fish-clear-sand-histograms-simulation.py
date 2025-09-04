'''
Given clear or sanded tank data, this will iterate through the data and produce histograms for the radial speed, 
radial position, angle to the wall, and the angle to the wall given the initial reflection area.
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
    path = input('Physical Path? (Y/N) : ')
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
    elif x == 7:
        file1 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear1/trajectories/trajectories.npy"
        file2 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear2/trajectories/trajectories.npy"
        file3 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear3/trajectories/trajectories.npy"
        file4 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear4/trajectories/trajectories.npy"
        file5 = "data/06.18.25/session_1fish-1fps-15min-7dpf-clear5/trajectories/trajectories.npy"
        file6 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy"
        file7 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear2/trajectories/validated.npy"
        file8 = "data/07.02.24/session_1fish-1fps-15min-7dpf-clear3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear2-crispr/trajectories/validated.npy"
            file8 = "data/10.17.24/session_1fish-1fps-15min-7dpf-clear3-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file4,file5,file6,file7,file8]
    elif x == 14:
        file1 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear1/trajectories/validated.npy"
        file2 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear2/trajectories/validated.npy"
        file3 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear3/trajectories/validated.npy"
        file4 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear4/trajectories/validated.npy"
        file5 = "data/07.09.24/session_1fish-1fps-15min-14dpf-clear5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-clear5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-clear1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-clear2-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file5,file6, file7]
    elif x==21:
        file1 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file2 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file3 = "data/5.21.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file4 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear1/trajectories/validated.npy"
        file5 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear2/trajectories/validated.npy"
        file6 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear3/trajectories/validated.npy"
        file7 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear4/trajectories/validated.npy"
        file8 = "data/07.16.24/session_1fish-1fps-15min-21dpf-clear5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-clear3-crispr/trajectories/validated.npy"
            files = [file1,file2,file3]
    elif x==70:
        files = ["modeling/data/boundary/const10radius0boxradius5iter10fish1_15min_7dpf_sanded.npy"]
        if blind == 'Y':
            file1 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "data/07.30.24/session_1fish-1fps-15min-7dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "data/10.17.24/session_1fish-1fps-15min-7dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "data/10.17.24/session_1fish-1fps-15min-7dpf-sanded2-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file5,file6,file7]
    elif x==140:
        files = ["modeling/data/boundary/const10radius0boxradius5iter10fish1_15min_14dpf_sanded.npy"]
        if blind == 'Y':
            file1 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded3-crispr/trajectories/validated.npy"
            file4 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded4-crispr/trajectories/validated.npy"
            file5 = "data/08.12.24/session_1fish-1fps-15min-14dpf-sanded5-crispr/trajectories/validated.npy"
            file6 = "data/11.13.24/session_1fish-1fps-15min-14dpf-sanded1-crispr/trajectories/validated.npy"
            file7 = "data/11.13.24/session_1fish-1fps-15min-14dpf-sanded2-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file4,file5, file6, file7]
    elif x == 210:
        files = ["modeling/data/boundary/const10radius0boxradius5iter10fish1_15min_21dpf_sanded.npy"]
        if blind == 'Y':
            file1 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded1-crispr/trajectories/validated.npy"
            file2 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded2-crispr/trajectories/validated.npy"
            file3 = "data/11.20.24/session_1fish-1fps-15min-21dpf-sanded3-crispr/trajectories/validated.npy"
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
    def border_turning(tr):
    #phalf = np.concatenate([tr1.s*(10/tr1.params['radius']), tr2.s*(10/tr2.params['radius']), tr3.s*(10/tr3.params['radius']), tr4.s*(10/tr4.params['radius']), tr5.s*(10/tr5.params['radius'])],axis=0)
    #phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])
        pos1= tr.s*tr.params['length_unit']*(20/2048)

        pos1 = np.array(pos1.reshape(pos1.shape[0],2))

        for i in range(len(pos1)):
            pos1[i][1]*=(-1)

        v1 = np.array(tr.v).reshape(tr.v.shape[0],2)

        for i in range(len(v1)):
            v1[i][1]*=(-1)

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
    #phalf = np.concatenate(processedpos,axis=0)
    #print(phalf.shape)
    #phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])

    '''plt.hist2d(phalf[:, 0], phalf[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015)
    plt.xlabel('X-bins')
    plt.ylabel('Y-bins')
    plt.title('Heatmap for 1 Fish Half Sanded Tank ' + str(x)+'dpf')
    plt.colorbar(label='Frequency')
    plt.show()'''

    #vhalf = np.concatenate(processedvel,axis=0)
    #print(vhalf.shape)
    #vhalf = np.reshape(vhalf, [vhalf.shape[0]*vhalf.shape[1], 2])
    #dist = []
    

    
    #center = (0, 0)

    # Create circle
    #theta = np.linspace(0, 2 * np.pi, 300)
    #xc = center[0] + radius * np.cos(theta)
    #yc = center[1] + radius * np.sin(theta)
    #plt.figure(figsize=(3, 3))
    #plt.rcParams['figure.dpi'] = 100
    #plt.plot(xc, yc, label=f'Circle with radius {radius}')
    #plt.hist2d(phalf[:, 0], -1*phalf[: , 1], bins=(10, 10), range=[[-5,5],[-5,5]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.05
    #        )
    #plt.xlabel('X-bins')
    #plt.ylabel('Y-bins')
    '''if x % 10 == 0:
        n = int(x/10)
        plt.title('Heatmap for 1 Fish Sanded Tank ' +str(n) +'dpf')
    else:
        plt.title('Heatmap for 1 Fish Clear Tank ' + str(x)+'dpf')'''
    #plt.colorbar(label='Frequency')
    #plt.savefig("/Users/ezhu/Downloads/cs-histogram.png", dpi=3000, bbox_inches='tight')
    #plt.show()




    #phalf= pd.DataFrame(phalf,columns=['x','y'])
    #phalf.rename(columns={'A': 'x', 'B': 'y'})
    #phalf['r'] = np.sqrt(phalf['x']**2 + phalf['y']**2)
    #phalf['y'] = -1*phalf['y']
    #phalf['theta'] = np.arctan2(-1*phalf['x'],phalf['y'])
    #print(phalf)
    #outputs.append(phalf)

    #vhalf= pd.DataFrame(vhalf,columns=['vx','vy'])
    #vhalf.rename(columns={'A': 'vx', 'B': 'vy'})
    #vhalf['spd'] = np.sqrt(vhalf['vx']**2 + vhalf['vy']**2)
    #vhalf['vy'] = -1*vhalf['vy']
    #vhalf['vtheta'] = np.arctan2(-1*vhalf['vx'],vhalf['vy'])
    
    #print('avg spd: ' + str(np.mean(vhalf['spd'])))
    #voutputs.append(vhalf)

    #half_df =  pd.concat([phalf, vhalf], axis=1)
    #print(half_df)
    #half_df['vrx'] = half_df['vx']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    #half_df['vry'] = half_df['vy']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    #half_df['vr'] = half_df['spd']*(half_df['x']*half_df['vx']+half_df['y']*half_df['vy'])/(half_df['r']*half_df['spd'])
    #half_df['spd_r'] = np.abs(half_df['vr'])
    #half_df['vtx'] = half_df['vx']-half_df['vrx']
    #half_df['vty'] = half_df['vy']-half_df['vry']
    #half_df['spd_t'] = np.sqrt(half_df['vtx']**2+half_df['vty']**2)
    #phi_temp = np.arccos((-np.cos(half_df['theta'])*half_df['vx']-np.sin(half_df['theta']*half_df['vy']))/half_df['spd'])
    #print(phi_temp)
    '''for i in range(len(phi_temp)):
        if phi_temp[i] > np.pi/2:
            phi_temp[i] = np.pi-phi_temp[i]'''
    #half_df['phi'] = phi_temp
    #half_df['refl_prop'] = refl_prop
    #print(half_df['refl_prop'])

    #turn_times = []
    #turns = []
    #temp_counter = 0
    #min = 0
    #max = np.pi
    #for index, row in half_df.iterrows():
    #    if temp_counter == 0 and row['r'] > 8 and row['phi'] < 0.4:
    #        max = row['phi'] + np.pi/2
    #        temp_counter+=1
    #    elif temp_counter == 0 and row['r'] > 8 and row['phi'] > np.pi - 0.4:
    #        min = row['phi'] - np.pi/2
    #        temp_counter+=1
    #    elif temp_counter > 0 and row['phi'] >= min and row['phi'] <= max:
    #        temp_counter+=1
    #    elif temp_counter > 0:
    #        turns.append([row['x'],row['y']])
    #        turn_times.append(temp_counter)
    #        temp_counter = 0
    #        min = 0
    #        max = np.pi
    #tt_avg.append(np.mean(turn_times))
    #tt_std.append(np.std(turn_times))

    
             
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

    #outputs.append(half_df)
dpf = [7,14,21]

'''colors=sns.color_palette(palette='YlGnBu_r')
for i in range(len(outputs)):
    #plt.hist(data=outputs[i],x='center',density=True,bins=10,range=[0,10],color=(colors[i], 0.3))
    if arr[0]%10==0:
        sns.histplot(data=outputs[i], x='center',stat='percent',bins=20,binrange=[0,10],color=colors[2*i+1],alpha=1-i/3, label = str(arr[i]/10) + 'dpf')
    else:
        sns.histplot(data=outputs[i], x='center',stat='percent',bins=20,binrange=[0,10],color=colors[2*i+1],alpha=1-i/3, label = str(arr[i]) + 'dpf')'''

plt.figure(figsize=(10, 6))
colors = ['blue', 'purple','red']

# 1D histogram settings
bin_edges = np.linspace(0, 5, 11)  # 0–10 range into 20 bins

for i, output in enumerate(outputs):
    data = output['r'].dropna().values
    if data.size == 0:
        continue
    # Scale to percent per bin so each series sums to 100%
    weights = np.ones_like(data, dtype=float) * (100.0 / data.size)
    # Draw normal histogram bars but unfilled (outlined only)
    plt.hist(
        data,
        bins=bin_edges,
        weights=weights,
        histtype='bar',
        linewidth=2.5,
        edgecolor=colors[i % len(colors)],
        facecolor='none',
        fill=False,
        label=(f"{int(arr[i]/10)}dpf" if arr[i] % 10 == 0 else f"{arr[i]}dpf"),
        alpha=0.4
    )

plt.xlabel('Radial Position (cm)')
plt.ylabel('Percentage (%)')
plt.xlim([0, 5])
plt.ylim([0, 60])
plt.legend()
plt.savefig("/Users/ezhu/Downloads/radial_histogram.png", dpi=3000, bbox_inches='tight')
plt.show()