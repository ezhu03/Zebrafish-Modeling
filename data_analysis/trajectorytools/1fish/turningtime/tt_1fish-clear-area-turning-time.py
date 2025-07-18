'''
This code processes trajectory data of zebrafish from multiple files based on the days post-fertilization (dpf) input 
provided by the user. Depending on the input, it selects a corresponding set of trajectory files. The code includes 
functions to open and process these files using the trajectorytools package, converting the data into a usable format 
for analysis. It further processes the trajectory data to estimate positions, velocities, and accelerations, and 
analyzes the turning behavior of the fish near the border of its environment via calculating batched turning times
for given area bins, plotting the turning time values with respect to reflection area.
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
from scipy.optimize import curve_fit
while(True):
    '''
    SET THESE VALUES BEFORE RUNNING THE CODE
    radius: radius of the tank (for reflection calculation)
    times: number of time points to calculate the turning time
    '''
    radius = 10
    times = 20

    '''
    Take in the input of the dpf, here we go by convention that 7,14,21 is clear and 70,140,210 is sanded, and 700,1400,2100 is half sanded
    This code is designed to be used with clear data (7,14,21), the sanded and half sanded data is provided but not meant to be used for this code
    '''
    x = int(input('dpf: '))
    if x==0:
        break
    if x == 7:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-clear-1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-clear-2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-clear-3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-clear-4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-clear-5/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy"
        file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-clear2/trajectories/validated.npy"
        file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-clear3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]

    if x == 14:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-clear5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]


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

    if x==70:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded2/trajectories/validated.npy"
        file3= "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-sanded3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded1/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded2/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6]

    if x==140:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-sanded5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]


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

    if x==700:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half1/trajectories/validated.npy"
        file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half2/trajectories/validated.npy"
        file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]

    elif x==1400:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half1/trajectories/validated.npy"
        file2= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half2/trajectories/validated.npy"
        file3= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half3/trajectories/validated.npy"
        file4= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half4/trajectories/validated.npy"
        file5= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]

    elif x==2100:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-2-21dpf/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-3-21dpf/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/5.21.24/session_1fish-1fps-15min-21dpf-half-4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-half1/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-half2/trajectories/validated.npy"
        file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-half3/trajectories/validated.npy"
        file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-half4/trajectories/validated.npy"
        file9 = "/Volumes/Hamilton/Zebrafish/AVI/07.16.24/session_1fish-1fps-15min-21dpf-half5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8,file9]
        files = [file1,file2,file3,file4,file5,file6,file7,file8]

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

    processedtrs = []
    for temp in trs:
        processed_temp = processtr(temp)
        processedtrs.append(processed_temp)


    def count_ones(array):
        count = 0
        for num in array:
            if num == 1:
                count += 1
        return count
    
    '''
    function that calculates where a fish can see its reflection given a position and velocity
    '''
    def plotReflection(xposition, yposition, xvelocity, yvelocity):
        mag = np.sqrt(xposition **2 + yposition**2)
        magv = np.sqrt(xvelocity **2 + yvelocity**2)
        distance = 2*(10 - mag)

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

    '''
    converts the position and velocity data into a useable format for analysis
    takes each time point to find the border correlation given the initial border reflection area
    '''
    def border_turning(tr):
    #phalf = np.concatenate([tr1.s*(10/tr1.params['radius']), tr2.s*(10/tr2.params['radius']), tr3.s*(10/tr3.params['radius']), tr4.s*(10/tr4.params['radius']), tr5.s*(10/tr5.params['radius'])],axis=0)
    #phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])
        pos1= tr.s*tr.params['length_unit']*(20/2048)

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
        while i<len(pos1)-times:
            pos_mag = np.sqrt(pos1[i][0]**2+pos1[i][1]**2)
            prop = plotReflection(pos1[i][0],pos1[i][1],v1[i][0],v1[i][1])
            dps = []
            for j in range(times):
                dp = np.dot(v1[i],v1[i+j])
                dps.append(dp)
            if prop>0 and pos_mag>0:
                refl_prop.append(prop)
                correlations.append(dps)
                pos_arr.append(pos_mag)
            i+=1
            

    for temp in processedtrs:
        border_turning(temp)
    '''
    process the data from border_turning function by binning based on area to find the critical turning time
    '''
    # create a dataframe to store the data from border_turning function
    data = {'x': refl_prop, 'y':correlations}

    df = pd.DataFrame(correlations)
    df['area']=refl_prop
    df['area']*=2
    #plt.scatter(x=refl_prop, y=correlations, s=1)
    # Scatter plot
    bin_edges = [0,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,1]
    #if x ==14:
    #    bin_edges = [0,0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,1]

        # Bin data by 'Category' using pd.cut() and calculate mean
    df['bins'] = pd.cut(df['area'], bins=bin_edges)
    bin_values = sorted(df['bins'].drop_duplicates().tolist())
    print(bin_values)
        # Define the exponential function
    def exponential_func(x, a, b):
        return a * np.exp(b * x)
    tstars = []
    errors = []
    '''
    for each bin, calculate the individual critical turning coefficient for each time point, then plot it as a histogram
    '''
    for bin_value in bin_values:
        if bin_value == bin_values[-1]:
            break
        bin1 = df[df['bins'] == bin_value]

        bin1 = bin1.drop(['area', 'bins'], axis=1)

        # Prepare the data for plotting
        # Melt the DataFrame to long format
        df_long = bin1.melt(var_name='time', value_name='position')
        # Define the exponential function
        df_long['time'] = df_long['time'].astype(float)



        # Ensure time and position are numpy arrays for curve fitting
        time_values = df_long['time'].to_numpy()
        position_values = df_long['position'].to_numpy()

        # Fit the curve with an initial guess for a and b
        popt, pcov = curve_fit(exponential_func, time_values, position_values, p0=(1, 0.1))

        # Create the scatter plot
        fitted_time_values = np.linspace(time_values.min(), time_values.max(), 500)
        fitted_position_values = exponential_func(fitted_time_values, *popt)
        #plt.plot(fitted_time_values, fitted_position_values, color='red', label=f'Fit: $y = {popt[0]:.2f} e^{{ {popt[1]:.2f} x }}$')
        print(bin_value, *popt)
        a,b = popt
        t_star=(1/b)*np.log(1/(2*a))
        tstars.append(t_star)
        # Calculate standard deviations for parameters
        perr = np.sqrt(np.diag(pcov))
        ci = perr
        a_up, b_up = popt + ci
        a_low, b_low = popt - ci
        t_up =(1/b_up)*np.log(1/(2*a_up))
        t_low =(1/b_low)*np.log(1/(2*a_low))
        print(t_up, t_star, t_low)

        errors.append([t_star-t_low, t_up-t_star])


        print('t* = ', t_star)
        #plt.scatter(df_long['time'], df_long['position'], alpha=0.6,s=1)

        # Adding title and labels
        #plt.title('Position vs Time Scatterplot')
        #plt.xlabel('Time')
        #plt.ylabel('Position')

        # Display the plot
        #plt.show()
    '''
    given each bin, produce an overall turning time variation given the area parameter
    plotting the critical turning time vs. the area parameter with errorbars
    '''
    mean_areas = df.groupby('bins')['area'].agg(['mean', 'std']).reset_index()
    mean_values = df.groupby('bins')[df.columns[0:times]].agg(['mean']).reset_index()
    std_values = df.groupby('bins')[df.columns[0:times]].agg(['std']).reset_index()
    #mean_values['bin'] = [0.05,0.15,0.25,0.35,0.45]

    #colors = ['red','orange','green','blue','purple']
    #for a in range(4):
    plt.figure(figsize=(8, 6))
    print(len(mean_areas['mean'][:-1]),len(tstars),len(bin_edges))
    plt.errorbar(x=mean_areas['mean'][:-1],y=tstars,yerr = np.array(errors).T,xerr=mean_areas['std'][:-1],fmt='o',color='cornflowerblue')
    if x%10==0:
        plt.title('Critical Turning Time vs. Artificial Area Parameter for Sanded Tank at ' + str(int(x/10)) +'dpf')

    else:
        plt.title('Critical Turning Time vs. Area Parameter for Clear Tank at ' + str(x) +'dpf')
    plt.ylabel('t* (s)')
    plt.xlabel('area parameter')
    plt.ylim(0,6)

    plt.show()
    #print(mean_areas)


