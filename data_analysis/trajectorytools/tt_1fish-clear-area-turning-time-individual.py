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
    x = int(input('dpf: '))
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
    
    if x == 700:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"
        file6 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half1/trajectories/validated.npy"
        file7 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half2/trajectories/validated.npy"
        file8 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-half3/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5,file6,file7,file8]
    
    if x==1400: 
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half1/trajectories/validated.npy"
        file2= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half2/trajectories/validated.npy"
        file3= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half3/trajectories/validated.npy"
        file4= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half4/trajectories/validated.npy"
        file5= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]

    if x==2100:
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

    # Save the merged array to a new .npy file
    #np.save("merged_file.npy", merged_array)

    def openfile(file, sigma = 1):
        tr = tt.Trajectories.from_idtrackerai(file, 
                                        interpolate_nans=True,
                                        smooth_params={'sigma': sigma})
        return tr
    trs = []
    for file in files:
        tr_temp = openfile(file)
        trs.append(tr_temp)

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
    radius = 10
    times = 10
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
    def exponential_func(x, a):
        return np.exp(a * x)
    tstars = []
    errors = []
    for bin_value in bin_values:
        if bin_value == bin_values[-1]:
            break
        bin1 = df[df['bins'] == bin_value]

        bin1 = bin1.drop(['area', 'bins'], axis=1)
        print("bin1: ", bin1)

        # Prepare the data for plotting
        # Melt the DataFrame to long format
        df_long = bin1.melt(var_name='time', value_name='position')
        # Define the exponential function
        df_long['time'] = df_long['time'].astype(float)
        temp_tstars = []
        for index, row in bin1.iterrows():
            popt, pcov = curve_fit(exponential_func, range(times), row, p0=(-0.2))
            a = popt
            t_star=(1/a)*np.log(1/(2))
            temp_tstars.append(t_star)
        ts = pd.DataFrame(temp_tstars,columns=['ts'])
        print(ts)
        plt.figure(figsize=(9, 6))
    #ax = sns.histplot(half_df, x="x", y="y",bins=(10, 10), binrange=[[-10,10],[-10,10]],cmap = sns.color_palette("light:b",as_cmap=True),cbar=True)
    #ax.set_aspect('equal')
        sns.histplot(data=ts, x='ts',stat='percent',bins=20,binrange=[0,20],palette=sns.color_palette(palette='YlGnBu_r'),alpha=0.75)
        plt.xlabel('X-bins')
        plt.ylabel('Y-bins')
        if x%100==0:
            plt.title('Heatmap for 1 Fish Sanded Tank ' + str(int(x/100))+'dpf for Area Parameter' + str(bin_value))
        elif x % 10 ==0:
            plt.title('Heatmap for 1 Fish Sanded Tank ' + str(int(x/10))+'dpf for Area Parameter' + str(bin_value))
        else:
            plt.title('Heatmap for 1 Fish Clear Tank ' + str(x)+'dpf for Area Parameter' + str(bin_value))
        
        tstars.append(np.median(temp_tstars))
        errors.append([np.percentile(temp_tstars,25),np.percentile(temp_tstars,75)])
    #plt.colorbar(label='Frequency')
        plt.show()
   



        print('t* = ', np.median(temp_tstars))
        
        

        # Display the plot
        plt.show()
    mean_areas = df.groupby('bins')['area'].agg(['mean', 'std']).reset_index()
    mean_values = df.groupby('bins')[df.columns[0:times]].agg(['mean']).reset_index()
    std_values = df.groupby('bins')[df.columns[0:times]].agg(['std']).reset_index()
    #mean_values['bin'] = [0.05,0.15,0.25,0.35,0.45]

    #colors = ['red','orange','green','blue','purple']
    #for a in range(4):
    plt.figure(figsize=(8, 6))
    print(len(mean_areas['mean'][:-1]),len(tstars),len(bin_edges))
    plt.errorbar(x=mean_areas['mean'][:-1],y=tstars,yerr = np.array(errors).T,xerr=mean_areas['std'][:-1],fmt='o',color='cornflowerblue')
    if x%100==0:
        plt.title('Critical Turning Time vs. Artificial Area Parameter for Half Sanded Tank at ' + str(int(x/100)) +'dpf')
    elif x%10==0:
        plt.title('Critical Turning Time vs. Artificial Area Parameter for Sanded Tank at ' + str(int(x/10)) +'dpf')
    else:
        plt.title('Critical Turning Time vs. Area Parameter for Clear Tank at ' + str(x) +'dpf')
    plt.ylabel('t* (s)')
    plt.xlabel('area parameter')
    plt.ylim(0,20)

    plt.show()
    #print(mean_areas)


