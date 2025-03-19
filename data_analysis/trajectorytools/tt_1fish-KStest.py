import os
from pprint import pprint
import pathlib
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#from scipy import stats
from scipy.stats import ks_2samp, gaussian_kde
import seaborn as sns
import pandas as pd
import trajectorytools as tt
import trajectorytools.plot as ttplot
import trajectorytools.socialcontext as ttsocial
borders = ['Half','Sanded']
days = [2,2]
vision = ['N','N']
outputs = []
voutputs = []
arrdays = [7,14,21]
radius = 5
l1 = 'CDF of ' + str(borders[0]) + ' at ' + str(arrdays[days[0]]) + 'dpf'
l2 = 'CDF of ' + str(borders[1]) + ' at ' + str(arrdays[days[1]]) + 'dpf'
for i in range(len(borders)):
    print(i)
    val = borders[i]
    if val == 'Sanded':
        arr = [70,140,210]
    elif val == 'Clear':
        arr = [7,14,21]
    else:
        arr = [700,1400,2100]
    #indiv = input('Individual plots? (Y/N) : ')
    blind = vision[i]


    phalf = []
    tt_avg = []
    tt_std = []
    x = arr[days[i]]
    if x==0:
        break
    if x == 7:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/01.25.25/session_1fish-1fps-15min-7dpf-clear1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/01.25.25/session_1fish-1fps-15min-7dpf-clear2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/01.25.25/session_1fish-1fps-15min-7dpf-clear3/trajectories/validated.npy"
        file4 = "/Volumes/Hamilton/Zebrafish/AVI/01.25.25/session_1fish-1fps-15min-7dpf-clear4/trajectories/validated.npy"
        file5 = "/Volumes/Hamilton/Zebrafish/AVI/01.25.25/session_1fish-1fps-15min-7dpf-clear5/trajectories/validated.npy"
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
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded1/trajectories/validated.npy"
        file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded2/trajectories/validated.npy"
        file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.02.24/session_1fish-1fps-15min-7dpf-sanded3/trajectories/validated.npy"
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
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-half1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-half2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-half3-crispr/trajectories/validated.npy"
            file4 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-half4-crispr/trajectories/validated.npy"
            file5 = "/Volumes/Hamilton/Zebrafish/AVI/07.30.24/session_1fish-1fps-15min-7dpf-half5-crispr/trajectories/validated.npy"
            file6 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-half1-crispr/trajectories/validated.npy"
            file7 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-half2-crispr/trajectories/validated.npy"
            file8 = "/Volumes/Hamilton/Zebrafish/AVI/10.17.24/session_1fish-1fps-15min-7dpf-half3-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file4,file5,file6,file7,file8]

    elif x==1400:
        file1 = "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half1/trajectories/validated.npy"
        file2= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half2/trajectories/validated.npy"
        file3= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half3/trajectories/validated.npy"
        file4= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half4/trajectories/validated.npy"
        file5= "/Volumes/Hamilton/Zebrafish/AVI/07.09.24/session_1fish-1fps-15min-14dpf-half5/trajectories/validated.npy"
        files = [file1,file2,file3,file4,file5]
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-half1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-half2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-half3-crispr/trajectories/validated.npy"
            file4 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-half4-crispr/trajectories/validated.npy"
            file5 = "/Volumes/Hamilton/Zebrafish/AVI/08.12.24/session_1fish-1fps-15min-14dpf-half5-crispr/trajectories/validated.npy"
            file6 = "/Volumes/Hamilton/Zebrafish/AVI/11.13.24/session_1fish-1fps-15min-14dpf-half1-crispr/trajectories/validated.npy"
            file7 = "/Volumes/Hamilton/Zebrafish/AVI/11.13.24/session_1fish-1fps-15min-14dpf-half2-crispr/trajectories/validated.npy"
            files = [file1,file2,file3,file4,file5,file6,file7]
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
        if blind == 'Y':
            file1 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-half1-crispr/trajectories/validated.npy"
            file2 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-half2-crispr/trajectories/validated.npy"
            file3 = "/Volumes/Hamilton/Zebrafish/AVI/11.20.24/session_1fish-1fps-15min-21dpf-half3-crispr/trajectories/validated.npy"
            files = [file1,file2,file3]




    def count_ones(array):
        count = 0
        for num in array:
            if num == 1:
                count += 1
        return count
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
    times=10
    def border_turning(tr):
    #phalf = np.concatenate([tr1.s*(10/tr1.params['radius']), tr2.s*(10/tr2.params['radius']), tr3.s*(10/tr3.params['radius']), tr4.s*(10/tr4.params['radius']), tr5.s*(10/tr5.params['radius'])],axis=0)
    #phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])
        pos1= tr.s*tr.params['length_unit']*(2*radius/2048)

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

    processedpos = []
    processedvel = []
    for temp in trs:
        processed_temp = processtr(temp)
        border_turning(processed_temp)
        temppos = processed_temp.s*processed_temp.params['length_unit']*(2*radius/2048)
        tempvel = processed_temp.v*(processed_temp.params['length_unit']/processed_temp.params['time_unit'])*(20/2048)
        processedpos.append(temppos)
        processedvel.append(tempvel)
        




    
    phalf = np.concatenate(processedpos,axis=0)
    print(phalf.shape)
    phalf = np.reshape(phalf, [phalf.shape[0]*phalf.shape[1], 2])

    '''plt.hist2d(phalf[:, 0], phalf[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015)
    plt.xlabel('X-bins')
    plt.ylabel('Y-bins')
    plt.title('Heatmap for 1 Fish Half Sanded Tank ' + str(x)+'dpf')
    plt.colorbar(label='Frequency')
    plt.show()'''

    vhalf = np.concatenate(processedvel,axis=0)
    print(vhalf.shape)
    vhalf = np.reshape(vhalf, [vhalf.shape[0]*vhalf.shape[1], 2])
    dist = []
    



    ###plt.hist2d(phalf[:, 0], -1*phalf[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015
            #)
    ###plt.xlabel('X-bins')
    #plt.ylabel('Y-bins')
    #if x % 100 == 0:
    #    n = int(x/100)
    #    plt.title('Heatmap for 1 Fish Sanded Tank ' +str(n) +'dpf')
    #elif x % 10 == 0:
    #    n = int(x/10)
    #    plt.title('Heatmap for 1 Fish Sanded Tank ' +str(n) +'dpf')
    #else:
    #    plt.title('Heatmap for 1 Fish Clear Tank ' + str(x)+'dpf')
    #plt.colorbar(label='Frequency')
    #plt.show()




    phalf= pd.DataFrame(phalf,columns=['x','y'])
    phalf.rename(columns={'A': 'x', 'B': 'y'})
    phalf['r'] = np.sqrt(phalf['x']**2 + phalf['y']**2)
    phalf['y'] = -1*phalf['y']
    phalf['theta'] = np.arctan2(-1*phalf['x'],phalf['y'])
    print(phalf)
    #outputs.append(phalf)

    vhalf= pd.DataFrame(vhalf,columns=['vx','vy'])
    vhalf.rename(columns={'A': 'vx', 'B': 'vy'})
    vhalf['spd'] = np.sqrt(vhalf['vx']**2 + vhalf['vy']**2)
    vhalf['vy'] = -1*vhalf['vy']
    vhalf['vtheta'] = np.arctan2(-1*vhalf['vx'],vhalf['vy'])
    
    print('avg spd: ' + str(np.mean(vhalf['spd'])))
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
    '''for i in range(len(phi_temp)):
        if phi_temp[i] > np.pi/2:
            phi_temp[i] = np.pi-phi_temp[i]'''
    half_df['phi'] = phi_temp
    half_df['refl_prop'] = refl_prop
    print(half_df['refl_prop'])

    turn_times = []
    turns = []
    temp_counter = 0
    min = 0
    max = np.pi
    for index, row in half_df.iterrows():
        if temp_counter == 0 and row['r'] > 8 and row['phi'] < 0.4:
            max = row['phi'] + np.pi/2
            temp_counter+=1
        elif temp_counter == 0 and row['r'] > 8 and row['phi'] > np.pi - 0.4:
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

    
             

    if x%10 ==0:
        color = 'red'
    else:
        color = 'blue'

    '''if(indiv == 'Y'):

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
        plt.xlim(0,10)
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
        plt.xlim(0,10)
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
    turns = np.array(turns)'''
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
rvals = []
tvals = []
for output in outputs:
    rvals.append(output['r'])
    tvals.append(output['theta'])
print(rvals)
subsample_size = 100
count = 0
p_values=[]
for i in range(1000):
    # Perform the Kolmogorov-Smirnov test
    x = np.random.choice(rvals[0], subsample_size, replace=False)
    y = np.random.choice(rvals[1], subsample_size, replace=False)
    ks_statistic, p_value = ks_2samp(x, y)
    p_values.append(p_value)

    # Output the results
    #print(f"KS Statistic: {ks_statistic}")
    #print(f"P-Value: {p_value}")
    print(ks_statistic)
    # Interpret the results
    if p_value < 0.01:
        #print("The two distributions are significantly different.")
        count+=1
        
    #else:
        #print("The two distributions are not significantly different.")
# Sort the samples
print(f"Confidence: {count}")
x_sorted = np.sort(rvals[0])
y_sorted = np.sort(rvals[1])
colors = []
for i in range(2):
    border= borders[i]
    sim="N"
    if border == 'Clear':
        if sim == 'Y':
            colors.append('lightblue')
        else:
            colors.append('blue')
    elif border == 'Sanded':
        if sim == 'Y':
            colors.append('lightsalmon')
        else:
            colors.append('red')
    elif border == 'Brownian':
        colors.append('brown')
    else:
        if sim == 'Y':
            colors.append('orchid')
        else:
            colors.append('purple')
# Compute the empirical cumulative distribution function (CDF)
cdf_x = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
cdf_y = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
# Create a common grid (the union of the unique values from both datasets)
xx = np.sort(np.unique(np.concatenate([x_sorted, y_sorted])))

# Compute the empirical cdf values at these grid points.
# For a given point, the cdf is the fraction of sample values <= that point.
F_x = np.searchsorted(x_sorted, xx, side='right') / len(x_sorted)
F_y = np.searchsorted(y_sorted, xx, side='right') / len(y_sorted)

# Compute the area between the two curves using the trapezoidal rule.
area = np.trapz(np.abs(F_x - F_y), xx)

print("Area between the two cdfs:", area)
# Plot the CDFs of both samples
plt.figure(figsize=(4, 4))
plt.step(x_sorted, cdf_x,color = colors[0],label=l1, where='post')
plt.step(y_sorted, cdf_y, color = colors[1],label=l2, where='post')
plt.xlabel('Radius')
plt.ylabel('CDF')
plt.xlim(0,radius)
plt.legend()
plt.grid(True)


# Show the plot
plt.show()

# Plot KDEs for both distributions
sns.kdeplot(x_sorted, color = colors[0], label='Distribution 1', fill=True,bw_method=0.25)
sns.kdeplot(y_sorted, color = colors[1],label='Distribution 2', fill=True,bw_method=0.25)

plt.xlabel('Radius')
plt.ylabel('Density')
plt.title('Kernel Density Estimation of Radial Distributions')
plt.legend()
plt.show()
# Number of samples to draw from each KDE (adjust as needed)
n_samples = 100
kde_x = gaussian_kde(x_sorted,bw_method=0.25)
kde_y = gaussian_kde(y_sorted,bw_method=0.25)

count = 0
p_values_r=[]
D_values_r=[]
for i in range(1000):
    # Perform the Kolmogorov-Smirnov test
    samples_x = kde_x.resample(n_samples).flatten()  # flatten to 1D array
    samples_y = kde_y.resample(n_samples).flatten()
    ks_statistic, p_value = ks_2samp(samples_x, samples_y)
    p_values_r.append(p_value)
    D_values_r.append(ks_statistic)
    # Output the results
    #print(f"KS Statistic: {ks_statistic}")
    #print(f"P-Value: {p_value}")
    #print(ks_statistic)
    # Interpret the results
    if p_value < 0.05:
        #print("The two distributions are significantly different.")
        count+=1
print(f"Confidence: {count}")
# Show the plot
plt.show()

pos_left = [0.7, 1.3]   # positions for the violins/boxplots
fig, ax = plt.subplots(figsize=(4,6))

# Create the violin plots
parts = ax.violinplot([p_values_r, D_values_r], vert=True, showextrema=False, positions=pos_left)    
cp = ['blue', 'green']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(cp[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.25)
# Overlay the box plots to show quartiles and the median
boxplot = ax.boxplot([p_values_r, D_values_r],
                     positions=pos_left,
                     widths=0.1,
                     patch_artist=True,
                     showfliers=False,
                     vert=True)

# Customize the box plots (optional)
for patch in boxplot['boxes']:
    patch.set_facecolor('white')  # fill color
    patch.set_edgecolor('black')
for median in boxplot['medians']:
    median.set_color('black')
for whisker in boxplot['whiskers']:
    whisker.set_color('black')
for cap in boxplot['caps']:
    cap.set_color('black')

# Set x-axis ticks and labels
ax.set_xticks(pos_left)
ax.set_xticklabels(["p-values", "D-statistic"])
ax.set_ylim(0, 0.5)

# Add a horizontal line at y=0.01
ax.plot([0.4, 1], [0.01, 0.01], color='red', linestyle='--', linewidth=2)
ax.plot([1, 1.7], [0.23, 0.23], color='red', linestyle='--', linewidth=2)

#plt.title("Violin Plot with Embedded Box Plot")
plt.show()
# Draw random samples from each KDE object


# Apply the two-sample KS test


count = 0
for i in range(1000):
    # Perform the Kolmogorov-Smirnov test
    x = np.random.choice(tvals[0], subsample_size, replace=False)
    y = np.random.choice(tvals[1], subsample_size, replace=False)
    ks_statistic, p_value = ks_2samp(x, y)

    # Output the results
    #print(f"KS Statistic: {ks_statistic}")
    #print(f"P-Value: {p_value}")

    # Interpret the results
    if p_value < 0.05:
        #print("The two distributions are significantly different.")
        count+=1
        print(p_value)
    #else:
        #print("The two distributions are not significantly different.")
# Sort the samples
print(f"Confidence: {count}")
# Sort the samples
x_sorted = np.sort(tvals[0])
y_sorted = np.sort(tvals[1])

# Compute the empirical cumulative distribution function (CDF)
cdf_x = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
cdf_y = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

# Plot the CDFs of both samples
plt.figure(figsize=(4, 4))
plt.step(x_sorted, cdf_x, label=l1, color = colors[0], where='post')
plt.step(y_sorted, cdf_y, label=l2, color = colors[1], where='post')
plt.xlabel('Angle')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Plot KDEs for both distributions
sns.kdeplot(x_sorted, color = colors[0], label='Distribution 1', fill=True,bw_method=0.25)
sns.kdeplot(y_sorted, color = colors[1],label='Distribution 2', fill=True,bw_method=0.25)

plt.xlabel('Radius')
plt.ylabel('Density')
plt.title('Kernel Density Estimation of Radial Distributions')
plt.legend()
plt.show()
# Number of samples to draw from each KDE (adjust as needed)
n_samples = 100
kde_x = gaussian_kde(x_sorted,bw_method=0.25)
kde_y = gaussian_kde(y_sorted,bw_method=0.25)

count = 0
p_values_t=[]
D_values_t=[]
for i in range(1000):
    # Perform the Kolmogorov-Smirnov test
    samples_x = kde_x.resample(n_samples).flatten()  # flatten to 1D array
    samples_y = kde_y.resample(n_samples).flatten()
    ks_statistic, p_value = ks_2samp(samples_x, samples_y)
    p_values_t.append(p_value)
    D_values_t.append(ks_statistic)
    # Output the results
    #print(f"KS Statistic: {ks_statistic}")
    #print(f"P-Value: {p_value}")
    #print(ks_statistic)
    # Interpret the results
    if p_value < 0.05:
        #print("The two distributions are significantly different.")
        count+=1
print(f"Confidence: {count}")
# Show the plot
plt.show()

import matplotlib.patches as mpatches

pos_left = [0.7, 1.3, 1.9, 2.5]   # updated positions for the violins/boxplots
fig, ax = plt.subplots(figsize=(4,6))
#plt.rcParams['figure.dpi'] = 3000
# Create the violin plots
parts = ax.violinplot([p_values_r, D_values_r, p_values_t, D_values_t],
                       vert=True, showextrema=False, positions=pos_left)    
cp = ['blue', 'green', 'blue', 'green']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(cp[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.25)

# Overlay the box plots to show quartiles and the median
boxplot = ax.boxplot([p_values_r, D_values_r, p_values_t, D_values_t],
                     positions=pos_left,
                     widths=0.1,
                     patch_artist=True,
                     showfliers=False,
                     vert=True)

# Customize the box plots (optional)
for patch in boxplot['boxes']:
    patch.set_facecolor('white')  # fill color
    patch.set_edgecolor('black')
for median in boxplot['medians']:
    median.set_color('black')
for whisker in boxplot['whiskers']:
    whisker.set_color('black')
for cap in boxplot['caps']:
    cap.set_color('black')

# Set x-axis ticks and labels
ax.set_xticks([1, 2.2])
ax.set_xticklabels(["radial", "angular"])
ax.set_ylim(0, 0.5)

# Add horizontal lines at specified y-values
ax.plot([0.4, 1], [0.05, 0.05], color='red', linestyle='--', linewidth=2)
ax.plot([1, 1.6], [0.192, 0.192], color='red', linestyle='--', linewidth=2)
ax.plot([1.6, 2.2], [0.05, 0.05], color='red', linestyle='--', linewidth=2)
ax.plot([2.2, 2.8], [0.192, 0.192], color='red', linestyle='--', linewidth=2)

# Add legend labeling blue as "p-values" and green as "D-statistic"
blue_patch = mpatches.Patch(color='blue', label='p-values', alpha=0.25)
green_patch = mpatches.Patch(color='green', label='D-statistic', alpha=0.25)
#ax.legend(handles=[blue_patch, green_patch], loc='center left', bbox_to_anchor=(1, 0.5))

#plt.title("Violin Plot with Embedded Box Plot")
#plt.savefig('/Users/ezhu/Documents/test.svg')
plt.show()

# -------------------------------------------------------------------
# NEW PLOT: Split p-values to one axis and D-statistics to another
plt.rcParams['figure.dpi'] = 100
# Create a new figure and a primary axis for p-values
fig, ax1 = plt.subplots(figsize=(8, 6))
# Create a secondary y-axis for D-statistics that shares the same x-axis
ax2 = ax1.twinx()

# --- Plot for p-values on ax1 ---
pos = [0.7, 1.7]
parts_p = ax1.violinplot([p_values_r, p_values_t],
                           vert=True, showextrema=False,
                           positions=pos)
# Set the violin colors for p-values (blue)
for pc in parts_p['bodies']:
    pc.set_facecolor('blue')
    pc.set_edgecolor('black')
    pc.set_alpha(0.25)

# Overlay the box plots for p-values
boxplot_p = ax1.boxplot([p_values_r, p_values_t],
                        positions=pos,
                        widths=0.1,
                        patch_artist=True,
                        showfliers=False,
                        vert=True)
for patch in boxplot_p['boxes']:
    patch.set_facecolor('white')
    patch.set_edgecolor('black')
for median in boxplot_p['medians']:
    median.set_color('black')
for whisker in boxplot_p['whiskers']:
    whisker.set_color('black')
for cap in boxplot_p['caps']:
    cap.set_color('black')

ax1.set_xticks(pos)
ax1.set_xticklabels(["radial", "angular"])
ax1.set_ylim(0, 1)
ax1.set_ylabel("p-values", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.set_title("Violin Plots with Different y-axis Limits for p-values and D-statistic")

# --- Plot for D-statistics on ax2 ---
pos = [1.3, 2.3]
parts_d = ax2.violinplot([D_values_r, D_values_t],
                           vert=True, showextrema=False,
                           positions=pos)
# Set the violin colors for D-statistics (green)
for pc in parts_d['bodies']:
    pc.set_facecolor('green')
    pc.set_edgecolor('black')
    pc.set_alpha(0.25)

# Overlay the box plots for D-statistics
boxplot_d = ax2.boxplot([D_values_r, D_values_t],
                        positions=pos,
                        widths=0.1,
                        patch_artist=True,
                        showfliers=False,
                        vert=True)
for patch in boxplot_d['boxes']:
    patch.set_facecolor('white')
    patch.set_edgecolor('black')
for median in boxplot_d['medians']:
    median.set_color('black')
for whisker in boxplot_d['whiskers']:
    whisker.set_color('black')
for cap in boxplot_d['caps']:
    cap.set_color('black')
pos=[1,2]
ax2.set_xticks(pos)
ax2.set_xticklabels(["radial", "angular"])
ax2.set_ylim(0, 0.5)
ax2.set_ylabel("D-statistic", color="green")
ax2.tick_params(axis='y', labelcolor="green")

plt.tight_layout()
plt.show()
