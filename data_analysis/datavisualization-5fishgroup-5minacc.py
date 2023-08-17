from numpy import load
import numpy as np
import math
import matplotlib.pyplot as plt
file0 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish0.npz'
file1 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish1.npz'
file2 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish2.npz'
file3 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish3.npz'
file4 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish4.npz'
global length
length = 9
def open_file(file_name):
    data = load(file_name)
    lst = data.files
    #print(lst)
    xpos=[]
    ypos=[]
    vx=[]
    vy = []
    for item in lst:
        print(item)
        #print(data[item])
        if item == 'X':
            for pos in data[item]:
                print(pos)
                xpos.append(pos)
        if item == 'Y':
            for pos in data[item]:
                print(pos)
                ypos.append(pos)
            
        if item == 'VX':
            for pos in data[item]:
                print(pos)
                vx.append(pos)
        if item == 'VY':
            for pos in data[item]:
                print(pos)
                vy.append(pos)
    positions=np.stack((xpos, ypos), axis=-1)
    directions = np.stack((vx, vy), axis = -1)
    return positions, directions
positions0, directions0 = open_file(file0)
positions1, directions1 = open_file(file1)
positions2, directions2 = open_file(file2)
positions3, directions3 = open_file(file3)
positions4, directions4 = open_file(file4)
positions=[positions0, positions1, positions2, positions3, positions4]
directions=[directions0, directions1, directions2, directions3, directions4]
print(np.flatten(positions))
