from numpy import load
import numpy as np
import math
import matplotlib.pyplot as plt
file0 = 'D:\\output\\5fish_single\\5fish_single_maxbright_5minacc_2023-08-09-153535-0000_fish0.npz'
file1 = 'D:\\output\\5fish_single\\5fish_single_maxbright_5minacc_2023-08-09-153535-0000_fish1.npz'
file2 = 'D:\\output\\5fish_single\\5fish_single_maxbright_5minacc_2023-08-09-153535-0000_fish2.npz'
file3 = 'D:\\output\\5fish_single\\5fish_single_maxbright_5minacc_2023-08-09-153535-0000_fish3.npz'
file4 = 'D:\\output\\5fish_single\\5fish_single_maxbright_5minacc_2023-08-09-153535-0000_fish4.npz'
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
        #print(data[item])
        if item == 'X':
            for pos in data[item]:
                xpos.append(pos)
        if item == 'Y':
            for pos in data[item]:
                ypos.append(pos)
            
        if item == 'VX':
            for pos in data[item]:
                vx.append(pos)
        if item == 'VY':
            for pos in data[item]:
                vy.append(pos)
    positions=np.stack((xpos, ypos), axis=-1)
    directions = np.stack((vx, vy), axis = -1)
    return positions, directions
positions0, directions0 = open_file(file0)
positions1, directions1 = open_file(file1)
positions2, directions2 = open_file(file2)
positions3, directions3 = open_file(file3)
positions4, directions4 = open_file(file4)
positions = []
directions = []
for i in range(len(positions0)):
    positions.append(positions0[i])
    directions.append(directions0[i])
    positions.append(positions1[i])
    directions.append(directions1[i])
    positions.append(positions2[i])
    directions.append(directions2[i])
    positions.append(positions3[i])
    directions.append(directions3[i])
    positions.append(positions4[i])
    directions.append(directions4[i])
distances=[]
total_distance=0
count = 0
for position in positions:
    distance = math.sqrt((length*(position[0]/30-1/2))**2 + (length*(position[1]/30-1/2))**2)
    if distance < 100:
        total_distance += distance
        count +=1
    distances.append(distance)
plt.hist(distances,bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5, 5])
avgdistance = total_distance / count
plt.title("avg distance = " + str(avgdistance))
plt.show()
speeds=[]
for direction in directions:
    speed = math.sqrt((direction[0])**2 + (direction[1])**2)
    speeds.append(speed)
plt.hist(speeds,bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7])
plt.show()