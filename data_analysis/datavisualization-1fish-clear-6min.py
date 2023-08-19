from numpy import load
import numpy as np
import math
import matplotlib.pyplot as plt
file0 = 'D:\\output\\1fish_single\\1fish_single_maxbright_6minacc_2023-08-09-145619-0000_fish0.npz
file1 = 'D:\\output\\1fish_single\\1fish_single_maxbright_6minacc_clear_2023-08-09-161213-0000_fish0.npz
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
distances = []
total_distance=0
count = 0
for position in positions0:
    distance = math.sqrt((length*(position[0]/30-1/2))**2 + (length*(position[1]/30-1/2))**2)
    if distance < 100:
        total_distance += distance
        count +=1
    distances.append(distance)
plt.hist(distances,bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5, 5])
avgdistance0 = total_distance / count
plt.title("avg distance = " + str(avgdistance0))
plt.show()
distances = []
total_distance=0
count = 0
for position in positions1:
    distance = math.sqrt((length*(position[0]/30-1/2))**2 + (length*(position[1]/30-1/2))**2)
    if distance < 100:
        total_distance += distance
        count +=1
    distances.append(distance)
plt.hist(distances,bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5, 5])
avgdistance1 = total_distance / count
plt.title("avg distance = " + str(avgdistance1))
plt.show()

speeds=[]
total_speed = 0
count = 0
for direction in directions0:
    speed = math.sqrt((direction[0])**2 + (direction[1])**2)
    speeds.append(speed)
    if speed < 10000:
        total_speed+=speed
        count +=1
    speeds.append(speed)
plt.hist(speeds,bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7])
avgspeed0 = total_speed/count
plt.title("avg speed = " + str(avgspeed0))
plt.show()
speeds=[]
total_speed = 0
count = 0
for direction in directions1:
    speed = math.sqrt((direction[0])**2 + (direction[1])**2)
    speeds.append(speed)
    if speed < 10000:
        total_speed+=speed
        count +=1
    speeds.append(speed)
plt.hist(speeds,bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7])
avgspeed1 = total_speed/count
plt.title("avg speed = " + str(avgspeed1))
plt.show()