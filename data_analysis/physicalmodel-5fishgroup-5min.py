from numpy import load
import numpy as np
import math
import matplotlib.pyplot as plt
file0 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish0.npz'
file1 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish1.npz'
file2 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish2.npz'
file3 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish3.npz'
file4 = 'D:\\output\\5fish_group\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish4.npz'
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
positions1, directions1 = open_file(file0)
positions2, directions2 = open_file(file0)
positions3, directions3 = open_file(file0)
positions4, directions4 = open_file(file0)



def plot_quiver(positions, directions, sc=1.0, title=None):
    """
    Plot a quiver plot of positions and directions.

    Parameters:
        positions (numpy.ndarray): An array of shape (N, 2) containing x and y positions.
        directions (numpy.ndarray): An array of shape (N, 2) containing x and y directions.
        scale (float, optional): Scaling factor for the arrow length. Default is 1.0.
        title (str, optional): Title of the plot. Default is None.
    """
    fig, ax = plt.subplots()
    for i in range(len(positions)):
        ax.clear()
        ax.quiver(positions0[i, 0], positions0[i, 1], sc*directions0[i, 0], sc*directions0[i, 1], scale=1)
        ax.quiver(positions1[i, 0], positions1[i, 1], sc*directions1[i, 0], sc*directions1[i, 1], scale=1)
        ax.quiver(positions2[i, 0], positions2[i, 1], sc*directions2[i, 0], sc*directions2[i, 1], scale=1)
        ax.quiver(positions3[i, 0], positions3[i, 1], sc*directions3[i, 0], sc*directions3[i, 1], scale=1)
        ax.quiver(positions4[i, 0], positions4[i, 1], sc*directions4[i, 0], sc*directions4[i, 1], scale=1)

        ax.set_aspect('equal')
        ax.set_xlim((0,20))
        ax.set_ylim((0,20))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.pause(0.0001)

    if title:
        ax.set_title(title)

    plt.show()

plot_quiver(positions, directions, sc=0.05, title="Quiver Plot Example")
