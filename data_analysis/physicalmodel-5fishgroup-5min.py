from numpy import load
import numpy as np
import math
import matplotlib.pyplot as plt

data = load('D:\\output\\5fish_single\\5fish_group_maxbright_5minacc_2023-08-09-155141-0000_fish0.npz')
lst = data.files
for item in lst:
    if item == 'fish_pos':
        positions = data[item]

    if item == 'fish_angle':
        angles = data[item]
        directions = np.zeros((len(angles),2))
        for i in range(len(angles)):
            directions[i,0]=math.cos(angles[i])
            directions[i,1]=math.sin(angles[i])
print(positions[:,1])
print(directions)



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
        ax.quiver(positions[i, 0], positions[i, 1], sc*directions[i, 0], sc*directions[i, 1], scale=1)
        ax.set_aspect('equal')
        ax.set_xlim((300,1000))
        ax.set_ylim((0,700))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.pause(0.0001)

    if title:
        ax.set_title(title)

    plt.show()

plot_quiver(positions, directions, sc=0.05, title="Quiver Plot Example")
