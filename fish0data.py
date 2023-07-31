from numpy import load

data = load('2023-07-27_16-08-57.mp4_visual_field_fish0.npz')
lst = data.files
for item in lst:
    if item == 'fish_pos':
        positions = data[item]

    if item == 'fish_angle':
        directions = data[item]
print(positions)
print(directions)
import numpy as np
import matplotlib.pyplot as plt

def plot_quiver(positions, directions, scale=1.0, title=None):
    """
    Plot a quiver plot of positions and directions.

    Parameters:
        positions (numpy.ndarray): An array of shape (N, 2) containing x and y positions.
        directions (numpy.ndarray): An array of shape (N, 2) containing x and y directions.
        scale (float, optional): Scaling factor for the arrow length. Default is 1.0.
        title (str, optional): Title of the plot. Default is None.
    """
    fig, ax = plt.subplots()
    ax.quiver(positions[:, 0], positions[:, 1], directions[:, 0], directions[:, 1], scale=scale)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if title:
        ax.set_title(title)

    plt.show()

plot_quiver(positions, directions, scale=1.0, title="Quiver Plot Example")
