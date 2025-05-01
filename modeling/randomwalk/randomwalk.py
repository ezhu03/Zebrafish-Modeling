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
x = int(input('dpf: '))
num = int(input('iterations: '))
time = int(input('time: '))
while x != 0:
    
    def random_walk_within_circle(radius=10, step_size=1, time_steps=1000):
        # Initialize position
        x, y = 0, 0
        positions = [(x, y)]

        for _ in range(time_steps):
            # Random step direction
            angle = np.random.uniform(0, 2 * np.pi)
            dx = step_size * np.cos(angle)
            dy = step_size * np.sin(angle)

            # New position
            new_x = x + dx
            new_y = y + dy

            # Check if within circle
            if new_x**2 + new_y**2 <= radius**2:
                x, y = new_x, new_y
                positions.append((x, y))
            else:
                # If step goes out of bounds, retry without updating position
                continue

        return positions

    def plot_random_walk(positions, radius=10, color = 'blue'):
        # Extract x and y positions
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]    
        plt.plot(x_positions, y_positions, marker='o', markersize=1, color = color, alpha = 0.1)
        plt.gca().add_patch(plt.Circle((0, 0), radius, color='r', fill=False))
        plt.xlim(-radius-1, radius+1)
        plt.ylim(-radius-1, radius+1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Random Walk within a Circle')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        
    if x ==7:
        step = 0.78
    elif x==14:
        step = 0.99
    elif x ==21:
        step = 1.43
    else:
        break


    # Example usage
    plt.figure(figsize=(8, 8))
    position_array = []
    for i in range(num):
        positions = random_walk_within_circle(radius=10, step_size=step, time_steps=time)
        plot_random_walk(positions, radius=10)
        position_array.append(positions)

    plt.show()

    position_array = np.vstack(position_array)

    plt.hist2d(position_array[:, 0], position_array[: , 1], bins=(10, 10), range=[[-10,10],[-10,10]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015)
    plt.xlabel('X-bins')
    plt.ylabel('Y-bins')
    plt.title('Heatmap for Random Walk' + str(x)+'dpf')
    plt.colorbar(label='Frequency')
    plt.show()
    x = int(input('dpf: '))