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
import matplotlib.animation as animation

day = int(input('dpf: '))
spds = np.load('speeddistribution'+str(day)+'dpf.npy')

num = 10  # Number of particles
time = 900  # Time steps

positions = [[] for _ in range(num)]  # Store paths for all particles

def bounded_gaussian(mean=0, std=1, lower=-5, upper=5):
    while True:
        x = np.random.normal(mean, std)
        if lower <= x <= upper:
            return x

def random_walk_within_circle(radius=10, time_steps=1000, particle_index=0):
    x, y = bounded_gaussian(), bounded_gaussian()
    positions[particle_index].append((x, y))

    for _ in range(time_steps):
        step_size = spds[np.random.randint(0, len(spds))]
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)

        new_x = x + dx
        new_y = y + dy

        if new_x**2 + new_y**2 <= radius**2:
            x, y = new_x, new_y
        positions[particle_index].append((x, y))
    # Define file path
    output_dir = './modeling/data/randomwalk/'  # Change to relative path for better portability
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Generate file name
    pos_file = os.path.join(output_dir, f'positions{day}dpf{particle_index}.npy')
    # Save the file
    with open(pos_file, 'wb') as f:
        np.save(f, np.array(positions[particle_index]))

# Run the random walks
for i in range(num):
    random_walk_within_circle(radius=5, time_steps=time, particle_index=i)

# Visualization function
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('Random Walk within a Circle')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
circle = plt.Circle((0, 0), 5, color='r', fill=False)
ax.add_patch(circle)
scat = ax.scatter([], [], s=10, color='blue')

# Add time annotation
time_text = ax.text(-4, 4, '', fontsize=12, color='black')

def update(frame):
    """Updates the animation frame-by-frame."""
    x_data = []
    y_data = []
    for i in range(num):
        if frame < len(positions[i]):
            x_data.append(positions[i][frame][0])
            y_data.append(positions[i][frame][1])
    
    scat.set_offsets(np.c_[x_data, y_data])
    time_text.set_text(f'Time: {frame}')
    return scat, time_text

ani = animation.FuncAnimation(fig, update, frames=time, interval=10, blit=False, repeat=False)

plt.show()
    
    

def plot_random_walk(positions, radius=10, color = 'blue'):
        # Extract x and y positions
    all_x = [pos[0] for sublist in positions for pos in sublist]
    all_y = [pos[1] for sublist in positions for pos in sublist]   
    plt.figure(figsize=(3, 3))
    plt.hist2d(all_x, all_y, bins=(10, 10), range=[[-5,5],[-5,5]], cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.05)    
    center = (0, 0)

    # Create circle
    theta = np.linspace(0, 2 * np.pi, 300)
    xc = center[0] + radius * np.cos(theta)
    yc = center[1] + radius * np.sin(theta)

    xc = center[0] + radius * np.cos(theta)
    yc = center[1] + radius * np.sin(theta)
    plt.plot(xc, yc, label=f'Circle with radius {radius}')
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.title('Random Walk within a Circle for '+ str(x)+'dpf fish')
    #plt.xlabel('X Position')
    #plt.ylabel('Y Position')
    plt.show()

plot_random_walk(positions, radius=5, color = 'blue')