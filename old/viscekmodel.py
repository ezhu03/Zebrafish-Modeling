import numpy as np
import matplotlib.pyplot as plt
import time

# Set up the simulation parameters
num_agents = 50
speed = 0.1
noise = 0.001
radius = 0.3

# Set up the initial positions and velocities of the agents
positions = np.random.uniform(size=(num_agents, 2))
velocities = np.random.uniform(size=(num_agents, 2)) * speed

# Define a function to update the velocities of the agents
def update_velocities(positions, velocities, radius, speed, noise):
    # Compute the distances between all pairs of agents

    distances = np.sqrt(np.sum((positions[:, np.newaxis] - positions[np.newaxis, :])**2, axis=2))
    
    # Find the indices of the neighbors within the specified radius
    neighbors = np.argwhere(distances < radius)
    
    # Compute the average direction of the neighbors
    mean_direction = np.empty([num_agents, 2])
    for i in range(num_agents):
        sum = 0
        count = 0
        for j in neighbors:
            if j[0] == i:
                sum+=j[0]
                count += 1
        if count > 0:
            mean_direction[i]==sum/count
        else:
            mean_direction[i]==0
    
    # Add some random noise to the direction
    noise_vector = np.random.normal(size=2) * noise
    mean_direction += noise_vector
    
    # Normalize the direction and set the velocity of each agent
    norm = np.sqrt(np.sum(mean_direction**2))
    if norm > 0:
        mean_direction /= norm
    velocities = mean_direction * speed
    
    return velocities

# Run the simulation and display the results
fig, ax = plt.subplots()

for i in range(100):
    # Update the velocities of the agents
    velocities = update_velocities(positions, velocities, radius, speed, noise)
    
    # Update the positions of the agents
    positions += velocities
    
    # Plot the agents as arrows
    ax.clear()
    ax.quiver(positions[:,0], positions[:,1], velocities[:,0], velocities[:,1], color='red', units='xy', scale=40,headwidth=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.pause(0.01)

plt.show()
plt.close()
