import numpy as np
import matplotlib.pyplot as plt

def initialize_particles(num_particles, box_size):
    particles = np.random.rand(num_particles, 2) * box_size
    velocities = np.random.rand(num_particles, 2) - 0.5
    return particles, velocities

def update_particle_velocity(particles, velocities, radius, eta):
    num_particles = particles.shape[0]
    for i in range(num_particles):
        # Find neighbors within the given radius
        distances = np.linalg.norm(particles - particles[i], axis=1)
        neighbors = np.where(distances < radius)[0]
        
        # Calculate the average heading direction of neighbors
        mean_heading = np.mean(velocities[neighbors], axis=0)
        
        # Tune the particle's velocity based on the average heading direction
        theta = np.arctan2(mean_heading[1], mean_heading[0])
        velocities[i] = [np.cos(theta + eta), np.sin(theta + eta)]
    
    return velocities

def update_particle_position(particles, velocities, box_size):
    particles += 3*velocities
    particles %= box_size  # Periodic boundary conditions

def simulate_vicsek_model(num_particles, box_size, radius, eta, num_steps):
    particles, velocities = initialize_particles(num_particles, box_size)
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('Vicsek Model Simulation')
    
    for step in range(num_steps):
        update_particle_velocity(particles, velocities, radius, eta)
        update_particle_position(particles, velocities, box_size)
        
        # Plot the particles as arrows
        ax.cla()
        ax.quiver(particles[:, 0], particles[:, 1], velocities[:, 0], velocities[:, 1], color='b', scale=20, minshaft = 1)
        plt.pause(0.01)
    
    plt.show()

# Simulation parameters
num_particles = 1000
box_size = 100
radius = 10
eta = 0.1
num_steps = 100

# Run the simulation
simulate_vicsek_model(num_particles, box_size, radius, eta, num_steps)
