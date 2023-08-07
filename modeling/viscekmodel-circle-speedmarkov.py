import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math

class MarkovChain:
    def __init__(self):
        self.transition_matrix = {
            'A': {'A': 0.99, 'B': 0.005, 'C': 0.005},
            'B': {'A': 0.005, 'B': 0.99, 'C': 0.005},
            'C': {'A': 0.001, 'B': 0.004, 'C': 0.995}
        }
        sampleList = ['A','B','C']
        self.current_state = random.choices(sampleList, weights=(10, 10, 10), k=1)[0]


    def next_state(self):
        transition_probabilities = self.transition_matrix[self.current_state]
        next_state = random.choices(list(transition_probabilities.keys()), list(transition_probabilities.values()))[0]
        self.current_state = next_state
        return next_state
    def get_state(self):
        return self.current_state

# Usage example

# Set up the simulation parameters
box_radius = 10
num_agents = 20
speed = 0.1*np.ones(num_agents)
noise = 0.01*np.ones(num_agents)
radius = np.ones(num_agents)
time = 100
const = 10
mc = []
colors = []
for i in range(num_agents):
    mc.append(MarkovChain())
    colors.append('red')
    
# Set up the initial positions and velocities of the agents
angles = np.random.uniform(0, 2*np.pi, num_agents)
    
    # Generate random radii (distance from the center) uniformly distributed between 0 and the circle's radius
radii = np.random.uniform(0, box_radius, num_agents)
    
    # Convert polar coordinates to Cartesian coordinates (x, y)
x = radii * np.cos(angles)
y = radii * np.sin(angles)
    
positions = np.column_stack((x, y))
velocities = np.random.uniform(size=(num_agents, 2)) * speed[0]
def boundary_distance(r,x,y,vx, vy):
    # Calculate the vector from the object to the center of the boundary
    b = 2*(x*vx+y*vy)
    a = (vx**2+vy**2)
    c = (x**2+y**2-r**2)
    return (-b+math.sqrt(b**2-4*a*c))/(2*a)

# Define a function to update the velocities of the agents
def update_velocities(positions, velocities, radius, speed, noise):
    # Compute the distances between all pairs of agents
    distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    noise_vector = []
    # Find the indices of the neighbors within the specified radius
    mean_direction = np.zeros((num_agents, 2))
    for i in range(num_agents):
        tempneigh = np.argwhere(distances<radius[i])
        sum_direction = np.zeros(2)
        count = 0
        for j in tempneigh:
            if j[0]==i:
                sum_direction += velocities[j[1]]*speed[j[1]]
                count += speed[j[1]] 
        if count > 0:
            mean_direction[i] = sum_direction / count
            norm = np.linalg.norm(mean_direction[i], axis=1)
            if norm == 0:
                norm = 1
            mean_direction[i] /= norm
        noise_vector.append(np.random.normal(size=2) * noise[i])
    
    # Normalize the direction and set the velocity of each agent
    for i in range(len(mean_direction)):
        if mean_direction[i][0] == 0 and mean_direction[i][1] == 0:
            velocities[i]+=noise_vector[i]
        else:
            velocities[i] = mean_direction[i] * speed[i] + noise_vector[i]
    normv = np.linalg.norm(velocities, axis=1)
    normv[normv == 0] = 1  # Avoid division by zero
    for i in range(num_agents):
        velocities[i] = velocities[i]/normv[i] * speed[i]
    #for i in range(num_agents):
        #sum_direction = np.zeros(2)
        #count = 0
        #for j in neighbors:
            #if j[0] == i:
                #weight = abs(np.dot(positions[j[1]]-positions[j[0]],velocities[j[0]])/speed)
                #sum_direction += weight * velocities[j[1]]
                #count += weight
        #if count > 0:
            #mean_direction[i] = sum_direction / count
        #if mean_direction[i][0] == 0 and mean_direction[i][1] == 0 :
            #mean_direction[i]=positions[i]
    
    # Add some random noise to the direction
    #noise_vector = np.random.normal(size=(num_agents, 2)) * noise
    #mean_direction += noise_vector
    
    # Normalize the direction and set the velocity of each agent
    #norm = np.linalg.norm(mean_direction, axis=1)
    #norm[norm == 0] = 1  # Avoid division by zero
    #mean_direction /= norm[:, np.newaxis]
    #for i in range(len(mean_direction)):
        #if mean_direction[i][0] == 0 and mean_direction[i][1] == 0:
            #velocities[i]=velocities[i]
        #else:
            #velocities[i] = mean_direction[i] * speed
    
    return velocities
# Run the simulation and display the results
fig, ax = plt.subplots()

randtime = random.randint(time/2,time)
#disthist = np.zeros(num_agents)
for i in range(time):
    # Update the velocities of the agents
    



    velocities = update_velocities(positions, velocities, radius, speed, noise)
    
    # Update the positions of the agents


    for j in range(num_agents):
        current_state = mc[j].next_state()
        #print(current_state)
        if current_state == 'A':
            speed[j] = 0.05
            colors[j] = 'black'
            noise[j] = 0.005
            radius[j] = 1
            const = 10
        elif current_state == 'B':
            speed[j] = 0.05
            colors[j] = 'blue'
            noise[j] = 0.005
            radius[j] = 0.1
            const = 15
        else:
            speed[j] = 0.005
            colors[j] = 'grey'
            noise[j] = 0.0005
            radius[j] = 0
            const = 20
        
        distance = boundary_distance(box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1])
        weight = math.exp(-const*(distance*speed[j])/box_radius)
        sample = [0, 1]
        randomval= random.choices(sample, weights=(weight, 1-weight), k=1)
        if randomval[0] == 0:
            vector =[positions[j][0]+velocities[j][0]*distance,positions[j][1]+velocities[j][1]*distance]
            vecnorm = np.linalg.norm(vector)
            vector /= vecnorm
            vec_perp = np.dot(positions[j],vector)*vector
            vec_adj = vector - vec_perp
            vector = vec_adj - vec_perp
            veladj = np.random.normal(size=2) * noise[j]
            velocities[j]=speed[j]*vector/np.linalg.norm(vector) + veladj
    
    positions += velocities
            
    #positions %= box_size
    
    # Plot the agents as arrows
    ax.clear()
    circle = Circle([0,0], box_radius, edgecolor='b', facecolor='none')
    plt.gca().add_patch(circle)
    ax.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], color=colors,
              units='xy', scale=0.1, headwidth=2)
    ax.set_xlim(-box_radius, box_radius)
    ax.set_ylim(-box_radius, box_radius)
    #if current_state == 'A':
    #    ax.set_title("Schooling")
    #elif current_state == 'B':
    #    ax.set_title("Swimming")
    #else:
    #    ax.set_title("Resting")
    plt.pause(0.001)
plt.show()
plt.close()
#plt.hist(disthist, bins=10)
#plt.show()
