import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math

# Set up the simulation parameters
box_radius = 10
num_agents = 20
speed = 0.05
noise = 0.005
radius = 1
time = 200
const = 10
# Set up the initial positions and velocities of the agents
angles = np.random.uniform(0, 2*np.pi, num_agents)
    
    # Generate random radii (distance from the center) uniformly distributed between 0 and the circle's radius
radii = np.random.uniform(0, box_radius, num_agents)
    
    # Convert polar coordinates to Cartesian coordinates (x, y)
x = radii * np.cos(angles)
y = radii * np.sin(angles)
    
positions = np.column_stack((x, y))
velocities = np.random.uniform(size=(num_agents, 2)) * speed

def reflection(r,x,y,vx,vy):
    magv = np.sqrt(vx **2 + vy**2)
    angles = np.arange(0,6.28,0.01)
    xbound = r*np.cos(angles) 
    ybound = r*np.sin(angles) 
    labels=np.zeros(len(angles))
    for i in range(len(angles)):
        magd = np.sqrt((xbound[i]-x)**2+(ybound[i]-y)**2)
        theta = np.arccos((xbound[i]*(xbound[i]-x)+ybound[i]*(ybound[i]-y))/(r*magd))
        phi = np.arccos((vx*(xbound[i]-x)+vy*(ybound[i]-y))/(magv*magd))
        if theta > 0.85 and theta < 2.29 and phi < 2.958:
            labels[i]=1
    return labels
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
    
    # Find the indices of the neighbors within the specified radius
    neighbors = np.argwhere(distances < radius)
    
    
    # Compute the average direction of the neighbors
    mean_direction = np.zeros((num_agents, 2))
    for i in range(num_agents):
        sum_direction = np.zeros(2)
        count = 0
        for j in neighbors:
            if j[0] == i:
                sum_direction += velocities[j[1]]
                count += 1
        if count > 0:
            mean_direction[i] = sum_direction / count
    
    # Add some random noise to the direction
    noise_vector = np.random.normal(size=(num_agents, 2)) * noise
    mean_direction += noise_vector
     # Normalize the direction and set the velocity of each agent
    norm = np.linalg.norm(mean_direction, axis=1)
    norm[norm == 0] = 1  # Avoid division by zero
    mean_direction /= norm[:, np.newaxis]
    for i in range(len(mean_direction)):
        if mean_direction[i][0] == 0 and mean_direction[i][1] == 0:
            velocities[i]+=noise_vector[i]
        else:
            velocities[i] = mean_direction[i] * speed + noise_vector[i]
    normv = np.linalg.norm(velocities, axis=1)
    normv[normv == 0] = 1  # Avoid division by zero
    for i in range(num_agents):
        velocities[i] = velocities[i]/normv[i] * speed
    # Normalize the direction and set the velocity of each agent
    #norm = np.linalg.norm(mean_direction, axis=1)
    #norm[norm == 0] = 1  # Avoid division by zero
    #mean_direction /= norm[:, np.newaxis]
    #for i in range(len(mean_direction)):
        #if mean_direction[i][0] == 0 and mean_direction[i][1] == 0:
            #velocities[i]=velocities[i]
        #else:
            #velocities[i] = mean_direction[i] * speed

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
disthist = np.zeros(num_agents)
for i in range(time):
    # Update the velocities of the agents
    velocities = update_velocities(positions, velocities, radius, speed, noise)
    
    # Update the positions of the agents


    for j in range(num_agents):
        distance = boundary_distance(box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1])
        weight = math.exp(-const*(distance*speed)/box_radius)
        sample = [0, 1]
        randomval= random.choices(sample, weights=(weight, 1-weight), k=1)
        if randomval[0] == 0:
            anglabels = (box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1])
            # Find indices where the value is 1
            indices_with_1 = [i for i, value in enumerate(my_array) if value == 1]

            # Pick a random index where the value is 1
            random_index = random.choice(indices_with_1)

            vector =[positions[j][0]+velocities[j][0]*distance,positions[j][1]+velocities[j][1]*distance]
            vecnorm = np.linalg.norm(vector)
            vector /= vecnorm
            vec_perp = np.dot(positions[j],vector)*vector
            vec_adj = vector - vec_perp
            vector = vec_adj - vec_perp
            veladj = np.random.normal(size=2) * noise
            velocities[j]=speed*vector/np.linalg.norm(vector) + veladj
    
    positions += velocities
            
    #positions %= box_size
    
    # Plot the agents as arrows
    ax.clear()
    circle = Circle([0,0], box_radius, edgecolor='b', facecolor='none')
    plt.gca().add_patch(circle)
    ax.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], color='red',
              units='xy', scale=0.1, headwidth=2)
    ax.set_xlim(-box_radius, box_radius)
    ax.set_ylim(-box_radius, box_radius)
    plt.pause(0.01)

plt.show()
#plt.close()
#plt.hist(disthist, bins=10)
#plt.show()
