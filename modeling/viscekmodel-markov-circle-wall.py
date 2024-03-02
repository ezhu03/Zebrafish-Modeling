import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
from time import perf_counter

class MarkovChain:
    def __init__(self):
        self.transition_matrix = {
            'A': {'A': 0.01, 'B': 0.99},
            'B': {'A': 0.2, 'B': 0.8}
        }
        sampleList = ['A','B']
        self.current_state = random.choices(sampleList, weights=(10, 10), k=1)[0]


    def next_state(self):
        transition_probabilities = self.transition_matrix[self.current_state]
        next_state = random.choices(list(transition_probabilities.keys()), list(transition_probabilities.values()))[0]
        self.current_state = next_state
        return next_state
    def get_state(self):
        return self.current_state

# Set up the simulation parameters
box_radius = 10
num_agents = 25
speed = 0.1*np.ones((num_agents,1))
noise = 0.01*np.ones(num_agents)
time = 200
const = 1
<<<<<<< Updated upstream
radius = 2.5
=======
radius = 1
>>>>>>> Stashed changes

mc = []
for i in range(num_agents):
    mc.append(MarkovChain())
# Set up the initial positions and velocities of the agents
angles = np.random.uniform(0, 2*np.pi, num_agents)
    
    # Generate random radii (distance from the center) uniformly distributed between 0 and the circle's radius
radii = np.random.uniform(0, box_radius, num_agents)
    
    # Convert polar coordinates to Cartesian coordinates (x, y)
x = radii * np.cos(angles)
y = radii * np.sin(angles)
    
positions = np.column_stack((x, y))
velocities = np.random.uniform(size=(num_agents, 2)) * speed
print(velocities.shape)

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
    return (-b+math.sqrt(np.abs(b**2-4*a*c)))/(2*a)
# Define a function to update the velocities of the agents
def update_velocities(positions, velocities, radius, speed, noise):

    
    # Compute the distances between all pairs of agents
    distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    
    # Find the indices of the neighbors within the specified radius
    neighbors = np.argwhere(distances < radius)
    
    
    # Compute the average direction of the neighbors
    mean_direction = np.zeros((num_agents, 2))
    for i in range(num_agents):
        current_state = mc[i].next_state()
        #print(current_state)
        if current_state == 'A':
            speed[i] = 1.5+0.25*np.random.normal()
            noise[i] = 0.25
        elif current_state == 'B':
            speed[i] = 0.001
            noise[i] = 0.00001
        sum_direction = np.zeros(2)
        count = 0
        for j in neighbors:
            if j[0] == i:
                sum_direction += velocities[j[1]]
                count += 1
        if count > 0:
            mean_direction[i] = sum_direction / count
    
    #Add some random noise to the direction
    noise_vector = np.random.normal(size=(num_agents, 2)) * np.vstack((noise,noise)).transpose()
    mean_direction += noise_vector
     # Normalize the direction and set the velocity of each agent
    norm = np.linalg.norm(mean_direction, axis=1)
    norm[norm == 0] = 1  # Avoid division by zero
    mean_direction /= norm[:, np.newaxis]
    for i in range(len(mean_direction)):
        if mean_direction[i][0] == 0 and mean_direction[i][1] == 0:
            velocities[i]+=noise_vector[i]
        else:
            velocities[i] = mean_direction[i] * speed[i] + noise_vector[i]
    normv = np.linalg.norm(velocities, axis=1)
    normv[normv == 0] = 1  # Avoid division by zero
    for i in range(num_agents):
        velocities[i] = velocities[i]/normv[i] * speed[i] * np.random.uniform()
    #Normalize the direction and set the velocity of each agent
    norm = np.linalg.norm(mean_direction, axis=1)
    norm[norm == 0] = 1  # Avoid division by zero
    mean_direction /= norm[:, np.newaxis]
    for i in range(len(mean_direction)):
        if mean_direction[i][0] == 0 and mean_direction[i][1] == 0:
            velocities[i]=velocities[i]
        else:
            velocities[i] = mean_direction[i] * speed[i]

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
    t1 = perf_counter()
    # Update the velocities of the agents
    velocities = update_velocities(positions, velocities, radius, speed, noise)
    
    # Update the positions of the agents


    for j in range(num_agents):
        distance = boundary_distance(box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1])
<<<<<<< Updated upstream
=======
        #weight = math.exp(-const*(distance*speed[j])/box_radius)
>>>>>>> Stashed changes
        weight = math.exp(-const*(distance*speed[j])/box_radius)
        sample = [0, 1]
        randomval= random.choices(sample, weights=(weight, 1-weight), k=1)
        if randomval[0] == 0:
            anglabels = (box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1])
            # Find indices where the value is 1
            indices_with_1 = [i for i, value in enumerate(anglabels) if value == 1]
            if not indices_with_1:
                pass
            else:
            # Pick a random index where the value is 1
                random_index = random.choice(indices_with_1)

                #angle = random_index/100

                #vxn = np.cos(angle)*distance*0.01
                #vyn = np.sin(angle)*distance*0.01

                #vector =[vxn,vyn]
                #veladj = np.random.normal(size=2) * noise[j] * distance * 0.01
                #velocities[j]=vector+ veladj

                veladj = np.random.normal(size=2) * noise[j] * distance * 0.01
                xbord = distance*velocities[j][0]+positions[j][0]
                ybord = distance*velocities[j][1]+positions[j][1]

                dirr = np.cross([xbord,ybord],velocities[j])

                if dirr>0 and velocities[j][0]>0:
                    curr_angle = np.arctan(velocities[j][1]/velocities[j][0])
                    new_angle = curr_angle + np.random.normal(0.5, 0.25, 1)
                elif dirr>0:
                    curr_angle = np.pi + np.arctan(velocities[j][1]/velocities[j][0])
                    new_angle = curr_angle + np.random.normal(0.5, 0.25, 1)

                elif velocities[j][0]>0:
                    curr_angle = np.arctan(velocities[j][1]/velocities[j][0])
                    new_angle = curr_angle - np.random.normal(0.5, 0.25, 1)
                else:
                    curr_angle = np.pi + np.arctan(velocities[j][1]/velocities[j][0])
                    new_angle = curr_angle - np.random.normal(0.5, 0.25, 1)
                        
                velocities[j][0]=speed[j]*np.cos(new_angle)+veladj[0]
                velocities[j][1]=speed[j]*np.sin(new_angle)+veladj[1]
    
    newpositions = positions + velocities

    for j in range(num_agents):
        if newpositions[j][0]**2+newpositions[j][1]**2>box_radius**2-np.random.uniform()*noise[j]*box_radius**2:
            newpositions[j]=positions[j]
    positions = newpositions
            
    #positions %= box_size
    
    # Plot the agents as arrows
    ax.clear()
    circle = Circle([0,0], box_radius, edgecolor='b', facecolor='none')
    plt.gca().add_patch(circle)
    vstandard = np.random.uniform(size=(num_agents, 2))
    for j in range(num_agents):
        vs = np.sqrt(velocities[j,0]**2 + velocities[j,1]**2)
        vstandard[j][0]=velocities[j,0]/vs
        vstandard[j][1]=velocities[j,1]/vs

    ax.quiver(positions[:, 0], positions[:, 1], vstandard[:, 0], vstandard[:, 1], color='red',
              units='xy', scale=1, headwidth=2)
    ax.set_xlim(-box_radius, box_radius)
    ax.set_ylim(-box_radius, box_radius)
<<<<<<< Updated upstream
    plt.pause(1)
=======
    plt.pause(0.01)
>>>>>>> Stashed changes
    #t2 = perf_counter()
    #print(t2-t1)

plt.show()
#plt.close()
#plt.hist(disthist, bins=10)
#plt.show()
