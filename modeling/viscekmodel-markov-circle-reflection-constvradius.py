import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
from time import perf_counter
from scipy import stats
import seaborn as sns
import pandas as pd

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
            speed[i] = speed0+speed0/noiseratio*np.random.normal()
            noise[i] = speed0/noiseratio
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

# Set up the simulation parameters

consts = [1,3,5,7,9]
radiuslist = [0,0.5,1,1.5,2,5,10]
rc = len(consts)
rl = len(radiuslist)
fig, axs = plt.subplots(rc, rl, figsize=(8, 6))
for k in range(rc):
    for l in range(rl):
        box_radius = 4.5
        num_agents = 25
        speed = 0.1*np.ones((num_agents,1))
        noise = 0.01*np.ones(num_agents)
        speed0 = 1
        noiseratio = 10
        time = 2000
        begin = 1400
        radius = radiuslist[l]
        const = consts[k]
        sc = []
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
            #print(velocities.shape)

# Run the simulation and display the results
#fig, ax = plt.subplots()
        for i in range(time):
    #t1 = perf_counter()
    # Update the velocities of the agents
            velocities = update_velocities(positions, velocities, radius, speed, noise)
    
    # Update the positions of the agents


            for j in range(num_agents):
                distance = boundary_distance(box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1])
                #weight = math.exp(-const*(distance*speed[j])/box_radius)
                weight = math.exp(-const*(distance)/box_radius)
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

                        angle = random_index/100

                        vxn = np.cos(angle)*distance*0.01
                        vyn = np.sin(angle)*distance*0.01

                        vector =[vxn,vyn]
                        veladj = np.random.normal(size=2) * noise[j] * distance * 0.01
                        velocities[j]=vector+ veladj
    
            newpositions = positions + velocities

            for j in range(num_agents):
                if newpositions[j][0]**2+newpositions[j][1]**2>box_radius**2-np.random.uniform()*noise[j]*box_radius**2:
                    newpositions[j]=positions[j]
            positions = newpositions
            
    #positions %= box_size
    
    # Plot the agents as arrows
    #ax.clear()
    #circle = Circle([0,0], box_radius, edgecolor='b', facecolor='none')
    #plt.gca().add_patch(circle)
    #vstandard = np.random.uniform(size=(num_agents, 2))
    #for j in range(num_agents):
    #    vs = np.sqrt(velocities[j,0]**2 + velocities[j,1]**2)
    #    vstandard[j][0]=velocities[j,0]/vs
    #    vstandard[j][1]=velocities[j,1]/vs
            if i > begin:
                sc.append(positions)
    #ax.quiver(positions[:, 0], positions[:, 1], vstandard[:, 0], vstandard[:, 1], color='red',
    #          units='xy', scale=1, headwidth=2)
    #ax.set_xlim(-box_radius, box_radius)
    #ax.set_ylim(-box_radius, box_radius)
    #plt.pause(0.095)
    #t2 = perf_counter()
    #print(t2-t1)

#plt.show()
#plt.close()
        sc = np.array(sc)
        msdc = []
        meanc = []
        varc = []
        for i in range(sc.shape[0]):
            msd = (sc[i,:,0]-sc[0,:,0])**2+(sc[i,:,1]-sc[0,:,1])**2
            msdc.append(msd)
            meanc.append(np.mean(msd))
            varc.append(1.96*np.std(msd)/np.sqrt(sc.shape[1]))
    
#sns.lineplot(x=range(len(meanc)), y=meanc, errorbar = ("ci",95))
#plt.show()
        df = pd.DataFrame(msdc)

# Calculate mean and confidence interval for each row
        row_means = df.mean(axis=1)
        confidence_intervals = df.sem(axis=1)  # Assuming normal distribution

# Create a line plot with 95% confidence interval
        #axs[9-k][j].figure(figsize=(10, 6))
        print(len(consts)-1-k)
        print(l)
        sns.lineplot(x=df.index, y=row_means,ax=axs[len(consts)-1-k][l])
        axs[len(consts)-1-k][l].fill_between(df.index, row_means - confidence_intervals, row_means + confidence_intervals, alpha=0.2)
        axs[len(consts)-1-k][l].set_ylim(0,30)
        if l==0:
            axs[len(consts)-1-k][l].set_ylabel(consts[k])
        else: 
            axs[len(consts)-1-k][l].set_ylabel('')
            axs[len(consts)-1-k][l].set_yticks([])
            axs[len(consts)-1-k][l].tick_params(axis='y', which='both', left=False, right=False)
        
        if k == 0:
            axs[len(consts)-1-k][l].set_xlabel(radiuslist[l])
        else: 
            axs[len(consts)-1-k][l].set_xlabel('')
            axs[len(consts)-1-k][l].set_xticks([])
            axs[len(consts)-1-k][l].tick_params(axis='x', which='both', top=False, bottom=False)
        
fig.supxlabel('radius')
fig.supylabel('const')
plt.show()
#plt.show()
