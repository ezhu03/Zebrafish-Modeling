import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
from time import perf_counter
import os
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

day = int(input('dpf: '))
spds = np.load('speeddistribution'+str(day)+'dpf.npy')
sv = input('save (Y/N): ')
class MarkovChain:
    def __init__(self):
        self.transition_matrix = {
            'A': {'A': 1}
        }
        self.current_state = 'A'


    def next_state(self):
        transition_probabilities = self.transition_matrix[self.current_state]
        next_state = random.choices(list(transition_probabilities.keys()), list(transition_probabilities.values()))[0]
        self.current_state = next_state
        return next_state
    def get_state(self):
        return self.current_state




def boundary_distance(r, x, y, vx, vy):
    # Ensure that the velocity is not zero.
    if vx == 0 and vy == 0:
        raise ValueError("Velocity cannot be zero.")
    
    # Coefficients for the quadratic equation a*t^2 + b*t + c = 0
    a = vx*vx + vy*vy
    b = 2 * (x*vx + y*vy)
    c = x*x + y*y - r*r

    # Compute the discriminant
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        # This case should not happen if (x,y) is inside the circle.
        raise ValueError("No intersection found; check that (x,y) is inside the circle.")
    
    # We choose the positive root
    t = (-b + math.sqrt(discriminant)) / (2 * a)
    
    # The distance is the time multiplied by the speed (sqrt(a))
    distance = t * math.sqrt(a)
    return distance

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
        speed[i] = random.choice(spds)
        noise[i] = noise_ratio*speed[i]
        sum_direction = np.zeros(2)
        count = 0
        for j in neighbors:
            if j[0] == i:
                sum_direction += velocities[j[1]]
                count += 1
        if count > 0:
            mean_direction[i] = sum_direction / count
        else:
            mean_direction[i] = velocities[i]
    
    #Add some random noise to the direction
    theta = np.random.uniform(0, 2*np.pi, num_agents)  # Generate n random angles
    unit_vectors = np.column_stack((np.cos(theta), np.sin(theta)))  # Convert to 2D vectors
    noise_vector = unit_vectors * np.vstack((noise,noise)).transpose()
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
        velocities[i] = velocities[i]/normv[i] * speed[i]
    """#Normalize the direction and set the velocity of each agent
    norm = np.linalg.norm(mean_direction, axis=1)
    norm[norm == 0] = 1  # Avoid division by zero
    mean_direction /= norm[:, np.newaxis]
    for i in range(len(mean_direction)):
        if mean_direction[i][0] == 0 and mean_direction[i][1] == 0:
            velocities[i]=velocities[i]
            print('no direction')
        else:
            velocities[i] = mean_direction[i] * speed[i]"""
            

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

iterations=10
allxpos = []
allypos = []
for a in range(iterations):
    # Set up the simulation parameters
    box_radius = 5
    num_agents = 1
    speed = np.zeros((num_agents,1))
    noise = np.zeros(num_agents)
    time = 1200
    const = 1
    radius = 0
    starttime=300
    noise_ratio = 0.6
    
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
    spd = random.choice(spds)
    velocities = np.ones((num_agents, 2)) * spd
    print(a, "...running...")
    #fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(time):
        t1 = perf_counter()
        # Update the velocities of the agents
        velocities = update_velocities(positions, velocities, radius, speed, noise)
        #print(velocities)
        # Update the positions of the agents


        for j in range(num_agents):
            distance =  np.abs(boundary_distance(box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1]))
            weight = math.exp(-const*(distance)/box_radius)
       #print(weight)
            move = np.array([positions[j][0]+velocities[j][0],positions[j][1]+velocities[j][1]])
            if move[0]**2+move[1]**2>(box_radius-0.4)**2:
                print("movement failed")
                weight = 1
            if weight>1: 
                weight = 1
        #weight = math.exp(-const*(distance*speed[j])/box_radius)
            sample = [0, 1]
            randomval= random.choices(sample, weights=(weight, 1-weight), k=1)
            if randomval[0] == 0:
            
            # Pick a random index where the value is 1
                
                truth = True
                while(truth):
                    theta = np.random.uniform(0, 2*np.pi)  # Random angle in [0, 2π)
                    unit_vector = np.array([np.cos(theta), np.sin(theta)])
                    veladj = unit_vector * noise[j]
                    xbord = distance*velocities[j][0]+positions[j][0]
                    ybord = distance*velocities[j][1]+positions[j][1]
                    dirr = np.cross([xbord,ybord],velocities[j])
                
                    anglabels = reflection(box_radius,positions[j][0],positions[j][1],velocities[j][0],velocities[j][1])
                    # Find indices where the value is 1
                    indices_with_1 = [i for i, value in enumerate(anglabels) if value == 1]
                    if not indices_with_1:
                        print(len(indices_with_1))
                        print('failing here1')
                        noise_ratio = 1
                        theta = np.random.uniform(0, 2*np.pi)  # Random angle in [0, 2π)
                        speed[j] = random.choice(spds)
                        noise[j] = noise_ratio*speed[j]
                        theta = np.random.uniform(0, 2*np.pi)  # Random angle in [0, 2π)
                        unit_vector = np.array([np.cos(theta), np.sin(theta)])
                        norm_v = np.linalg.norm(velocities[j])
                        veladj = unit_vector * noise[j]
                        velocities[j]=speed[j]*norm_v+veladj
                        noise_ratio = 0.6

                        '''xbord = distance*velocities[j][0]+positions[j][0]
                        ybord = distance*velocities[j][1]+positions[j][1]
                        dirr = np.cross([xbord,ybord],velocities[j])
                        

                        if dirr>0 and velocities[j][0]>0:
                            curr_angle = np.arctan(velocities[j][1]/velocities[j][0])
                            new_angle = curr_angle + np.random.normal(0.75, 0.5, 1)
                        elif dirr>0:
                            curr_angle = np.pi + np.arctan(velocities[j][1]/velocities[j][0])
                            new_angle = curr_angle + np.random.normal(0.75, 0.5, 1)

                        elif velocities[j][0]>0:
                            curr_angle = np.arctan(velocities[j][1]/velocities[j][0])
                            new_angle = curr_angle - np.random.normal(0.75, 0.5, 1)
                        else:
                            curr_angle = np.pi + np.arctan(velocities[j][1]/velocities[j][0])
                            new_angle = curr_angle - np.random.normal(0.75, 0.5, 1)
                                        
                        velocities[j][0]=speed[j]*np.cos(new_angle)+veladj[0]
                        velocities[j][1]=speed[j]*np.sin(new_angle)+veladj[1]'''
                        move = np.array([positions[j][0]+velocities[j][0],positions[j][1]+velocities[j][1]])
                        if move[0]**2+move[1]**2<(box_radius-0.4)**2:
                            truth = False
                            print("sucess")
                        else:
                            print(move)
                    else:
                        #print('failing here2')
                        noise_ratio = 1
                        speed[j] = random.choice(spds)
                        noise[j] = noise_ratio*speed[j]
                        theta = np.random.uniform(0, 2*np.pi)  # Random angle in [0, 2π)
                        unit_vector = np.array([np.cos(theta), np.sin(theta)])
                        veladj = unit_vector * noise[j]
                        random_index = random.choice(indices_with_1)
                        print(random_index)
                        angle = random_index/100
                        rx = box_radius*np.cos(angle)
                        ry = box_radius*np.sin(angle)
                        vx = rx - positions[j][0]
                        vy = ry - positions[j][1]
                        vn = np.array([vx,vy])
                        vn = np.linalg.norm(vn)
                        new_v = vn * speed[j]
                            #print(vector,veladj)
                        velocities[j]=new_v+ veladj
                        move = np.array([positions[j][0]+velocities[j][0],positions[j][1]+velocities[j][1]])
                        if move[0]**2+move[1]**2<(box_radius-0.4)**2:
                            truth = False
                        noise_ratio = 0.6
                
        newpositions = positions + velocities

        for j in range(num_agents):
            if newpositions[j][0]**2+newpositions[j][1]**2>(box_radius-0.4)**2:
                newpositions[j]=positions[j]

        positions = newpositions
        if i%100 == 0:
            print("time: ",i)
        #positions %= box_size
        if(i>starttime):
            for p in positions:
                allxpos.append(p[0])
                allypos.append(p[1])
        # Plot the agents as arrows
        #ax.clear()
        #circle = Circle([0,0], box_radius, edgecolor='b', facecolor='none')
        #plt.gca().add_patch(circle)
        vstandard = np.random.uniform(size=(num_agents, 2))
        for j in range(num_agents):
            vs = np.sqrt(velocities[j,0]**2 + velocities[j,1]**2)
            vstandard[j][0]=velocities[j,0]/vs
            vstandard[j][1]=velocities[j,1]/vs

        
        #ax.quiver(positions[:, 0], positions[:, 1], vstandard[:, 0], vstandard[:, 1], color='red',
        #        units='xy', scale=1, headwidth=2)
        #ax.set_xlim(-box_radius, box_radius)
        #ax.set_ylim(-box_radius, box_radius)
        #plt.pause(0.005)
        #t2 = perf_counter()
        #print(t2-t1)

    #plt.show()

    #for position in positions3:
    #    xpositions.append(position[0,0])
    #    ypositions.append(position[0,1])
    if a == 0 or a==1: 
        fig, ax = plt.subplots(figsize=(6,6))
        center = (0, 0)
        theta = np.linspace(0, 2 * np.pi, 300)
        xc = center[0] + box_radius * np.cos(theta)
        yc = center[1] + box_radius * np.sin(theta)
        plt.plot(xc, yc, label=f'Circle with radius {radius}')

        # Generate a color gradient
        norm = mcolors.Normalize(vmin=0, vmax=len(allxpos))
        cmap = sns.color_palette("light:b", as_cmap=True)
        colors = [cmap(norm(i)) for i in range(len(allxpos) - 1)]
        print(len(allxpos))
        # Plot arrows between successive points

# Plot the trajectory
        for i in range(len(allxpos) - 1):
            ax.arrow(allxpos[i], allypos[i],
                    allxpos[i+1] - allxpos[i], allypos[i+1] - allypos[i], 
                    head_width=0.05, head_length=0.05, 
                    fc=cmap(norm(i)), ec=cmap(norm(i)), alpha=0.75)
        plt.title("Simulated Path")
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required to initialize data for colorbar
        #cbar = fig.colorbar(sm, ax=ax)  # Assign colorbar to figure and axis
        #cbar.set_label("Time(s)")
        #cbar.set_ticks([0,300,600,900])
        plt.grid(False)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()
center = (0, 0)

    # Create circle
theta = np.linspace(0, 2 * np.pi, 300)
xc = center[0] + box_radius * np.cos(theta)
yc = center[1] + box_radius * np.sin(theta)
plt.figure(figsize=(3, 3))
plt.plot(xc, yc, label=f'Circle with radius {radius}')
plt.hist2d(allxpos, allypos, bins=(10, 10),range = [[-1*box_radius,box_radius],[-1*box_radius,box_radius]],  cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.05)

# Add labels and a colorbar
#plt.xlabel('X-bins')
#plt.ylabel('Y-bins')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
#plt.title('Heatmap for Half Tank')
#plt.colorbar(label='Frequency')
arrsize = len(allxpos)
data = np.array([allxpos,allypos]).T
print(data)
if sv == 'Y':
    os.chdir('modeling/data')
    file_name = 'const%sradius%sboxradius%siter%sfish%s_15min_%sdpf_clear.npy'%(const,radius,box_radius,iterations,num_agents,day)
    with open(file_name, 'w') as file:
        pass
    np.save(file_name, data)
# Show the plot
plt.show()
print(np.mean(allxpos),np.std(allxpos))
