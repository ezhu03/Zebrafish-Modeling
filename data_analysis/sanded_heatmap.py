import numpy as np
from numpy import load
import matplotlib.pyplot as plt
file1 = "/Volumes/Hamilton/Derksen/AVI/9.13/session_fish1-2/trajectories/validated.npy"
file2 = "/Volumes/Hamilton/Derksen/AVI/9.13/session_fish2-3/trajectories/validated.npy"
file3 = "/Volumes/Hamilton/Derksen/AVI/9.13/session_fish3-1/trajectories/validated.npy"
def open_file(file_name):
    data = load(file_name, allow_pickle=True)
    lst = data.item()
    positions = lst['trajectories']
    return positions
positions1 = open_file(file1)
positions2 = open_file(file2)
positions3 = open_file(file3)

xpositions = []
ypositions = []
for position in positions1:
    xpositions.append(position[0,0])
    ypositions.append(position[0,1])
for position in positions2:
    xpositions.append(position[0,0])
    ypositions.append(position[0,1])
for position in positions3:
    xpositions.append(position[0,0])
    ypositions.append(position[0,1])


# Create a 2D histogram
plt.hist2d(xpositions, ypositions, bins=(20, 20), cmap=plt.cm.jet)

# Add labels and a colorbar
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Frequency')

# Show the plot
plt.show()
