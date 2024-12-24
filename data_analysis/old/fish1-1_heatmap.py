import numpy as np
from numpy import load
import matplotlib.pyplot as plt
file = "/Volumes/Hamilton/Derksen/AVI/9.13/session_fish1-1/trajectories/validated.npy"
def open_file(file_name):
    data = load(file_name, allow_pickle=True)
    lst = data.item()
    positions = lst['trajectories']
    return positions
positions = open_file(file)
print(positions)
#print(positions.ndim)
# Create a 2D histogram
plt.hist2d(positions[:,0,0], positions[:,0,1], bins=(20, 20), cmap=plt.cm.jet)

# Add labels and a colorbar
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Frequency')

# Show the plot
plt.show()
