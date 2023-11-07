import numpy as np
from numpy import load
import matplotlib.pyplot as plt
file = "/Users/eric/Documents/session_25fish_15min_10fps/trajectories/validated.npy"
#file2 = "/Volumes/Hamilton/Derksen/AVI/9.13/session_fish2-2/trajectories/validated.npy"
#file3 = "/Volumes/Hamilton/Derksen/AVI/9.13/session_fish3-3/trajectories/validated.npy"
def open_file(file_name):
    data = load(file_name, allow_pickle=True)
    lst = data.item()
    positions = lst['trajectories']
    return positions
positions = open_file(file)
#positions3 = open_file(file3)
print(positions)
print(positions.shape[0])
positions = np.reshape(positions, (positions.shape[0]*positions.shape[1] , 2))
xpositions = []
ypositions = []
for position in positions:
    xpositions.append(position[0])
    ypositions.append(position[1])
#for position in positions3:
#    xpositions.append(position[0,0])
#    ypositions.append(position[0,1])

xsc = [x / 100 for x in xpositions]
ysc = [y / 100 for y in ypositions]
# Create a 2D histogram

plt.hist2d(xsc, ysc, bins=(20, 20), cmap=plt.cm.jet, density=True, vmin = 0, vmax = 0.05)

# Add labels and a colorbar
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for 25 Fish Clear Tank')
plt.colorbar(label='Frequency')

# Show the plot
plt.show()
