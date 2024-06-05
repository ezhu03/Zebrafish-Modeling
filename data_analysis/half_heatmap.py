import numpy as np
from numpy import load
import matplotlib.pyplot as plt
file1 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-1/trajectories/validated.npy"
file2 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-2/trajectories/validated.npy"
file3 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-3/trajectories/validated.npy"
file4 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-4/trajectories/validated.npy"
file5 = "/Volumes/Hamilton/Zebrafish/AVI/2.28.24/session_1fish15min1fps-half-5/trajectories/validated.npy"
file1 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-1-21dpf/trajectories/validated.npy"
file2 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-2-21dpf/trajectories/validated.npy"
file3 = "/Volumes/Hamilton/Zebrafish/AVI/3.13.24/session_1fish15min1fps-half-3-21dpf/trajectories/validated.npy"
def open_file(file_name):
    data = load(file_name, allow_pickle=True)
    lst = data.item()
    positions = lst['trajectories']
    return positions
positions1 = open_file(file1)
positions2 = open_file(file2)
positions3 = open_file(file3)
#positions4 = open_file(file4)
#positions5 = open_file(file5)
#print(positions2)

xpositions = []
ypositions = []
for position in positions1:
    if not np.isnan(position[0,0]):
        xpositions.append(position[0,0])
        ypositions.append(position[0,1])
for position in positions2:
    if not np.isnan(position[0,0]):
        xpositions.append(position[0,0])
        ypositions.append(position[0,1])
for position in positions3:
    if not np.isnan(position[0,0]):
        xpositions.append(position[0,0])
        ypositions.append(position[0,1])
'''for position in positions4:
    if not np.isnan(position[0,0]):
        xpositions.append(position[0,0])
        ypositions.append(position[0,1])
for position in positions5:
    if not np.isnan(position[0,0]):
        xpositions.append(position[0,0])
        ypositions.append(position[0,1])'''

xsc = [20 * x / 2048 for x in xpositions]
ysc = [20 * y / 2048 for y in ypositions]
print(xpositions)
# Create a 2D histogram
plt.hist2d(xsc, ysc, bins=(10, 10),range = [[0,20],[0,20]], cmap=plt.cm.jet, density=True, vmin = 0, vmax = 0.025)
print(np.mean(xsc)-10,np.std(xsc))

# Add labels and a colorbar
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.title('Heatmap for Half-Sanded Tank')
plt.colorbar(label='Frequency')

# Show the plot
plt.show()
