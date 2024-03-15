import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
from time import perf_counter
import os

file_name = 'modeling/data/const100radius0boxradius10iter1000fish1.npz'
data = np.load(file_name)
lst = data.files
allxpos = data['x']
allypos = data['y']
plt.hist2d(allxpos, allypos, bins=(10, 10),range = [[-10,10],[-10,10]], cmap=plt.cm.jet, density=True, vmin = 0, vmax = 0.01)


# Add labels and a colorbar
plt.xlabel('X-bins')
plt.ylabel('Y-bins')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title('Heatmap for Half Tank')
plt.colorbar(label='Frequency')


# Show the plot
plt.show()
print(np.mean(allxpos),np.std(allxpos))