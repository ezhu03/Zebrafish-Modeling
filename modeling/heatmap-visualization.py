import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
import random
import math
from time import perf_counter
import os

file_name = "modeling/data/const4radius0boxradius10iter10000fish1_15min.npz"
data = np.load(file_name)
lst = data.files
allxpos = data['x']
allypos = data['y']
plt.hist2d(allxpos, allypos, bins=(10, 10), cmap=sns.color_palette("light:b", as_cmap=True), density=True, vmin = 0, vmax = 0.015)


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