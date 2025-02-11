import numpy as np

import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('speeddistribution7dpf.npy')

# Plot the histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of 1D Array Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()