import numpy as np
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt

# Constants
R = 5
const = 3

# Define the function p(r)
def p(r):
    return np.exp(-const * r / R)

# Create an array of r values
r_values = np.linspace(0, 20, 400)
p_values = p(r_values)

#plt.style.use('seaborn')
fig, ax = plt.subplots(figsize=(8, 5))
plt.rcParams['figure.dpi'] = 100
ax.plot(r_values, p_values, linewidth=2, color='darkblue')
ax.set_xlabel("Distance from Boundary")
ax.set_ylabel("Turning Probability")
plt.xlim(0, R*2)
ax.set_title("Turning Probability Function")
# Only show the major grid with dashed lines to keep it less dense
ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)

# Create a custom legend handle that shows a single dot
custom_legend = Line2D([], [], color='darkblue', marker='o', linestyle='None', markersize=8,
                         label="$p(d) = e^{-\gamma d/R}$ \n $\gamma = 3$ \n $R=5$")
plt.legend(handles=[custom_legend])
plt.savefig('/Users/ezhu/Downloads/turning_probability_function3.png', dpi=3000, bbox_inches='tight')
plt.show()