import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import stats
import seaborn as sns
import pandas as pd

radius = 10

# Get input for the first position
xposition = float(input("Enter the x position: "))

# Get input for the second position
yposition = float(input("Enter the y position: "))

# Get input for the first position
xvelocity = float(input("Enter the x velocity: "))

# Get input for the second position
yvelocity = float(input("Enter the y velocity: "))

mag = np.sqrt(xposition **2 + yposition**2)
magv = np.sqrt(xvelocity **2 + yvelocity**2)
distance = 2*(10 - mag)

reflection = 0.85

angles = np.arange(0,6.28,0.01)
xbound = 10*np.cos(angles) 
ybound = 10*np.sin(angles) 
labels=np.zeros(len(angles))
for i in range(len(angles)):
    magd = np.sqrt((xbound[i]-xposition)**2+(ybound[i]-yposition)**2)
    theta = np.arccos((xbound[i]*(xbound[i]-xposition)+ybound[i]*(ybound[i]-yposition))/(radius*magd))
    phi = np.arccos((xvelocity*(xbound[i]-xposition)+yvelocity*(ybound[i]-yposition))/(magv*magd))
    if theta > 0.85 and theta < 2.29 and phi < 2.958:
        labels[i]=1
fig, ax = plt.subplots()
sns.scatterplot(x=xbound, y=ybound, hue=labels,palette = sns.color_palette("Set1",2))
ax.quiver(xposition, yposition, xvelocity/magv, yvelocity/magv)
plt.show()
