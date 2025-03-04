from scipy.stats import ks_2samp
import numpy as np
count = 0
for i in range(1000):   
    x=np.random.normal(0,0.5,100)
    y=np.random.normal(0,1,100)
    if ks_2samp(x,y)[1]<0.00001:
        count +=1
print(count)