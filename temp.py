#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
path = r'C:\Master Thesis\data\1 optimal probe touching\data'
filePath = os.path.join(path, 'f20_2-16dm2step_2ds.npy')
dataSet = np.load(filePath)
for i in range(np.shape(dataSet)[0]):
    plt.figure()
    plt.imshow(np.log(np.abs(dataSet[i,4000-1000:4000+1000, 4000-1000:4000+1000])**2 + 1))
# %%
