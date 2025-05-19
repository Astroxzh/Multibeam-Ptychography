#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
path = r'C:\Master Thesis\data\1 optimal probe touching\data\f40\probeSize'
filePath = os.path.join(path, 'f40_2dm2step_4ds.npy')
dataSet = np.load(filePath)
for i in range(np.shape(dataSet)[0]):
    plt.figure()
    plt.imshow(np.log(np.abs(dataSet[i,...])**2 + 1))
# %%
