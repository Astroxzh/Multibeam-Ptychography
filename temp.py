#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

#%%
path = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f20\probeDistance'
filePath = os.path.join(path, 'f20_2-16dm2step_2ds.npy')
dataSet = np.load(filePath)
for i in range(np.shape(dataSet)[0]):
    plt.figure()
    plt.imshow(np.log(np.abs(dataSet[i,...])**2 + 1), cmap=utils.setCustomColorMap())
# %%
