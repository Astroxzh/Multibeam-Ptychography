#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from scipy.stats import trim_mean

#%%

folderPath = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f40\probeSize'
fileName = 'f20_6dm2step_4ds.npy'
filePath = os.path.join(folderPath, fileName)

dataSet = np.load(filePath)

#%%
N = 8000
size = 10e-3
dx = size / N

fig, ax = plt.subplots(1,4, sharex=True, sharey=True)
ax = np.ravel(ax)

localN = 1000
localx = np.arange(-localN / 2, localN / 2) * dx
localCoor = [localx[0]*1000, localx[-1]*1000, localx[0]*1000, localx[-1]*1000] 

for ii in range(np.shape(dataSet)[0]):
    data = np.abs(dataSet[ii, :, :])**2
    xMax, yMax = np.unravel_index(np.argmax(data), data.shape)
    cropData = data[xMax-500:xMax+500, yMax-500:yMax+500]
    ax[ii].imshow(cropData, extent=localCoor)
    
# fig.supxlabel("mm")
# fig.supylabel("mm", x=0.001, fontsize=12)

#%%
plt.tight_layout()
plt.show()
#%%

