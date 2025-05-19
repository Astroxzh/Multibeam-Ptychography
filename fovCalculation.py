#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from scipy.stats import trim_mean

#%%
pathProbeSize = r'C:\Master Thesis\data\1 optimal probe touching\data\f20_probeSizeMeasurement\data\ResultSize90%.npy'
pathProbeDistance = r'C:\Master Thesis\data\1 optimal probe touching\data\f20_fullMaskPattern\ResultDistance.npy'

resultDistance = np.load(pathProbeDistance)
resultProbeSize = np.load(pathProbeSize)

#%%

#coordinates
dsVals = np.arange(4, 18, 2)
dmVals = np.arange(2, 14, 2)

DM, DS = np.meshgrid(dmVals, dsVals)

resultDistanceRe = np.ravel(resultDistance)
resultProbeSizeRe = np.ravel(resultProbeSize)

#%%
# overlap = overlapCal(resultProbeSize, resultDistance)
# overlap = utils.area_overlap(2*resultProbeSize, resultDistance)
overlap = np.zeros_like(resultDistanceRe)

for ii in range(np.shape(resultDistanceRe)[0]):
    overlap[ii] = utils.fov(resultProbeSizeRe[ii], resultDistanceRe[ii])

overlap = np.reshape(overlap, np.shape(resultDistance))

#%%
# heat map
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(DM, DS, overlap, shading='auto')
plt.colorbar(pcm, label='mm')
plt.xlabel('dm/mm')
plt.ylabel('ds/mm')
plt.title('fov vs ds and dm \n @90% encircled energy')
plt.tight_layout()

plt.show()
#%%