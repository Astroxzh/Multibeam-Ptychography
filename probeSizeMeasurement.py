'''
aiming to measure the probe size after propagation

method:
use only one spiral mask do the same propagation, then measure the probe size.
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os

from skimage.measure import regionprops, label

#%%
def findPeak(data, peakNum: int = 1, peakDistance: int = 15):
    dataCopy = data.copy()
    peakList = []
    for i in range(peakNum):
        maxCoord = np.unravel_index(np.argmax(dataCopy), dataCopy.shape)
        peakList.append(list(maxCoord))
        i, j = maxCoord
        rowMin = max(0, i - peakDistance)
        rowMax = min(dataCopy.shape[0], i + peakDistance)
        colMin = max(0, j - peakDistance)
        colMax = min(dataCopy.shape[1], j + peakDistance)
        dataCopy[rowMin:rowMax, colMin:colMax] = 0
    coords = np.array(peakList)
    sortedIdx = np.argsort(coords[:,1])
    sortedCoords = coords[sortedIdx]
    a = int(np.sqrt(peakNum))
    for i in range(a):
        row = sortedCoords[i * a:(i+1) * a, :]
        sortedRowIdx = np.argsort(row[:, 0])
        sortedRow = row[sortedRowIdx]
        sortedCoords[i * a:(i+1) * a, :] = sortedRow
    
    return sortedCoords


#%%

#create file names
lensFocuLengths = [20]
steps = 2
filenames = []
for lensFocuLength in lensFocuLengths:
    if lensFocuLength % 2 == 0:
        ds = np.arange(4,lensFocuLength-steps-1, steps)
    elif lensFocuLength % 2 != 0:
        ds = np.arange(4, lensFocuLength-1-steps, steps)
    for dzs in ds:
        if lensFocuLength*1000 % 2 == 0:
            dm = np.arange(2,lensFocuLength-dzs, steps)
        elif lensFocuLength*1000 % 2 != 0:
            dm = np.arange(2, lensFocuLength-1-dzs, steps)
        for dzm in dm:
            filenames.append(f'f{lensFocuLength}_{dzm}dm{steps}step_{dzs}ds.npy')


#%%
# dataSet = np.squeeze(dataSet)
# filename = 'f20_2-4dm2step_14ds.npy'
# filePath = os.path.join(folderPath, filename)
# dataSet = np.load(filePath)
folderPath = r'C:\Master Thesis\data\1 optimal probe touching\data'
resultProbeSize = np.zeros([7,7])
jj = 0

N = 8000
size = 10e-3
dx = size / N
result = []
for filename in filenames:
    filePath = os.path.join(folderPath, filename)
    dataSet = np.load(filePath)
    temp = np.zeros([np.shape(dataSet)[2]])
    for ii in range(np.shape(dataSet)[2]):
        data = np.abs(dataSet[:, :, ii] ** 2)
        # peaks = findPeak(data, peakNum=1)
        # peaksX = peaks[:, 0]
        # peaksy = peaks[:, 1]
        mask = data > data.max()*0.05
        labelImg = label(mask)

        props = regionprops(label_image=labelImg, intensity_image=data)
        p = props[0]

        Ixx, Ixy, Iyy = p.inertia_tensor.flat
        E = p.imtensity_image.sum()
        sigmaR = np.sqrt((Ixx + Iyy) / E)
        rmsRadius = sigmaR * dx
        temp[ii] = rmsRadius
    meanProbeSize = np.mean(temp)
    result.append(meanProbeSize)

#%%

pattern = [7, 6, 5, 4, 3, 2, 1]
nRows = len(pattern)
nCols = max(pattern)

idx = 0
for i, length in enumerate(pattern):
    chunk = result[idx: idx + length]
    resultProbeSize[i, :chunk.size] = chunk
    idx += length

resultSizeFiltered = resultProbeSize[:,:-1]

dsVals = np.arange(4, 18, 2)
dmVals = np.arange(2, 14, 2)

DM, DS = np.meshgrid(dmVals, dsVals)

# heat map
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(DM, DS, resultSizeFiltered, shading='auto')
plt.colorbar(pcm, label='mm')
plt.xlabel('dm/mm')
plt.ylabel('ds/mm')
plt.title('probe size vs ds and dm')
plt.tight_layout()



#%%
plt.show()
# %%
