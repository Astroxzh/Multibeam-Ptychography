#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from scipy.stats import trim_mean

#%%

def findPeak(data, peakNum: int = 16, peakDistance: int = 15):
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

folderPath = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f20\probeDistance'
fileName = 'f20_2-12dm2step_6ds.npy'
filePath = os.path.join(folderPath, fileName)
dataSet = np.load(filePath)
data = np.abs(dataSet[0,...])**2
peaks = findPeak(data, peakDistance=130)

peaksX = peaks[:, 0]
peaksY = peaks[:, 1]

peakGrid = peaks.reshape(4,4,2)
hDiff = peakGrid[:, 1:, :] - peakGrid[:, :-1, :]
hDistance = np.linalg.norm(hDiff, axis=2)
vDiff = peakGrid[1:, :, :] - peakGrid[:-1, :, :]
vDistance = np.linalg.norm(vDiff, axis=2)

meanH = trim_mean(hDistance, proportiontocut=0.125, axis=None)
meanV = trim_mean(vDistance, proportiontocut=0.125, axis=None)

N = 4000
size = 4e-3  #for old f20mm data, new should be 4000 and 4e-3
dx = size / N
meanTotal = np.mean([meanH, meanV])
meanDistance = meanTotal * dx

plt.imshow((data), cmap=utils.setCustomColorMap())
plt.scatter(peaksX, peaksY, s=12, c='red')
plt.show()
# %%
