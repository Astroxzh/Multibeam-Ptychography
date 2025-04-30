'''
measure distance between probe after propagation.

only two spiral mask, measure peak distance value
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
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



#%%

#create file names
lensFocuLengths = [20]
steps = 2
filenames = []
for lensFocuLength in lensFocuLengths:
    if lensFocuLength % 2 == 0:
        ds = np.arange(2,lensFocuLength-steps-1, steps)
    elif lensFocuLength % 2 != 0:
        ds = np.arange(2, lensFocuLength-1-steps, steps)
    for dzs in ds:
        if lensFocuLength*1000 % 2 == 0:
            dm = np.arange(2,lensFocuLength-dzs, steps)
        elif lensFocuLength*1000 % 2 != 0:
            dm = np.arange(2, lensFocuLength-1-dzs, steps)
        filenames.append(f'f{lensFocuLength}_{dm[0]}-{dm[-1]}dm{steps}step_{dzs}ds.npy')

del filenames[0]
#%%
# dataSet = np.squeeze(dataSet)
# filename = 'f20_2-4dm2step_14ds.npy'
# filePath = os.path.join(folderPath, filename)
# dataSet = np.load(filePath)
folderPath = r'C:\Master Thesis\data\1 optimal probe touching\data'
resultDistance = np.zeros([7,7])
jj = 0

N = 8000
size = 10e-3
dx = size / N

for filename in filenames:
    filePath = os.path.join(folderPath, filename)
    dataSet = np.load(filePath)
    for ii in range(np.shape(dataSet)[0]):
        data = np.abs(dataSet[ii, :, :] ** 2)
        peaks = findPeak(data, peakDistance=70)

        peaksX = peaks[:, 0]
        peaksY = peaks[:, 1]

        peakGrid = peaks.reshape(4,4,2)
        hDiff = peakGrid[:, 1:, :] - peakGrid[:, :-1, :]
        hDistance = np.linalg.norm(hDiff, axis=2)
        vDiff = peakGrid[1:, :, :] - peakGrid[:-1, :, :]
        vDistance = np.linalg.norm(vDiff, axis=2)

        meanH = trim_mean(hDistance, proportiontocut=0.125, axis=None)
        meanV = trim_mean(vDistance, proportiontocut=0.125, axis=None)
        # meanAll = np.concatenate([hDistance.ravel(), vDistance.ravel()]).mean()
        meanTotal = np.mean([meanH, meanV])
        meanDistance = meanTotal * dx * 1000 #mm
        resultDistance[jj, ii] = meanDistance
    jj += 1

resultDistanceF = resultDistance[:,:-1]
# plt.figure()
# plt.imshow(np.log(data)+1)
# plt.scatter(peaksX, peaksY, s=20, marker='o', linewidths=1.2)

#coordinates
dsVals = np.arange(4, 18, 2)
dmVals = np.arange(2, 12, 2)

DM, DS = np.meshgrid(dmVals, dsVals)

# heat map
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(DM, DS, data, shading='auto')
plt.colorbar(pcm, label='mm')
plt.xlabel('dm (mm)')
plt.ylabel('ds (mm)')
plt.title('Probe Distance vs ds and dm')
plt.tight_layout()

#%%
#analyze and fit 4ds with changing dm and 2dm with changing ds

ds4mm = resultDistanceF[0, :]
dm2mm = resultDistanceF[:, 0]

coeds = np.polyfit(dmVals, ds4mm, 1)
fitds = np.poly1d(coeds)
dsFitx = np.linspace(dmVals[0], dmVals[-1], 100)
dsFity = fitds(dsFitx)

coedm = np.polyfit(dsVals, dm2mm, 1)
fitdm = np.poly1d(coedm)
dmFitx = np.linspace(dsVals[0], dsVals[-1], 100)
dmFity = fitdm(dmFitx)

plt.figure()
plt.scatter(dmVals, ds4mm)
plt.plot(dsFitx, dsFity)

plt.figure()
plt.scatter(dsVals, dm2mm)
plt.plot(dmFitx, dmFity)

#%%

# plt.axis('off')
plt.show()

#%%