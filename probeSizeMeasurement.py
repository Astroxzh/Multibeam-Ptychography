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
import cv2

from skimage.measure import regionprops, label
from scipy.ndimage import gaussian_filter
#%%

#create file names
lensFocuLengths = [40]
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
folderPath = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f40\probeSize'
resultProbeSize = np.zeros([17,17])  #17for40, 7for20
jj = 0

N = 4000
size = 4e-3
dx = size / N
result = [[] for _ in range(4)]
fileCount = 0

# threshold = 0.6
for filename in filenames:
    print(f'{filename} is processing')
    filePath = os.path.join(folderPath, filename)
    dataSet = np.load(filePath)
    # temp = np.zeros([np.shape(dataSet)[0]])
    if fileCount > 143 :
        threshold = 0.1
    elif fileCount > 130:
        threshold = 0.3
    else:
        threshold = 0.6
    fileCount += 1
    for ii in range(np.shape(dataSet)[0]):
        data = np.abs(dataSet[ii, :, :]) ** 2
        cXmean, cYmean = utils.getCenter(data, thres=threshold)
        center = np.array([cYmean, cXmean])
        # xMax, yMax = np.unravel_index(np.argmax(data), data.shape)
        # # edge
        # H, W = data.shape
        # x1 = max(0, xMax - 1000)
        # x2 = min(H, xMax + 1000)
        # y1 = max(0, yMax - 1000)
        # y2 = min(W, yMax + 1000)
        # cropData = data[x1:x2, y1:y2]

        radius = utils.encircledEnergyRadiusSubpixel(data, center, fraction=0.8, pixel_size=(4/4000))
        # temp[ii] = radius
        result[ii].append(radius)
    # meanProbeSize = np.mean(temp)
    # result.append(meanProbeSize)


#%%

pattern = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# pattern = [7, 6, 5, 4, 3, 2, 1]
nRows = len(pattern)
nCols = max(pattern)

result1 = result[0]
result2 = result[1]
result3 = result[2]
result4 = result[3]

idx = 0
for i, length in enumerate(pattern):
    chunk = result1[idx: idx + length]
    resultProbeSize[i, :len(chunk)] = chunk
    idx += length

resultSizeFiltered = resultProbeSize[:,:-1]

fileName = 'ResultSize80%_probe627.npy'
filepath = os.path.join(folderPath, fileName)
np.save(filepath, np.array(resultSizeFiltered))


idx = 0
for i, length in enumerate(pattern):
    chunk = result2[idx: idx + length]
    resultProbeSize[i, :len(chunk)] = chunk
    idx += length

resultSizeFiltered = resultProbeSize[:,:-1]

fileName = 'ResultSize80%_probe591.npy'
filepath = os.path.join(folderPath, fileName)
np.save(filepath, np.array(resultSizeFiltered))


idx = 0
for i, length in enumerate(pattern):
    chunk = result3[idx: idx + length]
    resultProbeSize[i, :len(chunk)] = chunk
    idx += length

resultSizeFiltered = resultProbeSize[:,:-1]

fileName = 'ResultSize80%_probe563.npy'
filepath = os.path.join(folderPath, fileName)
np.save(filepath, np.array(resultSizeFiltered))


idx = 0
for i, length in enumerate(pattern):
    chunk = result4[idx: idx + length]
    resultProbeSize[i, :len(chunk)] = chunk
    idx += length

resultSizeFiltered = resultProbeSize[:,:-1]

fileName = 'ResultSize80%_probe541.npy'
filepath = os.path.join(folderPath, fileName)
np.save(filepath, np.array(resultSizeFiltered))

# #for 20mm
# dsVals = np.arange(4, 18, 2) 
# dmVals = np.arange(2, 14, 2)

# #for 40mm
# # dsVals = np.arange(4, 38, 2)
# # dmVals = np.arange(2, 34, 2)

# DM, DS = np.meshgrid(dmVals, dsVals)

# # heat map
# plt.figure(figsize=(6,5))
# pcm = plt.pcolormesh(DM, DS, resultSizeFiltered, shading='auto')
# plt.colorbar(pcm, label='mm')
# plt.xlabel('dm/mm')
# plt.ylabel('ds/mm')
# plt.title('probe size vs ds and dm \n @80% encircled energy')
# plt.tight_layout()

# #%%
# dm2mm = resultSizeFiltered[:, 0]
# coedm = np.polyfit(dsVals, dm2mm, 1)
# fitdm = np.poly1d(coedm)
# dmFitx = np.linspace(dsVals[0], dsVals[-1], 100)
# dmFity = fitdm(dmFitx)
# dmPred = fitdm(dsVals)
# residualsDmFit = dm2mm - dmPred

# plt.figure()
# fig, (axData, axResid) = plt.subplots(
#     2, 1,
#     sharex=True,
#     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1},
#     figsize=(6, 6)
# )
# axData.scatter(dsVals, dm2mm, label='data')
# axData.plot(dmFitx, dmFity, label=f'{fitdm[1]:.5f}$x$+{fitdm[0]:.5f}')
# axData.set_title('probe size vs ds, @dm=2mm')
# axResid.set_xlabel('ds/mm')
# axData.set_ylabel('probe size/mm')
# axData.legend()

# axResid.scatter(dsVals, residualsDmFit)
# axResid.set_ylabel('residual')
# axResid.axhline(0, c='gray')

# ds4mm = resultSizeFiltered[0,:]
# coeds = np.polyfit(dmVals, ds4mm, 1)
# fitds = np.poly1d(coeds)
# dsFitx = np.linspace(dmVals[0], dmVals[-1], 100)
# dsFity = fitds(dsFitx)
# dsPred = fitds(dmVals)
# residualsDsFit = ds4mm - dsPred

# plt.figure()
# fig, (axData, axResid) = plt.subplots(
#     2, 1,
#     sharex=True,
#     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1},
#     figsize=(6, 6)
# )
# axData.scatter(dmVals, ds4mm, label='data')
# axData.plot(dsFitx, dsFity, label=f'{fitds[1]:.5f}$x$+{fitds[0]:.5f}')
# axData.set_title('probe size vs dm, @ds=4mm')
# axResid.set_xlabel('dm/mm')
# axData.set_ylabel('probe size/mm')
# axData.legend()

# axResid.scatter(dmVals, residualsDsFit)
# axResid.set_ylabel('residual')
# axResid.axhline(0, c='gray')

# #%%
# plt.show()
# %%
