#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os

#%%
wavelengths = [627e-9, 591e-9, 563e-9, 541e-9]
# k = 2 * np.pi / wavelength
N = 4000

#%%
#create mask (spiral aperture)
maskSize = 4e-3
maskN = N
maskdx = maskSize / maskN

maskx = np.arange(-maskN / 2, maskN / 2) * maskdx
[maskX, maskY] = np.meshgrid(maskx, maskx)

maskNum = 4
localMaskN = maskN / maskNum
localMaskSize = maskSize / maskNum
localMaskdx = localMaskSize / localMaskN
localMaskx = np.arange(-localMaskN / 2, localMaskN / 2) * localMaskdx
totalMaskx = np.arange(-N/2, N/2)*localMaskdx
[localMaskX, localMaskY] = np.meshgrid(localMaskx, localMaskx)
[totalMaskX, totalMaskY] = np.meshgrid(totalMaskx, totalMaskx)

apertureSize = 200e-6
numOfMask = 8
# aperture = utils.spiral_blade_mask(wavelength=wavelength, N=localMaskN, f=5e-3, dx=localMaskdx, n_blades=4, blades_diameter=200e-6)
mask = np.fliplr(utils.maskGenerationMultiWavelength(numOfMask=numOfMask, wavelength=wavelengths, N=localMaskN, dx=localMaskdx, blades_diameter=apertureSize, angle=180))
mask = np.pad(mask, (N-maskN)//2)
[maskX, maskY] = np.meshgrid(localMaskx, localMaskx)
coorTran = [totalMaskx[0]*1000, totalMaskx[-1]*1000, totalMaskx[0]*1000, totalMaskx[-1]*1000]


keep = [1, 2, 3, 4]
masks = []
for k in keep:
    blockMask = np.zeros((4, 4), dtype=int)
    i = (k-1) // 4
    j = (k-1) % 4
    blockMask[i, j] = 1
    maskFilter = np.kron(blockMask, np.ones((int(localMaskN), int(localMaskN)), dtype=int))
    maskNew = mask * maskFilter
    masks.append(np.pad(maskNew, (N-maskN)//2))

#multiwavelength band mask
# regionMask = []
# for ii in range(4):
#     subMask = np.zeros([maskN, maskN])
#     subMask[:,maskN//4*ii:maskN//4*(ii+1)] = 1
#     regionMask.append(subMask)

#%%
#propagation
#params
savedir = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f20'
datapath = os.path.join(savedir, 'probeSize')
savepathcoor = os.path.join(datapath, 'coor.npy')
np.save(savepathcoor, totalMaskx)

steps = 2
fs = [20]

for f in fs:
    if f % 2 == 0:
        dss = np.arange(2, f-steps-1, steps)
    elif f % 2 != 0:
        dss = np.arange(2, f-1-steps, steps)
    
    for ds in dss:
        if f % 2 == 0:
            dms = np.arange(2, f-ds, steps)
        elif f % 2 != 0:
            dms = np.arange(2, f-1-ds, steps)
                
        for dm in dms:
            saveList = []
            filename = f'f{f}_{dm}dm{steps}step_{ds}ds.npy'
            savepath = os.path.join(datapath, filename)
            for ii in range(len(wavelengths)):
                k = 2 * np.pi / wavelengths[ii]
                illu_wavefront = np.exp(-1.0j * k * (totalMaskX**2 + totalMaskY**2) / (2 * (f/1000-dm/1000)))
                propagated_field = utils.aspw(masks[ii]*illu_wavefront, wavelengths[ii], dx=maskdx, dz=ds/1000)
                saveList.append(propagated_field)
            saveArray = np.array(saveList)
            np.save(savepath, saveArray)




#%%