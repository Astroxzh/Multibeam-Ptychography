#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os

#%%
#unit: mm
#parameter
wavelength = 584e-9
k = 2 * np.pi / wavelength
N = 4000

#%%

#create mask (spiral)
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
totalMaskx = np.arange(-N/2, N/2) * localMaskdx
[localMaskX, localMaskY] = np.meshgrid(localMaskx, localMaskx)
[totalMaskX, totalMaskY] = np.meshgrid(totalMaskx, totalMaskx)

apertureSize = 200e-6
numOfMask = 8
mask = np.fliplr(utils.maskGeneration(numOfMask=numOfMask, wavelength=wavelength, f=7.5e-3, N=localMaskN, dx=localMaskdx, blades_diameter=apertureSize, angle=180))
mask = np.pad(mask, (N-maskN)//2)
coorTran = [totalMaskx[0]*1000, totalMaskx[-1]*1000, totalMaskx[0]*1000, totalMaskx[-1]*1000]
# plt.figure(figsize=(4,4), dpi=100)
# plt.imshow(np.abs(mask), extent=coorTran, cmap='gray')
# plt.xlabel('mm')
# plt.ylabel('mm')
# plt.title('mask')
# plt.tight_layout()
# plt.show()

keep = [7, 8, 9, 10]
masks = []
for k in keep:
    blockMask = np.zeros((4, 4), dtype=int)
    i = (k-1) // 4
    j = (k-1) % 4
    blockMask[i, j] = 1

    maskFilter = np.kron(blockMask, np.ones((int(localMaskN), int(localMaskN)), dtype=int))

    maskNew = mask * maskFilter
    masks.append(np.pad(maskNew, (N-maskN)//2))

#%% 
#illuminate lens
# dz = 5
# dm = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3]
# fs = [20, 25, 30, 35, 40, 45, 50]
fs = [40]
# ii = 0

savedir = r'C:\Master Thesis\data\1 optimal probe touching\data\f40'
datapath = os.path.join(savedir, 'probeSize')
savepathcoor = os.path.join(datapath, 'coor.npy')
np.save(savepathcoor, totalMaskx)
steps = 2
for f in fs:
    if f % 2 == 0:
        dss = np.arange(2, f-steps-1, steps)
    elif f % 2 != 0:
        dss = np.arange(2, f-1-steps, steps)
    
    for ds in dss:
        if f*1000 % 2 == 0:
            dms = np.arange(2, f-ds, steps)
        elif f*1000 % 2 != 0:
            dms = np.arange(2, f-1-ds, steps)
                
        for dm in dms:
            illu_wavefront = np.exp(-1.0j * k * (totalMaskX**2 + totalMaskY**2) / (2 * (f/1000-dm/1000)))
            saveList = []
            filename = f'f{f}_{dm}dm{steps}step_{ds}ds.npy'
            savepath = os.path.join(datapath, filename)
            for mask1 in masks:
                propagated_field = utils.aspw(mask1*illu_wavefront, wavelength, dx=maskdx, dz=ds/1000)
                saveList.append(propagated_field)
            saveArray = np.array(saveList)
            np.save(savepath, saveArray)



# %%

# savedir = r'C:\Master Thesis\data\1 optimal probe touching'

# saveArray = np.stack(saveList, axis=0)
# datapath = os.path.join(savedir, 'data')
# filename = f'f{f*1000}_{dm[0]*1000}-{dm[-1]*1000}dm1step_{ds*1000}ds.npy'
# savepath = os.path.join(datapath, filename)
# savepathcoor = os.path.join(datapath, 'coor.npy')
# np.save(savepath, saveArray)
# np.save(savepathcoor, cropcoor)


#%%
# figName = f'f{f*1000}_{dm[0]*1000}-{dm[-1]*1000}dm1step_{ds*1000}ds.png'
# figpath = os.path.join(savedir, figName)
# plt.savefig(figpath, dpi=300, bbox_inches='tight', pad_inches=0.05)
# plt.show()

#%%
