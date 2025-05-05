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
N = 8000

#%%
#square slits
# localSuqare = np.zeros([N, N // 4], dtype=complex)
# localSuqare[100 : -100, N // 4 // 4 : N // 4 - N // 4 // 4] = 1
# lightsource = np.tile(localSuqare, [1,4])
# lightsource = convolve2d(lightsource, utils.gaussian2D(5, 1).astype(np.float32), mode="same")

lightsourceSize = 10e-3
lightsourcedx = lightsourceSize / N
lightsourcex = np.arange(- N / 2 , N / 2) * lightsourcedx
[lightsourceX, lightsourceY] = np.meshgrid(lightsourcex, lightsourcex)
coorTran = [lightsourcex[0]*1000, lightsourcex[-1]*1000, lightsourcex[0]*1000, lightsourcex[-1]*1000]

# propagator = utils.angularPropagator(30,wavelength=wavelength, dx=lightsourcedx, N=N)
# initialField = utils.angularPro(lightsource, propagator)
# initialField = lightsource
initialField = np.ones([N, N])

# apertureLens = utils.circ(lensX, lensY, lensSize)
lensSize = 9e-3
apertureLens = utils.circ(lightsourceX, lightsourceY, lensSize)
initialWave = initialField
# lensTF = np.exp(-1.0j * k * (lensX**2 + lensY**2) / (2 * lensFocuLength))


#create mask (spiral)
maskSize = 4e-3
maskN = int(N / (lightsourceSize / maskSize))
maskdx = maskSize / maskN

maskx = np.arange(-maskN / 2, maskN / 2) * maskdx
[maskX, maskY] = np.meshgrid(maskx, maskx)
maskNum = 4
localMaskN = maskN / maskNum
localMaskSize = maskSize / maskNum
localMaskdx = localMaskSize / localMaskN
localMaskx = np.arange(-localMaskN / 2, localMaskN / 2) * localMaskdx
[localMaskX, localMaskY] = np.meshgrid(localMaskx, localMaskx)
apertureSize = 200e-6
numOfMask = 8
mask = np.fliplr(utils.maskGeneration(numOfMask=numOfMask, wavelength=wavelength, f=7.5e-3, N=localMaskN, dx=localMaskdx, blades_diameter=apertureSize, angle=180))

keep = [7, 8, 9, 10]
masks = []
# blockMask = np.zeros((4, 4), dtype=int)
# for k in keep:
#     i = (k-1) // 4
#     j = (k-1) % 4
#     blockMask[i, j] = 1
for k in keep:
    blockMask = np.zeros((4, 4), dtype=int)
    i = (k-1) // 4
    j = (k-1) % 4
    blockMask[i, j] = 1

    maskFilter = np.kron(blockMask, np.ones((int(localMaskN), int(localMaskN)), dtype=int))

    maskNew = mask * maskFilter
    masks.append(np.pad(maskNew, (N-maskN)//2))

# mask = np.tile(aperture, [4, 4])
[maskX, maskY] = np.meshgrid(lightsourcex, lightsourcex)
# plt.figure()
# plt.imshow(mask, extent=coorTran)

#%% 
#illuminate lens
# dz = 5
# dm = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3]
# lensFocuLengths = [20, 25, 30, 35, 40, 45, 50]
lensFocuLengths = [20]



# lensdx = wavelength * dz / 4
# lensSize = lensdx * N
# lensx = np.arange(- N / 2, N / 2) * lensdx
# [lensX, lensY] = np.meshgrid(lensx, lensx)
# fig, ax = plt.subplots(1,len(dm), figsize=(12,12))
# ax = ax.ravel()
# fig.suptitle(f'probe on sample plane \n f={lensFocuLength*1000}mm, ds={ds*1000}mm', y=0.60, ha='center') #记得说明，ds传到sample，dm到mask距离

# ii = 0
cropHalfSize = 2000
cropx = lightsourcex[N//2-cropHalfSize:N//2+cropHalfSize]
cropcoor = [cropx[0]*1000, cropx[-1]*1000, cropx[0]*1000, cropx[-1]*1000]

savedir = r'C:\Master Thesis\data\1 optimal probe touching\data\f20_probeSizeMeasurement'
datapath = os.path.join(savedir, 'data')
savepathcoor = os.path.join(datapath, 'coor.npy')
np.save(savepathcoor, lightsourcex)
steps = 2
for lensFocuLength in lensFocuLengths:
    if lensFocuLength % 2 == 0:
        ds = np.arange(4,lensFocuLength-steps-1, steps)
    elif lensFocuLength % 2 != 0:
        ds = np.arange(4, lensFocuLength-1-steps, steps)

    lensTF = np.exp(-1.0j * k * (lightsourceX**2 + lightsourceY**2) / (2 * lensFocuLength/1000))
    exitWave = initialWave * lensTF
    exitWave *= apertureLens
    
    for dzs in ds:
        if lensFocuLength*1000 % 2 == 0:
            dm = np.arange(2,lensFocuLength-dzs, steps)
        elif lensFocuLength*1000 % 2 != 0:
            dm = np.arange(2, lensFocuLength-1-dzs, steps)
        
        
        propagatorSample = utils.angularPropagator(dz=dzs/1000, wavelength=wavelength, N=N, dx=lightsourcedx)
        
        for dzm in dm:
            propagatorMask = utils.angularPropagator(dz=dzm/1000, wavelength=wavelength, N=N, dx=lightsourcedx)
            illuMask = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(exitWave))*propagatorMask))
            saveList = []
            filename = f'f{lensFocuLength}_{dzm}dm{steps}step_{dzs}ds.npy'
            savepath = os.path.join(datapath, filename)
            for mask1 in masks:
                exitWaveMask = illuMask * mask1
                demagWave = exitWaveMask
                illuSample = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(demagWave))*propagatorSample))
                saveList.append(illuSample)
            saveArray = np.array(saveList)
            np.save(savepath, saveArray)



# %%

# savedir = r'C:\Master Thesis\data\1 optimal probe touching'

# saveArray = np.stack(saveList, axis=0)
# datapath = os.path.join(savedir, 'data')
# filename = f'f{lensFocuLength*1000}_{dm[0]*1000}-{dm[-1]*1000}dm1step_{ds*1000}ds.npy'
# savepath = os.path.join(datapath, filename)
# savepathcoor = os.path.join(datapath, 'coor.npy')
# np.save(savepath, saveArray)
# np.save(savepathcoor, cropcoor)


#%%
# figName = f'f{lensFocuLength*1000}_{dm[0]*1000}-{dm[-1]*1000}dm1step_{ds*1000}ds.png'
# figpath = os.path.join(savedir, figName)
# plt.savefig(figpath, dpi=300, bbox_inches='tight', pad_inches=0.05)
# plt.show()

#%%
