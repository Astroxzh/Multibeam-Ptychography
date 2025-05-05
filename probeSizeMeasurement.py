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


#%%
#unit: m
#parameter
wavelength = 584e-9 #627e-9  
k = 2 * np.pi / wavelength
N = 8000

#%%
#light source
lightsourceSize = 10e-3
lightsourcedx = lightsourceSize / N
lightsourcex = np.arange(- N / 2 , N / 2) * lightsourcedx
[lightsourceX, lightsourceY] = np.meshgrid(lightsourcex, lightsourcex)
translationCoor = [lightsourceX ]
coorTran = [lightsourcex[0]*1000, lightsourcex[-1]*1000, lightsourcex[0]*1000, lightsourcex[-1]*1000]

initialField = np.ones([N, N])

#%% 
#illuminate lens

dz = 10e-3  # distance to mask

# lens and aperture
lensSize = 9e-3
lensFocuLength = 20e-3
apertureLens = utils.circ(lightsourceX, lightsourceY, lensSize)
initialWave = initialField

lensTF = np.exp(-1.0j * k * (lightsourceX**2 + lightsourceY**2) / (2 * lensFocuLength))

exitWaveWrap = initialWave * lensTF #unwrap is not helping
exitWavePhase = np.unwrap(np.unwrap(np.angle(exitWaveWrap), axis=0), axis=1)

exitWave = np.abs(exitWaveWrap) * np.exp(1j * exitWavePhase)
exitWave *= apertureLens


#%%

#propagate to mask
propagatorMask = utils.angularPropagator(dz=dz, wavelength=wavelength, N=N, dx=lightsourcedx)
illuMask = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(exitWave))*propagatorMask))


coorTran = [lightsourcex[0]*1000, lightsourcex[-1]*1000, lightsourcex[0]*1000, lightsourcex[-1]*1000]


#%%
#create mask (circ aperture)
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
# mask filter, keep 2 8 9 15 mask,
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
# maskFilter = np.kron(blockMask, np.ones((int(localMaskN), int(localMaskN)), dtype=int))
# mask *= maskFilter

#%%
# mask = np.tile(aperture, [4, 4])
[maskX, maskY] = np.meshgrid(lightsourcex, lightsourcex)

exitWaveMask = illuMask * mask
plt.figure()
plt.imshow(np.abs(exitWaveMask)**2, extent=coorTran)
#%%

#4f system mag effect
# demagWave = utils.demagFourier(exitWaveMask, 4)
demagWave = exitWaveMask

#%%
#propagate to sample plane
ds = 6e-3  # distance to sample after mask
propagatorSample = utils.angularPropagator(dz=ds, wavelength=wavelength, N=N, dx=lightsourcedx)
illuSample = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(demagWave))*propagatorSample))
# %%





plt.show()
# %%
