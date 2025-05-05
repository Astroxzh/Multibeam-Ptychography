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
#square slits
# localSuqare = np.zeros([N, N // 4], dtype=complex)
# localSuqare[100 : -100, N // 4 // 4 : N // 4 - N // 4 // 4] = 1
# lightsource = np.tile(localSuqare, [1,4])
# lightsource = convolve2d(lightsource, utils.gaussian2D(5, 1).astype(np.float32), mode="same")

#light source
lightsourceSize = 10e-3
lightsourcedx = lightsourceSize / N
lightsourcex = np.arange(- N / 2 , N / 2) * lightsourcedx
[lightsourceX, lightsourceY] = np.meshgrid(lightsourcex, lightsourcex)
translationCoor = [lightsourceX ]
coorTran = [lightsourcex[0]*1000, lightsourcex[-1]*1000, lightsourcex[0]*1000, lightsourcex[-1]*1000]
# propagator = utils.angularPropagator(30,wavelength=wavelength, dx=lightsourcedx, N=N)
# initialField = utils.angularPro(lightsource, propagator)
# initialField = lightsource
initialField = np.ones([N, N])

#%% 
#illuminate lens

dz = 10e-3  # distance to mask

# lensdx = wavelength * dz / 4
# lensSize = lensdx * N
# lensx = np.arange(- N / 2, N / 2) * lensdx
# [lensX, lensY] = np.meshgrid(lensx, lensx)

# lens and aperture
lensSize = 9e-3
lensFocuLength = 20e-3
apertureLens = utils.circ(lightsourceX, lightsourceY, lensSize)
initialWave = initialField

# lensTF = np.exp(-1.0j * k * (lensX**2 + lensY**2) / (2 * lensFocuLength))
lensTF = np.exp(-1.0j * k * (lightsourceX**2 + lightsourceY**2) / (2 * lensFocuLength))
# plt.figure()
# plt.imshow(np.angle(lensTF))
exitWaveWrap = initialWave * lensTF #unwrap is not helping
exitWavePhase = np.unwrap(np.unwrap(np.angle(exitWaveWrap), axis=0), axis=1)
# plt.figure()
# plt.imshow(exitWavePhase, extent=coorTran)
exitWave = np.abs(exitWaveWrap) * np.exp(1j * exitWavePhase)
exitWave *= apertureLens
# plt.figure()
# plt.imshow(np.angle(exitWave), extent=coorTran)

#%%

#propagate to mask
propagatorMask = utils.angularPropagator(dz=dz, wavelength=wavelength, N=N, dx=lightsourcedx)
illuMask = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(exitWave))*propagatorMask))
# illuMask = utils.fresnelPro(exitWave, wavelength=wavelength, dz=dz, dx=lightsourcedx)
# illuMask = exitWave 

coorTran = [lightsourcex[0]*1000, lightsourcex[-1]*1000, lightsourcex[0]*1000, lightsourcex[-1]*1000]
# plt.figure()
# plt.imshow(np.abs(illuMask)**2, extent=coorTran)

#%%
#create mask (spiral aperture)
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
# aperture = utils.circ(localMaskX, localMaskY, apertureSize)
# aperture = convolve2d(aperture, utils.gaussian2D(5, 1).astype(np.float32), mode="same")
# aperture = utils.spiral_blade_mask(wavelength=wavelength, N=localMaskN, f=5e-3, dx=localMaskdx, n_blades=4, blades_diameter=200e-6)
mask = np.fliplr(utils.maskGeneration(numOfMask=numOfMask, wavelength=wavelength, f=7.5e-3, N=localMaskN, dx=localMaskdx, blades_diameter=apertureSize, angle=180))
mask = np.pad(mask, (N-maskN)//2)
# mask = np.tile(aperture, [4, 4])
[maskX, maskY] = np.meshgrid(lightsourcex, lightsourcex)
# plt.figure()
# plt.imshow(np.abs(mask), extent=coorTran)
# plt.xlabel('mm')
# plt.ylabel('mm')
# plt.title('mask')


exitWaveMask = illuMask * mask
plt.figure()
plt.imshow(np.abs(exitWaveMask)**2, extent=coorTran)
#%%

#4f system mag effect
# demagWave = utils.demagFourier(exitWaveMask, 4)
demagWave = exitWaveMask
# plt.figure()
# plt.imshow(np.abs(demagWave)**2, extent=coorTran)

#%%
#propagate to sample plane
ds = 6e-3  # distance to sample after mask
propagatorSample = utils.angularPropagator(dz=ds, wavelength=wavelength, N=N, dx=lightsourcedx)
illuSample = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(demagWave))*propagatorSample))
# plt.figure()
# plt.imshow(np.log(np.abs(illuSample)**2+1), extent=coorTran)

# profile = (np.abs(illuSample)**2)[1570:2100, 2475]
# plt.figure()
# plt.plot(profile)
# %%

cmap = utils.setCustomColorMap()

fig, ax = plt.subplots(2,2, figsize=(10,10))
ax = ax.ravel()
fig.suptitle(f'probe simu with {lensFocuLength*1000}mm foculength lens')
ax[0].imshow(np.abs(mask), extent=coorTran)
ax[0].set_title('mask', fontsize=8)
ax[1].imshow(np.abs(illuMask)**2, extent=coorTran)
ax[1].set_title(f'illumination field on mask \n {dz*1000}mm after source', fontsize=8)
ax[2].imshow(np.abs(exitWaveMask)**2, extent=coorTran)
ax[2].set_title('exit field after mask', fontsize=8)
ax[3].imshow(np.log(np.abs(illuSample)**2+1), extent=coorTran)
# ax[3].imshow(utils.complex2rgb(illuSample), extent=coorTran, cmap = cmap)
ax[3].set_title(f'probe on sample plane \n {ds*1000}mm propagation', fontsize=8)
# ax[3].plot([0.7, 0.7], [0.0, 1.0], color='red', linewidth=0.5)
fig.text(0.5, 0.04, 'coordinate/mm', ha='center', fontsize=12)
fig.text(0.1, 0.5, 'coordinate/mm', va='center', rotation='vertical', fontsize=12)




plt.show()
# %%
