#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d

#%%
#unit: mm
#parameter
wavelength = 500e-6
k = 2 * np.pi / wavelength
N = 4096

#%%
#initial wave field

#%%
#simu after lens illu
dz = 500
lensdx = wavelength * dz / 4
lensSize = lensdx * N
lensx = np.arange(- N / 2, N / 2) * lensdx
[lensX, lensY] = np.meshgrid(lensx, lensx)

lensFocuLength = 1000
# apertureLens = utils.circ(lensX, lensY, lensSize)
initialWave = np.ones([N, N])

lensTF = np.exp(-1.0j * k * (lensX**2 + lensY**2) / (2 * lensFocuLength))
exitWave = initialWave * lensTF

propagatorMask = utils.angularPropagator(dz=2, wavelength=wavelength, N=N, dx=lensdx)
illuMask = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(exitWave))*propagatorMask))

plt.imshow(np.real(illuMask))
# plt.show()

#%%
#create mask (circ aperture)
maskSize = 4
maskdx = maskSize / N

maskx = np.arange(-N / 2, N / 2) * maskdx
[maskX, maskY] = np.meshgrid(maskx, maskx)

maskNum = 4
localMaskN = N / maskNum
localMaskSize = maskSize / maskNum
localMaskdx = localMaskSize / localMaskN
localMaskx = np.arange(-localMaskN / 2, localMaskN / 2) * localMaskdx
[localMaskX, localMaskY] = np.meshgrid(localMaskx, localMaskx)
apertureSize = 200e-3
aperture = utils.circ(localMaskX, localMaskY, apertureSize)
aperture = convolve2d(aperture, utils.gaussian2D(5, 1).astype(np.float32), mode="same")

mask = np.tile(aperture, [4, 4])

#%%
#demag
#spherical wave illu
exitWaveMask = mask * illuMask
#plane wave illu
# exitWaveMask = mask * np.ones(np.shape(mask))

# demagWave = utils.demagFourier(exitWaveMask, 20)
demagWave = exitWaveMask

# plt.imshow(np.real(demagWave))
# plt.show()

#%%
#propagate to sample plane
#distance = 100mm
samplePropagator = utils.angularPropagator(0.10, wavelength=wavelength, dx=maskdx, N=N)
illuSample = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(demagWave))*samplePropagator))

plt.figure()
plt.imshow(np.exp(np.real(illuSample)+1))
plt.show()

# %%
