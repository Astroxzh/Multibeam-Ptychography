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
#simu after lens illu
dz = 100
lensdx = wavelength * dz / 4
lensSize = lensdx * N
lensx = np.arange(- N / 2, N / 2) * lensdx
[lensX, lensY] = np.meshgrid(lensx, lensx)

lensFocuLength = 1000
# apertureLens = utils.circ(lensX, lensY, lensSize)
initialWave = np.ones([N, N], type=complex)

lensTF = np.exp(-1.0j * k * (lensX**2 + lensY**2) / (2 * lensFocuLength))
exitWave = initialWave * lensTF

propagatorMask = utils.angularPropagator(dz=2, wavelength=wavelength, N=N, dx=lensdx)
illuMask = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(exitWave))*propagatorMask))

plt.imshow((np.real(illuMask)))
plt.show()

# %%
