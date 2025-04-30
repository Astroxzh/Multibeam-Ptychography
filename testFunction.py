#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d

#%%
wavelength=584e-9
k = 2 * np.pi / wavelength
N = 4096

maskSize = 4e-3
maskdx = maskSize / N
maskx = np.arange(-N/2, N/2) * maskdx
[maskX, maskY] = np.meshgrid(maskx, maskx)
localMaskSize = maskSize / 4
localMaskN = N / 4
localMaskdx = localMaskSize / localMaskN
localMaskx = np.arange(-localMaskN/2, localMaskN/2) * localMaskdx
[localMaskX, localMaskY] = np.meshgrid(localMaskx, localMaskx)
# sprial = utils.spiral_blade_mask(wavelength=627e-9, f=7.25e-3, N=N/4, dx=localMaskdx, n_blades=1, blades_diameter=200e-6, angle=0)
# flipsprial = np.flipud(sprial)

mask = np.fliplr(utils.maskGeneration(numOfMask=8, wavelength=wavelength, f=7.5e-3, N=localMaskN, dx=localMaskdx, blades_diameter=200e-6, angle=180))

plt.imshow(mask,extent=[maskx[0]*1000, maskx[-1]*1000, maskx[0]*1000, maskx[-1]*1000])
# plt.figure()
# plt.imshow(flipsprial,extent=[localMaskx[0]*1000, localMaskx[-1]*1000, localMaskx[0]*1000, localMaskx[-1]*1000])
plt.show()
# %%
