#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d

#%%
wavelength = 500e-6
k = 2 * np.pi / wavelength
N = 4096

#%%
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