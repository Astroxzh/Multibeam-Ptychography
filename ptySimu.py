#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from PIL import Image

#%%
wavelengths = [685e-9, 670e-9, 655e-9, 640e-9, 
               625e-9, 610e-9, 595e-9, 580e-9, 
               565e-9, 550e-9, 535e-9, 520e-9,
               505e-9, 490e-9, 475e-9, 460e-9]
# k = 2 * np.pi / wavelength
N = 2048

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
plt.figure(figsize=(4,4), dpi=100)
plt.imshow(np.abs(mask), extent=coorTran, cmap='gray')
plt.xlabel('mm')
plt.ylabel('mm')
plt.title('mask')
plt.tight_layout()
plt.show()

#%%
#multiwavelength band mask
regionMask = []
for jj in range(4):
    for ii in range(4):
        subMask = np.zeros([maskN, maskN])
        row_start = (maskN // 4) * ii
        row_end   = (maskN // 4) * (ii + 1)
        col_start = (maskN // 4) * jj
        col_end   = (maskN // 4) * (jj + 1)
        subMask[row_start:row_end, col_start:col_end] = 1
        regionMask.append(subMask)

#%%
#propagation
#params
f = 40e-3
dm = 10e-3
ds = 20e-3

probe = np.zeros(mask.shape, dtype='complex128')
for ii in range(len(wavelengths)):
    k = 2 * np.pi / wavelengths[ii]
    illu_wavefront = np.exp(-1.0j * k * (totalMaskX**2 + totalMaskY**2) / (2 * (f-dm)))
    propagated_field = utils.aspw(regionMask[ii]*mask*illu_wavefront, wavelengths[ii], dx=maskdx, dz=ds)
    probe += (propagated_field)

# plt.figure(figsize=(4,4), dpi=100)
# plt.imshow(np.log10(abs(Itotal)**2+1), extent=coorTran, cmap=utils.setCustomColorMap())
# plt.xlabel('mm')
# plt.ylabel('mm')
# plt.title('propagated field')
# plt.tight_layout()
# plt.show()

#%%
#obj
image = Image.open(r'Canon-100Block.png')
obj = np.array(image)
objShape = obj.shape
obj = obj[objShape[0]//2-N//2:objShape[0]//2+N//2, objShape[1]//2-N//2:objShape[1]//2+N//2]
phase_obj = np.exp(1j * np.where(obj==255, 0, np.pi))
#%%
#illumination
objWave = probe * phase_obj
#%%
#build dataset
#%%