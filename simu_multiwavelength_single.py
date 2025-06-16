#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os

#%%
wavelengths = [627e-9, 591e-9, 563e-9, 541e-9]
# k = 2 * np.pi / wavelength
N = 2048
size = 4e-3
dx = size / N
x = np.arange(-N /2, N/2) * dx
[X,Y] = np.meshgrid(x,x)

dfx = 1/size
fmax = 1 / (2 * dx)
fx = np.arange(-fmax, fmax, dfx)


#%%
#create mask (spiral aperture)
maskSize = 4e-3
maskN = 2048
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
# plt.figure(figsize=(4,4), dpi=100)
# plt.imshow(np.abs(mask), extent=coorTran, cmap='gray')
# plt.xlabel('mm')
# plt.ylabel('mm')
# plt.title('mask')
# plt.tight_layout()
# maskPath = r'C:\Master Thesis\data\1 optimal probe touching\mask.npy'
# np.save(maskPath, mask)
#%%
#multiwavelength band mask
regionMask = [mask[:,0:maskN//4], mask[:,maskN//4:maskN//4*2],mask[:,maskN//4*2:maskN//4*3],mask[:,maskN//4*3:maskN//4*4]]
regionMask = []
for ii in range(4):
    subMask = np.zeros([maskN, maskN])
    subMask[:,maskN//4*ii:maskN//4*(ii+1)] = 1
    subMask = np.pad(subMask, (N-maskN)//2)
    regionMask.append(subMask)

#%%
#propagation
#params
f = 40e-3
dm = 26e-3
ds = 12e-3
Itotal = np.zeros([N,N], dtype='complex128')
# Itotal = np.zeros(mask.shape, dtype='complex128')
for ii in range(len(wavelengths)):
    k = 2 * np.pi / wavelengths[ii]
    illu_wavefront = np.exp(-1.0j * k * (X**2 + Y**2) / (2 * (f-dm)))
    propagated_field = utils.aspw(regionMask[ii]*mask*illu_wavefront, wavelengths[ii], dx=maskdx, dz=ds)
    Itotal += (propagated_field)

farField = np.fft.fftshift(np.fft.fft2(Itotal))


dirPath = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f40'
fileName = [f'dm{dm*1000}ds{ds*1000}Mask.png', f'dm{dm*1000}ds{ds*1000}FF.png']

fieldPath = os.path.join(dirPath, fileName[0])
FFPath = os.path.join(dirPath, fileName[1])

# plt.figure(figsize=(4,4), dpi=100)
# plt.imshow(np.log10(np.abs(farField)**2+1), extent=[-fmax/1000, fmax/1000, -fmax/1000, fmax/1000], cmap=utils.setCustomColorMap())
# plt.xlabel('$mm^{-1}$')
# plt.ylabel('$mm^{-1}$')
# plt.title('propagated field')
# plt.tight_layout()
# plt.show()
# plt.savefig(FFPath, dpi=100, bbox_inches='tight')

plt.figure(figsize=(4,4), dpi=100)
plt.imshow(np.log10(np.abs(Itotal)**2+1), extent=coorTran, cmap=utils.setCustomColorMap())
plt.xlabel('mm')
plt.ylabel('mm')
plt.title('propagated field')
plt.tight_layout()
plt.savefig(fieldPath, dpi=100, bbox_inches='tight')




#%%
plt.show()
#%%
