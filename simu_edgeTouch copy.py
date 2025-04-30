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
localSuqare = np.zeros([N, N // 4], dtype=complex)
localSuqare[100 : -100, N // 4 // 4 : N // 4 - N // 4 // 4] = 1
lightsource = np.tile(localSuqare, [1,4])
lightsource = convolve2d(lightsource, utils.gaussian2D(5, 1).astype(np.float32), mode="same")

lightsourceSize = 8e-3
lightsourcedx = lightsourceSize / N
lightsourcex = np.arange(- N / 2 , N / 2) * lightsourcedx
[lightsourceX, lightsourceY] = np.meshgrid(lightsourcex, lightsourcex)
coorTran = [lightsourcex[0]*1000, lightsourcex[-1]*1000, lightsourcex[0]*1000, lightsourcex[-1]*1000]

# propagator = utils.angularPropagator(30,wavelength=wavelength, dx=lightsourcedx, N=N)
# initialField = utils.angularPro(lightsource, propagator)
# initialField = lightsource
initialField = np.ones(np.shape(lightsource))

# lensFocuLengths = [20, 25, 30, 35, 40, 50]
lensFocuLength = 20e-3
# apertureLens = utils.circ(lensX, lensY, lensSize)
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
apertureSize = 245e-6
numOfMask = 8
mask = np.fliplr(utils.maskGeneration(numOfMask=numOfMask, wavelength=wavelength, f=11.5e-3, N=localMaskN, dx=localMaskdx, blades_diameter=apertureSize, angle=180))
mask = np.pad(mask, (N-maskN)//2)
# mask = np.tile(aperture, [4, 4])
[maskX, maskY] = np.meshgrid(lightsourcex, lightsourcex)
# plt.figure()
# plt.imshow(mask, extent=coorTran)

#%% 
#illuminate lens
# dz = 5
# dm = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3]
dm = [2e-3, 4e-3, 6e-3, 8e-3, 10e-3, 12e-3, 14e-3, 16e-3]
ds = 4e-3
# lensdx = wavelength * dz / 4
# lensSize = lensdx * N
# lensx = np.arange(- N / 2, N / 2) * lensdx
# [lensX, lensY] = np.meshgrid(lensx, lensx)
fig, ax = plt.subplots(1,len(dm), figsize=(12,12))
ax = ax.ravel()
fig.suptitle(f'probe on sample plane \n f={lensFocuLength*1000}mm, ds={ds*1000}mm', y=0.60, ha='center') #记得说明，ds传到sample，dm到mask距离

ii = 0
cropHalfSize = 700
cropx = lightsourcex[N//2-cropHalfSize:N//2+cropHalfSize]
cropcoor = [cropx[0]*1000, cropx[-1]*1000, cropx[0]*1000, cropx[-1]*1000]
saveList = []
for dz in dm:
    lensTF = np.exp(-1.0j * k * (lightsourceX**2 + lightsourceY**2) / (2 * lensFocuLength))
    exitWave = initialWave * lensTF
    
    propagatorMask = utils.angularPropagator(dz=dz, wavelength=wavelength, N=N, dx=lightsourcedx)
    illuMask = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(exitWave))*propagatorMask))
    # illuMask = utils.fresnelPro(exitWave, wavelength=wavelength, dz=dz, dx=lightsourcedx)
    # ax[0,ii].imshow(np.abs(illuMask)**2, extent=coorTran)

    # plt.figure()
    # plt.imshow(np.abs(mask), extent=[lightsourcex[0], lightsourcex[-1], lightsourcex[0], lightsourcex[-1]])

    exitWaveMask = illuMask * mask
    # plt.figure()
    # plt.imshow(np.abs(exitWaveMask)**2, extent=[lightsourcex[0], lightsourcex[-1], lightsourcex[0], lightsourcex[-1]])
    # demagWave = utils.demagFourier(exitWaveMask, 4)
    demagWave = exitWaveMask
    # plt.imshow(np.abs(demagWave)**2, extent=coorTran)


    propagatorSample = utils.angularPropagator(dz=ds, wavelength=wavelength, N=N, dx=lightsourcedx)
    illuSample = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(demagWave))*propagatorSample))
    cropRegion = illuSample[N//2-cropHalfSize:N//2+cropHalfSize,N//2-cropHalfSize:N//2+cropHalfSize]
    saveList.append(cropRegion)
    if ii == 0:
        ax[ii].imshow(np.log(np.abs(cropRegion)+1), extent=cropcoor)
        ax[ii].set_title(f'dm={dz*1000}mm', fontsize=8)
    else:
        ax[ii].imshow(np.log(np.abs(cropRegion)+1), extent=cropcoor)
        ax[ii].set_title(f'dm={dz*1000}mm', fontsize=8)
        ax[ii].axis('off')
    ii += 1
# %%



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
plt.show()

#%%
