#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from scipy.stats import trim_mean
from matplotlib.patches import Circle

from scipy.ndimage import center_of_mass

#%%

folderPath = r'C:\Master Thesis\data\1 optimal probe touching\data\f20_probeSizeMeasurement\data'
fileName = 'f20_6dm2step_4ds.npy'
filePath = os.path.join(folderPath, fileName)

dataSet = np.load(filePath)

#%%
N = 8000
size = 10e-3
dx = size / N

localN = 1000
localx = np.arange(-localN / 2, localN / 2) * dx
localCoor = [localx[0]*1000, localx[-1]*1000,
             localx[0]*1000, localx[-1]*1000]

fig, ax = plt.subplots(1,4, sharex=True, sharey=True)
ax = np.ravel(ax)

localN = 1000
localx = np.arange(-localN / 2, localN / 2) * dx
localCoor = [localx[0]*1000, localx[-1]*1000, localx[0]*1000, localx[-1]*1000] 

for ii in range(dataSet.shape[0]):
    data = np.abs(dataSet[ii])**2
    x0, y0 = np.unravel_index(np.argmax(data), data.shape)
    crop = data[x0-500:x0+500, y0-500:y0+500]

    # 1) 梯度幅值
    gy, gx = np.gradient(crop)
    grad = np.hypot(gx, gy)

    # 2) 在梯度图上求质心（像素坐标）
    cy_pix, cx_pix = center_of_mass(grad)

    # 3) 像素坐标 -> 物理坐标（mm）
    #    localx[0] 是左下角对应的物理坐标（mm），dx*1000 是每像素对应 mm
    x0_mm = localx[0]*1000 + cx_pix * dx * 1000
    y0_mm = localx[0]*1000 + cy_pix * dx * 1000

    # 4) 计算包围能量半径（单位：mm）
    r90 = utils.encircledEnergyRadius(crop, fraction=0.9, pixel_size=(10/8000))
    r80 = utils.encircledEnergyRadius(crop, fraction=0.8, pixel_size=(10/8000))

    ax[ii].imshow(crop, extent=localCoor)
    ax[ii].set_aspect('equal')

    # 5) 在 (x0_mm, y0_mm) 处画圆
    for r, col, lab in [(r90, 'white', '90% EE'), (r80, 'red', '80% EE')]:
        circ = Circle((x0_mm, y0_mm), r, fill=False, edgecolor=col, linewidth=2, label=lab)
        ax[ii].add_patch(circ)

    ax[ii].plot(x0_mm, y0_mm, 'yx', label='center')
    # ax[ii].legend(loc='upper right')

plt.tight_layout()
plt.show()
# fig.supxlabel("mm")
# fig.supylabel("mm", x=0.001, fontsize=12)

#%%


# %%
