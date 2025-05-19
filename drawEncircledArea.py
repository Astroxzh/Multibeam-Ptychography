#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from scipy.stats import trim_mean
from matplotlib.patches import Circle

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

for ii in range(np.shape(dataSet)[0]):
    data = np.abs(dataSet[ii, :, :])**2
    xMax, yMax = np.unravel_index(np.argmax(data), data.shape)
    cropData = data[xMax-500:xMax+500, yMax-500:yMax+500]
    radius90 = utils.encircledEnergyRadius(cropData, fraction=0.9, pixel_size=(10/8000))
    radius80 = utils.encircledEnergyRadius(cropData, fraction=0.8, pixel_size=(10/8000))
    ax[ii].imshow(cropData, extent=localCoor)
    circ90 = Circle(
        (0, 0),                 # 圆心
        radius90,               # 半径（mm）
        fill=False,             # 不填充
        edgecolor='white',      # 白色边框
        linewidth=2,            # 线宽
        label='90% EE'          # 图例标签（可选）
    )
    circ80 = Circle(
        (0, 0),
        radius80,
        fill=False,
        edgecolor='red',
        linewidth=2,
        label='80% EE'
    )
    ax[ii].add_patch(circ90)
    ax[ii].add_patch(circ80)

    # 可选：在圆心画个标记
    ax[ii].plot(0, 0, marker='x', color='yellow')

    ax[ii].set_aspect('equal')   # 保证 x/y 比例一致，这样圆才不变形
    ax[ii].legend(loc='upper right')
    
# fig.supxlabel("mm")
# fig.supylabel("mm", x=0.001, fontsize=12)

#%%
plt.tight_layout()
plt.show()
#%%

