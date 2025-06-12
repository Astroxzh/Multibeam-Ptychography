#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from scipy.stats import trim_mean

#%%
def mask_triangle(data: np.ndarray) -> np.ndarray:
    """
    对形状 (17,16) 的 data 应用“倒三角”掩码：
      - 行坐标 36、34（对应 idx=16,15）全置 0
      - 行坐标 32(idx=14) 保留前 1 列，其它置 0
        行坐标 30(idx=13) 保留前 2 列，其它置 0
        …
        行坐标 20(idx=8)  保留前 7 列，其它置 0
      - 行坐标 <20(idx=0..7) 完全保留
    """
    # 生成行坐标数组 [4,6,8,…,36]
    row_coords = np.arange(4, 38, 2)
    assert data.shape[0] == len(row_coords), "行数不匹配，需 17 行"
    
    out = data.copy()
    for i, coord in enumerate(row_coords):
        if coord >= 34:
            # 舍弃整行
            out[i, :] = 0
        elif coord >= 20:
            # 计算该行保留前几列
            # coord=32 -> keep=1; 30->2; …; 20->7
            keep = (34 - coord) // 2
            out[i, keep:] = 0
        # else coord<20 -> 全部保留，不做改动

    return out

#%%
pathProbeSize = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f40\probeSize'
pathProbeDistance = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f40\probeDistance\ResultDistanceCorrect.npy'
sizeNames = ['ResultSize80%_probe627.npy', 'ResultSize80%_probe591.npy', 'ResultSize80%_probe563.npy', 'ResultSize80%_probe541.npy']
resultDistance = np.load(pathProbeDistance)
# resultProbeSize = np.load(pathProbeSize)
size = np.zeros([4, np.shape(np.ravel(resultDistance))[0]])

ii = 0
for sizeName in sizeNames:
    pathSize = os.path.join(pathProbeSize, sizeName)
    data = np.load(pathSize)
    data = mask_triangle(data)
    mask = ~(data == 0).all(axis=1)
    data = data[mask]
    data = np.ravel(data)
    size[ii,:] = data
    ii += 1

#%%
#coordinates
# dsVals = np.arange(4, 18, 2)
# dmVals = np.arange(2, 14, 2)

#for 40mm
dsVals = np.arange(4, 34, 2) 
dmVals = np.arange(2, 34, 2)

DM, DS = np.meshgrid(dmVals, dsVals)

resultDistanceRe = np.ravel(resultDistance)
# resultProbeSizeRe = np.ravel(resultProbeSize)

#%%
overlap = np.zeros([3, np.shape(size)[1]])

for ii in range(1, np.shape(size)[0]):
    for jj in range(np.shape(size)[1]):
        overlap[ii-1,jj] = utils.area_overlap_diffSize(size[ii-1,jj], size[ii,jj], resultDistanceRe[jj])

overlap627591 = np.reshape(overlap[0,:], np.shape(resultDistance))
overlap591563 = np.reshape(overlap[1,:], np.shape(resultDistance))
overlap563541 = np.reshape(overlap[2,:], np.shape(resultDistance))

meanOverlap = (overlap627591 + overlap591563 + overlap563541) / 3

#%%
# heat map
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(DM, DS, meanOverlap*100, shading='auto')
plt.colorbar(pcm, label='%')
plt.xlabel('dm/mm')
plt.ylabel('ds/mm')
plt.title('overlap vs ds and dm \n @80% encircled energy')
plt.tight_layout()

plt.show()
#%%