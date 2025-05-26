#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from scipy.stats import trim_mean
from scipy.optimize import curve_fit

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
#read data
folderPath = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f40\probeDistance'
fileName = ['ResultDistance_peakDis90.npy', 'ResultDistance_peakDis100.npy', 'ResultDistance_peakDis130.npy']

filePath90 = os.path.join(folderPath, fileName[0])
filePath100 = os.path.join(folderPath, fileName[1])
filePath130 = os.path.join(folderPath, fileName[2])

pd90 = np.load(filePath90)
pd100 = np.load(filePath100)
pd130 = np.load(filePath130)


#%%
newRe913 = np.zeros_like(pd90)
newRe1013 = np.zeros_like(pd90)

newRe913[3:17,:] = pd90[3:17,:]
newRe913[0:3,:] = pd130[0:3,:]

newRe1013[3:17,:] = pd100[3:17, :]
newRe1013[0:3,:] = pd130[0:3,:]

# %%
#for 40mm
dsVals = np.arange(4, 34, 2) 
dmVals = np.arange(2, 34, 2)

DM, DS = np.meshgrid(dmVals, dsVals)

#没有太大区别，选100吧
resultDistanceFiltered = newRe1013
resultDistanceFiltered = mask_triangle(resultDistanceFiltered)
mask = ~(resultDistanceFiltered == 0).all(axis=1)
resultDistanceFiltered = resultDistanceFiltered[mask]
# heat map
plt.figure(figsize=(6,5))
pcm = plt.pcolormesh(DM, DS, resultDistanceFiltered, shading='auto')
plt.colorbar(pcm, label='mm')
plt.xlabel('dm/mm')
plt.ylabel('ds/mm')
plt.title('probe distance vs ds and dm')
plt.tight_layout()

#%%
#analyze and fit 4ds with changing dm and 2dm with changing ds

ds4mm = resultDistanceFiltered[0, :]
dm2mm = resultDistanceFiltered[:, 0]

# coeds = np.polyfit(dmVals, ds4mm, 2)
# fitds = np.poly1d(coeds)
# dsFitx = np.linspace(dmVals[0], dmVals[-1], 100)
# dsFity = fitds(dsFitx)
# dsPred = fitds(dmVals)
# residualsDsFit = ds4mm - dsPred
def inv_model(x, a, b):
    return (b - a / (40 - x))
popt, pcov = curve_fit(inv_model, dmVals, ds4mm, p0=(2.0, 2.0))
a, b = popt
fitds = inv_model(dmVals, a, b)
residualsDsFit = ds4mm - fitds

coedm = np.polyfit(dsVals, dm2mm, 1)
fitdm = np.poly1d(coedm)
dmFitx = np.linspace(dsVals[0], dsVals[-1], 100)
dmFity = fitdm(dmFitx)
dmPred = fitdm(dsVals)
residualsDmFit = dm2mm - dmPred

plt.figure()
fig, (axData, axResid) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1},
    figsize=(6, 6)
)
axData.scatter(dmVals, ds4mm, label='data')
# axData.plot(dsFitx, dsFity, label=f'{fitds[2]:.7f}$x^2${fitds[1]:.7f}$x$+{fitds[0]:.7f}')
axData.plot(dmVals, inv_model(dmVals, a, b), label=f'{b:.3f} - {a:.3f}/(40 - x)')
axData.set_title('probe distance vs dm, @ds=4mm')
axResid.set_xlabel('dm/mm')
axData.set_ylabel('probe distance/mm')
axData.legend()

axResid.scatter(dmVals, residualsDsFit)
axResid.set_ylabel('residual')
axResid.axhline(0, c='gray')

plt.figure()
fig, (axData, axResid) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1},
    figsize=(6, 6)
)
axData.scatter(dsVals, dm2mm, label='data')
axData.plot(dmFitx, dmFity, label=f'{fitdm[1]:.5f}$x$+{fitdm[0]:.5f}')
axData.set_title('probe distance vs ds, @dm=2mm')
axResid.set_xlabel('ds/mm')
axData.set_ylabel('probe distance/mm')
axData.legend()

axResid.scatter(dsVals, residualsDmFit)
axResid.set_ylabel('residual')
axResid.axhline(0, c='gray')

#%%

# plt.axis('off')
plt.show()

#%%