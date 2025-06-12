#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.signal import convolve2d
import os
from scipy.stats import trim_mean
from scipy.optimize import curve_fit

#%%
folderPath = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f40\probeSize'
# fileNames = ['ResultSize80%_probe627.npy', 'ResultSize80%_probe591.npy', 'ResultSize80%_probe563.npy', 'ResultSize80%_probe541.npy']
fileNames = ['ResultSize80%_probe591.npy']

# for 20mm
# dsVals = np.arange(4, 18, 2) 
# dmVals = np.arange(2, 14, 2)

#for 40mm
dsVals = np.arange(4, 38, 2) 
dmVals = np.arange(2, 34, 2)

DM, DS = np.meshgrid(dmVals, dsVals)

for fileName in fileNames:
    filePath = os.path.join(folderPath, fileName)
    data = np.load(filePath)

    # heat map
    plt.figure(figsize=(6,5))
    pcm = plt.pcolormesh(DM, DS, data, shading='auto')
    plt.colorbar(pcm, label='mm')
    plt.xlabel('dm/mm')
    plt.ylabel('ds/mm')
    plt.title('probe size vs ds and dm')
    plt.tight_layout()

    #analyze and fit 4ds with changing dm and 2dm with changing ds

    ds4mm = data[0, :]
    dm2mm = data[:, 0]

    def inv_model(x, a, b):
        return (b - a / (40 - x))
    popt, pcov = curve_fit(inv_model, dmVals, ds4mm, p0=(2.0, 2.0))
    a, b = popt
    fitds = inv_model(dmVals, a, b)
    residualsDsFit = ds4mm - fitds
    # coeds = np.polyfit(dmVals, ds4mm, 2)
    # fitds = np.poly1d(coeds)
    # dsFitx = np.linspace(dmVals[0], dmVals[-1], 100)
    # dsFity = fitds(dsFitx)
    # dsPred = fitds(dmVals)
    # residualsDsFit = ds4mm - dsPred

    sigma = np.std(residualsDsFit)
    mask = np.abs(residualsDsFit) <= 3.0 * sigma
    x_filt, y_filt = dmVals[mask], ds4mm[mask]

    # 第二次在剔除后的点上再拟合
    # coeffs2 = np.polyfit(x_filt, y_filt, deg=2)
    # fit2 = np.poly1d(coeffs2)
    # dsPred2 = fit2(dmVals)
    # dsFity2 = fit2(dsFitx)
    popt1, pcov1 = curve_fit(inv_model, x_filt, y_filt, p0=(2.0, 2.0))
    a1, b1 = popt1
    fitds2 = inv_model(dmVals, a1, b1)
    residualsDsFit = ds4mm - fitds2

    coedm = np.polyfit(dsVals, dm2mm, 1)
    fitdm = np.poly1d(coedm)
    dmFitx = np.linspace(dsVals[0], dsVals[-1], 100)
    dmFity = fitdm(dmFitx)
    dmPred = fitdm(dsVals)
    residualsDmFit = dm2mm - dmPred

    fig, (axData, axResid) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1},
        figsize=(6, 6)
    )
    axData.scatter(dmVals, ds4mm, label='data')
    # axData.plot(dsFitx, dsFity2, label=f'{fitds[2]:.7f}$x^2${fitds[1]:.7f}$x$+{fitds[0]:.7f}')
    axData.plot(dmVals, inv_model(dmVals, a1, b1), label=f'{b:.3f} - {a:.3f}/(40 - x)')
    axData.set_title('probe size vs dm, @ds=4mm')
    axResid.set_xlabel('dm/mm')
    axData.set_ylabel('probe size/mm')
    axData.legend()

    axResid.scatter(dmVals, residualsDsFit)
    axResid.set_ylabel('residual')
    axResid.axhline(0, c='gray')

    fig, (axData, axResid) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1},
        figsize=(6, 6)
    )
    axData.scatter(dsVals, dm2mm, label='data')
    axData.plot(dmFitx, dmFity, label=f'{fitdm[1]:.5f}$x$+{fitdm[0]:.5f}')
    axData.set_title('probe size vs ds, @dm=2mm')
    axResid.set_xlabel('ds/mm')
    axData.set_ylabel('probe size/mm')
    axData.legend()

    axResid.scatter(dsVals, residualsDmFit)
    axResid.set_ylabel('residual')
    axResid.axhline(0, c='gray')

#%%

# plt.axis('off')
plt.show()

#%%