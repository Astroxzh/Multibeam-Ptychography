#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path 
import os
import re
import utils
# %%
folderPath = Path(r'C:\Master Thesis\data\1 optimal probe touching\probeOnCamera')
folerName = 'f40dm24ds12_1024'
folder = os.path.join(folderPath, folerName)
suffix = '.hdf5'
# fileList = list(folder.glob(f'*{suffix}'))
# print(fileList)
for filename in os.listdir(folder):
    if filename.endswith(suffix):
        h5Path = os.path.join(folder, filename)
        with h5py.File(h5Path, 'r') as f:
            images = f['ptychogram'][()]
            im = images[0,0,:,:]
            shape = np.shape(im)
        break


img = np.zeros(shape)
for filename in os.listdir(folder):
    if filename.endswith(suffix):
        match = re.search(r'S(\d+)', filename)
        imgIndex = int(match.group(1))
        h5Path = os.path.join(folder, filename)
        with h5py.File(h5Path, 'r') as f:
            images = f['ptychogram'][()]
            img += images[0,imgIndex-1,:,:]

# %%
saveDir = r'C:\Master Thesis\data\1 optimal probe touching\probeOnCamera\result'
fileName = folerName + '.png'
savePath = os.path.join(saveDir, fileName)
plt.imshow(np.log10(np.abs(img)**2+1), cmap=utils.setCustomColorMap())
# plt.xlabel('mm')
# plt.ylabel('mm')
plt.savefig(savePath, dpi=100, bbox_inches='tight')
# plt.show()
#%%
