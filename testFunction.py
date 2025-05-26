#%%
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os
import cv2
import utils
from scipy.stats import trim_mean

from matplotlib.patches import Circle
#%%
# --- 1. 读取图像并预处理 ---
folderPath = r'C:\Master Thesis\data\1 optimal probe touching\multiWavelength\f20\probeSize'
fileName = 'f20_10dm2step_4ds.npy'
filePath = os.path.join(folderPath, fileName)

dataSet = np.load(filePath)
img = np.abs(dataSet[1,...])**2
# 可选：先平滑一下以抑制噪声
sm = gaussian_filter(img, sigma=2)


# %%

_, binaryImg = cv2.threshold(sm, 0.6, 2, cv2.THRESH_BINARY)
# plt.imshow(binaryImg)

#%%
binaryImg = binaryImg.astype(np.uint8)
contours, _ = cv2.findContours(binaryImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(img, cmap=utils.setCustomColorMap())
cXs = 0
cYs = 0
ii = 0
for contour in contours:
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        ii += 1
        cXs += cX
        cYs += cY
        print(cX,cY)
        # plt.plot(cX, cY, 'r+', markersize=12)
        
# cXmean = trim_mean(cXs, proportiontocut=0.2, axis=None)
# cYmean = trim_mean(cYs, proportiontocut=0.2, axis=None)  
cXmean = cXs / ii
cYmean = cYs / ii
print (cXmean, cYmean)
plt.plot(cXmean, cYmean, 'b+', markersize=12)



#%%
center = np.array([cYmean, cXmean])
radius = utils.encircledEnergyRadiusSubpixel(img, center=center, fraction=0.8, pixel_size=(4/4000))

# --- 5. 可视化结果 ---
fig, axes = plt.subplots(1,2, figsize=(12, 4))

# 原图+中心标记
ax = axes[0]
ax.imshow(img, cmap='gray')
ax.plot(cXmean, cYmean, 'rx')
ax.set_title('原图与探针中心')
circle = Circle((cXmean, cYmean), radius,
                edgecolor='r',      # 边框颜色
                facecolor='none',   # 填充颜色（none 表示不填充）
                linewidth=2)        # 边框宽度
ax.add_patch(circle)

# # 子图0
# axes[1].imshow(sub0, cmap='gray')
# axes[1].set_title(f'Probe 0 sub-image')

# # 子图1
# axes[2].imshow(sub1, cmap='gray')
# axes[2].set_title(f'Probe 1 sub-image')

# plt.tight_layout()
# plt.show()

# # 包围能量曲线
# plt.figure(figsize=(6,4))
# plt.plot(rs0, ee0, label='Probe 0')
# plt.xlabel('Radius (pixels)')
# plt.ylabel('Encircled Energy')
# plt.legend()
# plt.title('Encircled Energy vs. Radius')
# plt.grid(True)


#%%
plt.show()

# %%
