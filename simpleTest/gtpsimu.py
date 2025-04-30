import numpy as np
import matplotlib.pyplot as plt

# 参数设置
wavelength = 632.8e-9   # 例如：632.8 nm （氦氖激光）
k = 2 * np.pi / wavelength
f = 0.1                 # 焦距，比如 100 mm
L = 5e-3                # 模拟窗口尺寸（5 mm）
N = 1024                # 网格点数
dx = L / N
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# 定义单个光斑（使用矩形窗口或高斯函数）
def spot(x, center, width):
    return np.where(np.abs(x - center) < width/2, 1.0, 0.0)

# 四个光斑中心位置 (沿x轴均匀分布)
centers = [-1e-3, -0.33e-3, 0.33e-3, 1e-3]  # 单位: m
width = 0.2e-3  # 光斑宽度 (200 μm)

# 构造初始场：在 x 方向上叠加，y方向上取一个较宽的窗口（或采用高斯分布）
U0 = np.zeros((N, N), dtype=complex)
for cx in centers:
    U0 += np.outer(spot(x, cx, width), np.ones(N))  # 这里简单假设 y 方向均匀

# 加入凸透镜的相位因子
lens_phase = np.exp(-1j * k/(2*f) * (X**2 + Y**2))
U_lens = U0 * lens_phase


# 定义菲涅尔传播函数（简单方法，利用 FFT 实现）
def fresnel_propagation(Uin, z, wavelength, dx):
    N = Uin.shape[0]
    k = 2*np.pi / wavelength
    # 坐标轴
    x = np.linspace(-N/2, N/2 - 1, N) * dx
    X, Y = np.meshgrid(x, x)
    
    # 预乘二次相位因子
    pre_phase = np.exp(1j * k/(2*z) * (X**2 + Y**2))
    U_pre = Uin * pre_phase

    # FFT 移位
    U_freq = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_pre)))
    
    # 空间频率变量 (fx, fy)
    df = 1/(N * dx)
    fx = np.linspace(-N/2, N/2 - 1, N) * df
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    U_freq = U_freq * H
    # 逆 FFT
    U_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_freq)))
    
    # 后乘二次相位因子
    post_phase = np.exp(1j * k/(2*z) * (X**2 + Y**2))
    U_out = U_out * post_phase
    
    # 常数项 (不影响相对强度)
    U_out = U_out * np.exp(1j*k*z)/(1j * wavelength * z)
    
    return U_out

# 选择不同传播距离 z，例如：从 0.05 m 到 0.15 m
z_values = [0.05, 0.1, 0.15]
plt.figure(figsize=(12, 4))
for i, z in enumerate(z_values):
    U_z = fresnel_propagation(U_lens, z, wavelength, dx)
    intensity = np.abs(U_z)**2
    plt.subplot(1, len(z_values), i+1)
    plt.imshow(intensity, extent=[x[0]*1e3, x[-1]*1e3, y[0]*1e3, y[-1]*1e3], cmap='inferno')
    plt.title(f'z = {z*1e3:.0f} mm')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
plt.tight_layout()
plt.show()