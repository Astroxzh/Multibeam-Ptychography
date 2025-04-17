#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#%%
#unit: mm
def fresnelDiff(dz, wavelength=632e-6, dx=50e-3, N=2048):
    # source coordinates
    wavelength = 532e-6
    k = 2 * np.pi / wavelength
    L = N * dx
    x = np.arange(-N / 2, N / 2) * dx
    [X, Y] = np.meshgrid(x, x)

    #target coordiantes
    ds = wavelength * dz / L
    s = np.arange(-N / 2, N / 2) * ds
    [Sx, Sy] = np.meshgrid(s, s)
    propagator = np.exp(1j * k / (2 * dz) * (X**2 + Y**2))
    return propagator

def circ(x, y, D):
    circle = (x**2 + y**2) < (D / 2)**2
    return circle

def angularPropagator(dz, wavelength=632e-6, dx=50e-3, N=2048):
    k = 2 * np.pi / wavelength
    L = N * dx
    x = np.arange(-N / 2, N / 2) * dx
    [X, Y] = np.meshgrid(x, x)

    fx = np.arange(-N / 2, N / 2) / L
    Fx, Fy = np.meshgrid(fx, fx)
    fMax = L / (wavelength * np.sqrt(L**2 + 4 * dz**2))
    W = circ(Fx, Fy, 2 * fMax)
    H = np.exp(1j * k * dz * np.sqrt(1 - (Fx * wavelength)**2 - (Fy * wavelength)**2))
    return H * W

def sphWave(radius, dx=50e-3, wavelength=632e-6, N=2048):
  
    k = 2 * np.pi / wavelength
    L = N * dx
    x = np.arange(-N / 2, N / 2) * dx
    [X, Y] = np.meshgrid(x, x)

    R = np.sqrt(X**2 + Y**2 + radius**2) 
    if radius > 0 :
        sWave = 1 * np.exp(1j * k * R) / R
    elif radius < 0 :
        sWave = 1 * np.exp(-1j * k * R) / R
    return sWave

def objGeneration():
    obj = filePath = r'C:\0Something\Problem_sheet_7\XUV.png'
    aperture = Image.open(filePath).convert('L')
    obj = np.array(aperture)
    return obj


#%%

dx = 20e-3
N = 2048
wavelength = 632e-6

sphericalWave = sphWave(-10000, dx=dx, wavelength=wavelength, N=N)

obj = objGeneration()
exitWave = sphericalWave * obj

propagator = fresnelDiff(10000)
# finalWave = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(exitWave))*propagator))
finalWave = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(propagator * exitWave), norm='ortho'))
finalWaveP = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(obj))*propagator))


plt.figure()
plt.imshow(np.real(finalWaveP))
plt.figure()

plt.imshow(np.real(finalWave))
plt.show()
#%%
