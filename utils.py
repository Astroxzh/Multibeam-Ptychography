import numpy as np
from matplotlib.colors import LinearSegmentedColormap



def circ(x, y, D):
    circle = (x**2 + y**2) < (D / 2)**2
    return circle

def gaussian2D(n, std):
    # create the grid of (x,y) values
    n = (n - 1) // 2
    x, y = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    # analytic function
    h = np.exp(-(x**2 + y**2) / (2 * std**2))
    # truncate very small values to zero
    mask = h < np.finfo(float).eps * np.max(h)
    h *= 1 - mask
    # normalize filter to unit L1 energy
    sumh = np.sum(h)
    if sumh != 0:
        h = h / sumh
    return h

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

def angularPro(u, propagator):
    return np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u))*propagator))

def fresnelPro(u, dz, wavelength, dx):
    N = u.shape[0]
    k = 2 * np.pi / wavelength
    x = np.arange(-N / 2, N / 2) * dx
    X, Y = np.meshgrid(x, x)

    prePhase = np.exp(1j * k / (2 * dz) * (X**2 + Y**2))
    uPre = u * prePhase
    uFre = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(uPre)))

    df = 1 / (N * dx)
    fx = np.arange(-N / 2, N / 2) * df
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * dz * (FX**2 + FY**2))
    uFre = uFre * H
    uOut = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uFre)))
    # postPhase = np.exp(1j * k / (2 * dz) * (X**2 + Y**2))
    # uout = uout * postPhase
    
    return uOut


def padding(obj, targetSize):
    h, w = obj.shape
    if h > targetSize[0] or w > targetSize[1]:
        raise ValueError('targetSize must larger than objSize')
    
    padH = targetSize[0] - h 
    padW = targetSize[1] - w
    padTop = padH // 2
    padBottom = padH - padTop
    padLeft = padW // 2
    padRight = padW - padLeft
    padded = np.pad(obj, ((padTop, padBottom), (padLeft, padRight)), mode='constant', constant_values=0)
    return padded

def demagFourier(obj, magFactor):
    if magFactor < 1:
        raise ValueError('magFacor must larger than 1')
    else:
        centerx, centery = np.shape(obj)[0] // 2 - 1, np.shape(obj)[1] // 2 - 1
        shapeobj = np.shape(obj)
        freobj = np.fft.fftshift(np.fft.fft2(obj))
        cropRegion = freobj[centerx-(shapeobj[0]//magFactor)//2: centerx+(shapeobj[0]//magFactor)//2, 
                            centery-(shapeobj[1]//magFactor)//2: centery+(shapeobj[1]//magFactor)//2]
        demagObj = np.fft.ifft2(np.fft.ifftshift(cropRegion))
        demagObj = padding(demagObj, shapeobj)

        return demagObj
    
def spiral_blade_mask(wavelength=13.5e-9, f=0.6e-3, N=256, dx=10e-9, n_blades=3, blades_diameter=8e-6, angle=None, factor=0):
    """
    :param wavelength: target wavelength
    :param f: focus distance, the smaller --> more twisting of the blades around the center
    :param N: #pixels along 1-direction
    :param dx: pixel space
    :param n_blades: # of blades to generate
    :param blades_diameter: extension of the blades
    :param angle: used to strech along x-direction the pattern
    :param factor: [0-1] increases the fill factor of the spiral, default=0 is 50%, factor=0.6 is 70% fill factor
    :return: binary array NxN where 0 represents the blade structures
    """
    if angle is not None:
        stretching_factor = 1 / np.cos(np.deg2rad(angle))
    else:
        stretching_factor = 1

    x = np.arange(-N / 2, N / 2) * dx
    y = np.copy(x)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid /= stretching_factor
    # y_grid *= stretching_factor
    r = abs(x_grid ** 2 + y_grid ** 2) ** (1 / 2)
    # r = (x_grid ** 2 + (y_grid*stretching_factor) ** 2) ** (1 / 2)
    # r = ((x_grid/stretching_factor) ** 2 + (y_grid) ** 2) ** (1 / 2)


    phi = np.arctan2(y_grid, x_grid)
    # phi = np.arctan2(y_grid*stretching_factor, x_grid)
    # phi = np.arctan2(y_grid, x_grid/stretching_factor)

    # n_blades = n_blades % 5 + 2  #  minimum amount of blades = 2, increases to 6 anc cycles back
    data = np.exp(-1j * np.pi * r ** 2 / f / wavelength) * np.exp(1j * n_blades * phi)

    binary = np.real(data) < factor

    circ = x_grid ** 2 + y_grid ** 2 < (blades_diameter / 2) ** 2
    # circ = (x_grid/np.sqrt(2)) ** 2 + y_grid ** 2 < (blades_diameter / 2) ** 2

    binary = circ * binary
    binary = (binary).astype(int)

    return binary

def maskGeneration(numOfMask=8, wavelength=13.5e-9, f=0.6e-3, N=256, dx=10e-9, blades_diameter=200e-6 , angle=None, factor=0):
    maskList = []
    for ii in range(numOfMask):
        submask = spiral_blade_mask(wavelength=wavelength, f=f, N=N, dx=dx, n_blades=ii+1, blades_diameter=blades_diameter, angle=angle, factor=factor)
        flipedsubmask = np.flipud(submask)
        maskList.append(np.vstack([submask, flipedsubmask]))
    columns = [np.vstack([maskList[i], maskList[i+1]]) for i in range(0,numOfMask,2)]
    return np.array(np.hstack(columns))
    
def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

def complex2rgb(u, amplitudeScalingFactor=1, scalling=1):
    h = np.angle(u)
    h = (h + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = np.abs(u)
    if amplitudeScalingFactor != 1:
        v[v > amplitudeScalingFactor * np.max(v)] = amplitudeScalingFactor * np.max(v)
    if scalling != 1:
        local_max = np.max(v)
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)
        print(f'ratio: {local_max / scalling}, max(v): {np.max(v)}')

        v *= local_max / scalling
        print(f'max(v)L {np.max(v)}')
    
    else:
        v = v / (np.max(v) + np.finfo(float).eps) * (2 ** 8 - 1)

    hsv = np.dstack([h, s, v])
    rgb = hsv2rgb(hsv)
    return rgb

def setCustomColorMap():
    colors = [
        (1, 1, 1),
        (0, 0.0875, 1),
        (0, 0.4928, 1),
        (0, 1, 0),
        (1, 0.6614, 0),
        (1, 0.4384, 0),
        (0.8361, 0, 0),
        (0.6505, 0, 0),
        (0.4882, 0, 0),
    ]

    n = 255
    cm = LinearSegmentedColormap.from_list('cmap', colors, n)
    return cm

def ifft2c(array):
    """
    performs 2 - dimensional inverse Fourier transformation, where energy is reserved abs(G)**2==abs(fft2c(g))**2
    if G is two - dimensional, fft2c(G) yields the 2D iDFT of G
    if G is multi - dimensional, fft2c(G) yields the 2D iDFT of G along the last two axes
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array), norm='ortho'))


def fft2c(array):
    """
    performs 2 - dimensional unitary Fourier transformation, where energy is reserved abs(g)**2==abs(fft2c(g))**2
    if g is two - dimensional, fft2c(g) yields the 2D DFT of g
    if g is multi - dimensional, fft2c(g) yields the 2D DFT of g along the last two axes
    :param array:
    :return:
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(array), norm='ortho'))

def aspw(u, wavelength, dx, dz):

    k = 2 * np.pi / wavelength
    # source coordinates, this assumes that the field is NxN pixels
    N = u.shape[-1]
    L = N * dx
    linspacex = np.linspace(-N / 2, N / 2, N, endpoint=False).reshape(1, N)
    Fx = linspacex / L
    Fy = Fx.reshape(N, 1)

    f_max = 1 / (wavelength * np.sqrt(1 + (2 * dz / L) ** 2))
    W = circ(Fx, Fy, 2 * f_max)
    # w accounts for circular symmetry of transfer function and imposes bandlimit to avoid sampling issues
    w = 1 / wavelength ** 2 - Fx ** 2 - Fy ** 2
    w[w >= 0] = np.sqrt(w[w >= 0])
    w[w < 0] = 0
    # w = np.sqrt(w, dtype=complex)
    H = np.exp(1.j * 2 * np.pi * dz * w) * W
    # H = np.exp(1.j * k * dz * w) * W

    U = fft2c(u)
    u_new = ifft2c(U * H)
    return u_new

def encircledEnergyRadius(I, fraction=0.95, pixel_size=1.0):
    
    ny, nx = I.shape
    y_idx, x_idx = np.indices(I.shape)
    total_energy = I.sum()
    
    # x_c = (I * x_idx).sum() / total_energy
    # y_c = (I * y_idx).sum() / total_energy
    x_c, y_c = np.unravel_index(np.argmax(I), I.shape)

    dx = dy = pixel_size
    if not np.isscalar(pixel_size):
        dx, dy = pixel_size
    dist = np.sqrt(((x_idx - x_c) * dx)**2 + ((y_idx - y_c) * dy)**2)
    
    flat_I    = I.ravel()
    flat_dist = dist.ravel()
    order = np.argsort(flat_dist)
    cum_intensity = np.cumsum(flat_I[order])
    
    target = fraction * total_energy
    idx = np.searchsorted(cum_intensity, target)
    r = flat_dist[order][idx]
    return r