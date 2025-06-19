import os
import h5py
import logging
import matplotlib

import PtyLab
from PtyLab.utils.visualisation import show3Dslider
from PtyLab import Engines

matplotlib.use('tkagg')
from PtyLab.utils.utils import *

# import fracPy
# from fracPy.io import getExampleDataFolder
# from fracPy import Engines
# from fracPy.utils.visualisation import show3Dslider
# from fracPy.utils.utils_ import *

logging.basicConfig(level=logging.INFO)
import numpy as np
from tqdm import tqdm
from IAP.Tools.propagators import fft2c
from peak_finders import *

path = os.path.join(os.path.dirname(__file__), 'data/34')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def save_reconstruction(name_extension=''):
    # Save results
    path = f'{savepath}/{filename}_{npsm}npsm.hdf5' if save_name is None else f'{savepath}/{save_name}{name_extension}.hdf5'
    with h5py.File(path, 'w') as hf:
        hf['object'] = reconstruction.object
        hf['probe'] = reconstruction.probe
        hf['spectrum'] = reconstruction.spectralDensity
        hf['wavelength'] = reconstruction.wavelength
        hf['dxp'] = reconstruction.dxp
        hf['dxo'] = reconstruction.dxo
        try:
            hf['thetaHistory'] = reconstruction.thetaHistory
        except:
            pass
        try:
            hf['zHistory'] = reconstruction.zHistory
        except:
            pass
        hf.close()
    print(f'Results saved to {path}')


def load_probe(path, reconstruction):
    with h5py.File(path, 'r') as hf:
        probe = hf.get('probe')[()]
        hf.close()
    # if (not (probe.shape == reconstruction.probe.shape)):
    #     probe = cropCenter(probe, reconstruction.Np)
    reconstruction.probe = probe
    print("probe loaded")


def load_object(path, reconstruction):
    with h5py.File(path, 'r') as hf:
        object = hf.get('object')[()]
        hf.close()
    # if (not (object.shape == reconstruction.object.shape)):
    #     object = cropCenter(object, reconstruction.Np)
    reconstruction.object = object
    print("object loaded")


# specify experiment data folder
exp_number = '87'
# specify number of pixel
Nd = 256
# appendix to filename of results
save_appendix = f'_reconstruction_it'
# specify number of iterations
iteration_stop_1 = 100
iteration_stop_2 = 300
# run reconstruction until preset convergence criterion is met
iteration_until_convergence = True
# specify whether a seed from a previous reconstruction shall be used
c_seed_probe = 0
c_seed_object = 0

n_beams = 16
npsm = 2  # number of probe orthogonal modes to use in the reconstruction

save_appendix = f'_reconstruction_so_it'

data_path = f'/home/hinrich/python_projects/Reconstructions/data'
savepath = f'/home/hinrich/python_projects/Reconstructions/data/{exp_number}'
seedpath = f'/home/hinrich/python_projects/Reconstructions/data/{exp_number}'
seed_path_object = f'/home/hinrich/python_projects/Reconstructions/data/87/2024_07_11_16x_70ms_76_77pos_pp_1000pos_abg_32_256px_reconstruction_so_it_80.hdf5'

openpath = savepath

# specify which preprocessed data to reconstruct
filename_list = os.listdir(openpath)
for i, filename in enumerate(filename_list):
    print(i, filename)
c_index = int(input("choose file: "))
filename = filename_list[c_index].split('.')[0]

save_name = filename
filePath = f'{openpath}/{filename}.hdf5'


experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(filePath, operationMode='CPM')
'''alternative to adjust experimental parameters before reconstruction'''
spectral_power = np.array([1] * n_beams) / n_beams
spectral_power /= np.sum(spectral_power)

# experimentalData.showPtychogram()
Nd = experimentalData.Nd
experimentalData.setOrientation(1)  # !!! very important and dependent on camera orientation
experimentalData.spectralDensity = [experimentalData.wavelength, ] * n_beams
experimentalData.spectralPower = spectral_power
reconstruction.spectralDensity = experimentalData.spectralDensity
experimentalData.dq = 6.5e-6 * (int(2048 / Nd))
experimentalData._setData()
reconstruction.copyAttributesFromExperiment(experimentalData)
reconstruction.computeParameters()
print(f'exp_data_shape. {experimentalData.ptychogram.shape}')
# now, all our experimental data is loaded into experimental_data and we don't have to worry about it anymore.
# now create an object to hold everything we're eventually interested in
reconstruction.npsm = npsm  # Number of probe modes to reconstruct
reconstruction.nosm = 1  # Number of object modes to reconstruct
reconstruction.nlambda = len(experimentalData.spectralDensity)  # Number of wavelength
reconstruction.nslice = 1  # Number of object slice

# experimentalData.entrancePupilDiameter = 4e-3
reconstruction.initialProbe = 'ones'
reconstruction.initialObject = 'ones'
# initialize probe and object and related Params
reconstruction.initializeObjectProbe()

shift_m = 1318e-6
# spatial shift of probe modes
d_px = np.round(shift_m / reconstruction.dxp, decimals=0)
print(f'shift_px: {d_px}')
# 16 sources grid layout
shift_col = -1 * np.array([-d_px * 3 / 2, -d_px / 2, d_px / 2, d_px * 3 / 2] * 4)
shift_row = np.array([*[-d_px * 3 / 2] * 4, *[-d_px / 2] * 4, *[d_px / 2] * 4, *[d_px * 3 / 2] * 4])

new_order = [3, 2, 1, 0,
             7, 6, 5, 4,
             11, 10, 9, 8,
             15, 14, 13, 12]
# probe_mx = probe_mx[new_order, ...]
shift_col = shift_col[new_order]
shift_row = shift_row[new_order]
# duplicate coordinates
shift_row = np.hstack((shift_row, shift_row))
shift_col = np.hstack((shift_col, shift_col))

# # seed probe
 # add phase ramp for each source
if c_seed_probe != 1:
    c_mass = multiple_peak_finder_2D(np.sum(experimentalData.ptychogram, axis=0))
    print(c_mass)
    # show3Dslider(np.sum(experimentalData.ptychogram, axis=0))
    c_mass -= int(Nd / 2)

    zo = 45.6e-3
    wl = 632e-9
    dq = 6.5e-6 * (int(2048 / Nd))
    dx = wl * zo / (Nd * dq)
    xp = np.arange(-Nd / 2, Nd / 2) * dx
    Xp, Yp = np.meshgrid(xp, xp)
    x_pos_m = c_mass[:, -1] * dq
    y_pos_m = c_mass[:, 0] * dq

    if Nd == 512:
        seed = 'probe_mx_512px'
    elif Nd == 256:
        seed = 'probe_mx_256px'

    with h5py.File(f'{data_path}/{seed}.h5', 'r') as f:
        probe_mx = f.get('probe_mx')[()]
        probe_mx = np.abs(probe_mx) + 1j * 0

    for i in range(16):
        # probe_mx[i] = np.fliplr(probe_mx[i])
        # probe_mx[i] = np.flipud(probe_mx[i])
        probe_mx[i] = np.rot90(probe_mx[i])
        # probe_mx[i] = np.squeeze(reconstruction.probe)[i]

    # derived from 2x2 measurement
    new_order = [6, 5,
                 10, 9]

    new_order = [3, 2, 1, 0,
                 7, 6, 5, 4,
                 11, 10, 9, 8,
                 15, 14, 13, 12]

    # show3Dslider(np.abs(probe_mx))
    probe_mx = probe_mx[new_order, ...]
    # show3Dslider(np.abs(probe_mx))
    for i, probe in enumerate(probe_mx):
        probe *= np.exp(1j * 2 * np.pi * (1 / wl) * ((Xp + x_pos_m[i]) ** 2 + (Yp + y_pos_m[i]) ** 2) / (2 * zo))

        # probe *= np.exp(1j * 2 * np.pi * (1 / experimentalData.wavelength) * (
        #             (reconstruction.Xp + x_pos_m[i]) ** 2 + (reconstruction.Yp + y_pos_m[i]) ** 2) / (
        #                             2 * experimentalData.zo))
    # probe_mx = np.fliplr(probe_mx)
    # probe_mx = np.flipud(probe_mx)
    # show3Dslider(np.abs(probe_mx))

    for i in range(16):
        reconstruction.probe[i] *= probe_mx[i]

# reconstruction.probe = np.swapaxes(reconstruction.probe, 0, 2)
print(reconstruction.probe.shape)
# for j in range(2):
#   for i in range(16):
#       reconstruction.probe[0,0,int(i+16*j),0,...] = probe[i,0,j,0,...]
# seed reconstructed probe
# reconstruction.probe = probe

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
# monitor = Monitor()
monitor.figureUpdateFrequency = 5
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'low'  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 0.25  # .5#1.5  # control object plot FoV
monitor.probeZoom = 1  # control probe plot FoV

## main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagatorType = 'Fraunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP twoStepPolychrome
params.intensityConstraint = "standard"

## how do we want to reconstruct?
params.gpuSwitch = True
params.probePowerCorrectionSwitch = True
params.comStabilizationSwitch = False  # False
params.modulusEnforcedProbeSwitch = False
params.orthogonalizationSwitch = True
params.orthogonalizationFrequency = 10
params.fftshiftSwitch = False
params.absorbingProbeBoundary = True
params.objectContrastSwitch = False
params.absObjectSwitch = False
params.backgroundModeSwitch = False
# for non-dispersive objects coupling = True, Aleph between [0, 1], where 1 means average between wavelengths
params.couplingSwitch = False
params.couplingAleph = 0.05
params.positionCorrectionSwitch = False

'''Type of engine to use'''
# %%
mPIE = Engines.mPIE_mx2(reconstruction, experimentalData, params, monitor)


if iteration_stop_1 != 0:
    mPIE.numIterations = iteration_stop_1
    mPIE.betaProbe = 0.25
    mPIE.betaObject = 0.25
    # mPIE.alphaObject = 1  # object regularization
    mPIE.reconstruct(shift_col, shift_row)
    save_reconstruction(save_appendix + f'_{len(reconstruction.error)}')

# #manually extend iteration process
# while iteration_one_percent:
#     n_iterations = int(input('How many iterations? :'))
#     iteration_stop_1 += n_iterations
#     if n_iterations == 0:
#         save_reconstruction(save_appendix + f'_{int(iteration_stop_1)}_one_percent_optimized')
#         break
#     mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
#     mPIE.numIterations = n_iterations
#     mPIE.betaProbe = 0.025
#     mPIE.betaObject = 0.15
#     # mPIE.alphaObject = 1  # object regularization
#     mPIE.reconstruct()

# run reconstruction until certain convergence threshold is met, e.g. only 1% change in error metric over 100 iterations
while iteration_until_convergence:
    # get list of error metric for each iteration
    absolute_error = reconstruction.error
    # specify the iteration span to be compared, display is changed in PtyLab/Monitor/Plots.py line 152
    it_comp = 20
    # specify threshold to reach e.g 0.99 for 1% or 0.998 for 0.2%
    optimization_target = 0.99
    # do some more iterations if iterations<it_comp
    if len(absolute_error) > it_comp:
        rel_error = absolute_error[-it_comp:-1] / absolute_error[-it_comp]
    else:
        rel_error = [0]
        n_iterations = it_comp
    # check whether the threshold is reached
    if 1 > rel_error[-1] > optimization_target:
        if 1 > rel_error[-2] > optimization_target and 1 > rel_error[-3] > optimization_target and 1 > rel_error[
            -4] > optimization_target:
            save_reconstruction(save_appendix + f'_{len(reconstruction.error)}_one_percent_optimized')
            break
        else:
            n_iterations = 10
    else:
        n_iterations = 60
    iteration_stop_1 += n_iterations
    # mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
    mPIE.numIterations = n_iterations
    mPIE.betaProbe = 0.10
    mPIE.betaObject = 0.25
    # mPIE.alphaObject = 1  # object regularization
    mPIE.reconstruct(shift_col, shift_row)


if iteration_stop_2 != 0:
    # mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
    mPIE.numIterations = iteration_stop_2
    mPIE.betaProbe = 0.25
    mPIE.betaObject = 0.25
    # mPIE.alphaObject = 1  # object regularization
    mPIE.alphaProbe = 0.9  # probe regularization
    mPIE.alphaObject = 1  # object regularization
    mPIE.reconstruct(shift_col, shift_row)
    save_reconstruction(save_appendix + f'_{len(reconstruction.error)}')
