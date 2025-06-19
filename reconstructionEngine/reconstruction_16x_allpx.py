import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')
from PtyLab.utils.visualisation import show3Dslider
import PtyLab
import h5py
# import fracPy
# from fracPy.io import getExampleDataFolder
# from fracPy import Engines
from PtyLab import Engines
import logging

# from fracPy.utils.visualisation import show3Dslider
logging.basicConfig(level=logging.INFO)
import numpy as np
import os
# from fracPy.utils.utils_ import *
from PtyLab.utils.utils import *
from IAP.Tools.propagators import fft2c
from tqdm import tqdm
from peak_finders import *

# choose GPU by setting '0' or '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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


'''START'''
# choose exp number
exp_number = '87'
save_appendix = f'_reconstruction_it'
# choose plot frequency (every nth iteration)
c_plot_freq = 20
# iteration stops
iteration_stop_1 = 100
iteration_stop_2 = 5
# run reconstruction until preset convergence criterion is met
iteration_until_convergence = True
# choose to seed the reconstruction with a reconstructed probe beam or object by changing value 0->1
c_seed_probe = 0
c_seed_object = 0
# determine number of beams and modes
n_beams = 16
# number of probe orthogonal modes to use in the reconstruction
npsm = 2
# save results in same folder as data
data_path = f'/home/hinrich/python_projects/Reconstructions/data'
savepath = f'/home/hinrich/python_projects/Reconstructions/data/{exp_number}'
seedpath = f'/home/hinrich/python_projects/Reconstructions/data/{exp_number}'
openpath = savepath
# path that contains reconstruction results for seeding

# choose preprocessed data file
filename_list = os.listdir(openpath)
filename_list = [filename for filename in filename_list if
                 "reconstruction" not in filename.lower() and "pp" in filename.lower()]
for i, filename in enumerate(filename_list):
    print(i, filename)
c_index = int(input("choose data file: "))
filename = filename_list[c_index].split('.')[0]

save_name = filename
filePath = f'{openpath}/{filename}.hdf5'

save_name += f"_{npsm}npsm"
spectral_power = np.array([1] * n_beams) / n_beams
spectral_power /= np.sum(spectral_power)
experimentalData, reconstruction, params, monitor, ePIE_engine = PtyLab.easyInitialize(filePath, operationMode='CPM')

'''alternative to adjust experimental parameters before reconstruction'''
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

'''seeded probe with calculated phase ramp'''
# # add phase ramp for each source
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

    show3Dslider(np.abs(probe_mx))
    probe_mx = probe_mx[new_order, ...]
    show3Dslider(np.abs(probe_mx))
    for i, probe in enumerate(probe_mx):
        probe *= np.exp(1j * 2 * np.pi * (1 / wl) * ((Xp + x_pos_m[i]) ** 2 + (Yp + y_pos_m[i]) ** 2) / (2 * zo))

        # probe *= np.exp(1j * 2 * np.pi * (1 / experimentalData.wavelength) * (
        #             (reconstruction.Xp + x_pos_m[i]) ** 2 + (reconstruction.Yp + y_pos_m[i]) ** 2) / (
        #                             2 * experimentalData.zo))
    # probe_mx = np.fliplr(probe_mx)
    # probe_mx = np.flipud(probe_mx)
    show3Dslider(np.abs(probe_mx))

    for i in range(16):
        reconstruction.probe[i] *= probe_mx[i]

    show3Dslider(np.abs(reconstruction.probe[:, 0, 0, 0, :, :]))
    show3Dslider(np.abs(np.sum(experimentalData.ptychogram, axis=0)))
# show3Dslider(np.abs(probe_mx))
# show3Dslider(np.abs(np.sum(fft2c(probe_mx)))/np.max(np.abs(np.sum(fft2c(probe_mx))))
#              -np.abs(np.sum(experimentalData.ptychogram, axis=0))/np.max(np.abs(np.sum(experimentalData.ptychogram, axis=0))))


# seeded previous probe if desired
# load_probe(f'/home/hinrich/python_projects/Reconstructions/data/31/2024_05_13_b_scan_new_lens_pp_TEST_bg_158_60_it.hdf5',reconstruction)

# %% unknown
# f = 35e-3 #0.3e-2
# reconstruction.probe = reconstruction.probe * np.exp(1.j * 2 * np.pi / reconstruction.wavelength *
#                                                     ((reconstruction.Xp) ** 2 + reconstruction.Yp ** 2) / (2 * f))
# seed = "10082023_16x_8Bin_100ms_2000Pos_93Perc_1_2npsm_FRC_2000m"
# with h5py.File(f'{savepath}/{seed}.hdf5','r') as f:
#     probe = f.get('probe')[()]
#     f.close()
# reconstruction.probe = probe


# seed phase of reconstructed object
# for i in range(16):
#     norm_amp = np.abs(object_[i])
#     norm_amp /= np.amax(norm_amp)
#     reconstruction.object[i] *= np.exp(1j*np.angle(object_[i])*norm_amp)


# seed object
# choose seed file
if c_seed_probe or c_seed_object:
    filename_list = os.listdir(seedpath)
    filename_list = [filename for filename in filename_list if
                     "reconstruction" in filename.lower() and str(Nd) in filename.lower()]
    for i, filename in enumerate(filename_list):
        print(i, filename)
    c_index = int(input("choose seed file: "))
    filename = filename_list[c_index]
    seed_filepath = os.path.join(data_path, exp_number, filename)
    print(seed_filepath)
# seed object
if c_seed_object == 1:
    load_object(seed_filepath, reconstruction)
# seed probe
if c_seed_probe == 1:
    load_probe(seed_filepath, reconstruction)

# this will copy any attributes from experimental data that we might care to optimize
# # Set monitor properties
# monitor = Monitor()
monitor.figureUpdateFrequency = c_plot_freq
monitor.objectPlot = 'complex'  # complex abs angle
monitor.verboseLevel = 'low'  # high: plot two figures, low: plot only one figure
monitor.objectZoom = 0.99  # .5#1.5  # control object plot FoV
monitor.probeZoom = 0.5  # control probe plot FoV

# %% main parameters
params.positionOrder = 'random'  # 'sequential' or 'random'
params.propagatorType = 'Fraunhofer'  # Fraunhofer Fresnel ASP scaledASP polychromeASP scaledPolychromeASP twoStepPolychrome

# %% how do we want to reconstruct?
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

if iteration_stop_1 != 0:
    mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
    mPIE.numIterations = iteration_stop_1
    mPIE.betaProbe = 0.025
    mPIE.betaObject = 0.15
    # mPIE.alphaObject = 1  # object regularization
    mPIE.reconstruct()
    save_reconstruction(save_appendix + f'_{int(iteration_stop_1)}')

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
            save_reconstruction(save_appendix + f'_{int(iteration_stop_1)}_one_percent_optimized')
            break
        else:
            n_iterations = 10
    else:
        n_iterations = 60
    iteration_stop_1 += n_iterations
    # mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
    mPIE.numIterations = n_iterations
    # mPIE.betaProbe = 0.025
    # mPIE.betaObject = 0.15
    # mPIE.alphaObject = 1  # object regularization
    mPIE.reconstruct()

if iteration_stop_2 != 0:
    # mPIE = Engines.mPIE(reconstruction, experimentalData, params, monitor)
    mPIE.numIterations = iteration_stop_2
    mPIE.betaProbe = 0.25
    mPIE.betaObject = 0.25
    # mPIE.alphaObject = 1  # object regularization
    mPIE.alphaProbe = 0.9  # probe regularization
    mPIE.alphaObject = 1  # object regularization
    mPIE.reconstruct()
    save_reconstruction(save_appendix + f'_{int(iteration_stop_1 + iteration_stop_2 )}')
