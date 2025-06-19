import os
import sys
import h5py
# PtyLab Modules
from PtyLab.utils.utils import *
from PtyLab.utils.visualisation import show3Dslider
# IAP.Tools by Daniel SPM for additional preprocessing steps
from IAP.Tools.misc import binning
from IAP.Tools.misc import cropCenter

# append for using functions defined
sys.path.append('/home/hinrich/python_projects/Reconstructions')

# specify experiment data folder
exp_number = "87"

# wavelength
# bg(background) subtraction
# subtract mean of recorded bg after each position, for superK source set to 0, since the bg measurement is illuminated as wll
c_standard_bg = 1
# additional bg subtraction measured in counts, to enhance reconstruction results
bg_additional_subtraction = 10 ** 1.2  # 10**2.2 / 10**0.5 / 0
# bin data from 512px to 256px if set to 1
c_binning = 1
# ? only for single beam measurements
c_crop_center = 0
# only keep every second (=1) or fourth recording position (=2), reducing e.g 2000pos to 1000pos or 500pos
c_half_recording_positions = 0

# variation parameter for finding zo for new setup
# dist_i=10

raw_data_path = (f'/home/hinrich/python_projects/Reconstructions/data/' + exp_number)
result_data_path = (f'/home/hinrich/python_projects/Reconstructions/data/' + exp_number)
# appendix to filename of results
save_appendix = f''

# get all filenames in specified raw data folder
filename_list = os.listdir(raw_data_path)
# choose shortest filename that is raw original file
filename = (min(filename_list, key=len)).split('.')[0]

# # alternatively ask the user which file to preprocess
# for i,filename in enumerate(filename_list):
#     print(i, filename)
# c_index = int(input("choose file: "))
# filename = filename_list[c_index].split('.')[0]

print(filename)

# load raw data file
with h5py.File(f'{raw_data_path}/{filename}.hdf5', 'r') as hf:
    print(hf.keys())
    bk = hf.get('background')[()]
    ptychogram = np.squeeze(hf.get('ptychogram')[()])
    encoder = hf.get('encoder')[()]
    dxd = float(hf.get('dxd')[()])
    Nd = int(hf.get('Nd')[()])
    zo = float(hf.get('zo')[()])
    beam_diameter = float(hf.get('entrancePupilDiameter')[()])
    wavelength = float(hf.get('wavelength')[()])
    hf.close()
# %% reducing record postions
if c_half_recording_positions == 1:
    ptychogram = ptychogram[:1002, ...]
    bk = bk[:1002, ...]
    encoder = encoder[:1002, ...]
    save_appendix += f'_1000pos'
if c_half_recording_positions == 2:
    ptychogram = ptychogram[:1002:2, ...]
    bk = bk[:1002:2, ...]
    encoder = encoder[:1002:2, ...]
    save_appendix += f'_500pos'
# %% background correction
# standard bg correction, subtracts mean bg from every ptychogram
if c_standard_bg == 1:
    ptychogram -= np.mean(bk, axis=0)

# subtract additional arbitrarily chosen counts
if bg_additional_subtraction != 0:
    ptychogram -= bg_additional_subtraction
    save_appendix += f'_abg_{bg_additional_subtraction:.0f}'

# set negative values induced by bg subtraction to zero or to one to avoid zero division in reconstruction
ptychogram[ptychogram <= 0] = 1  # 0

# %% binning
if c_binning == 1:
    pty = np.zeros((ptychogram.shape[0], ptychogram.shape[-2] // 2, ptychogram.shape[-1] // 2), dtype=float)
    for i in range(pty.shape[0]):
        pty[i, ...] = binning(ptychogram[i, ...], 2)
    Nd = int(Nd / 2)
    dxd *= 2
    save_appendix += f'_256px'
    ptychogram = pty
else:
    save_appendix += f'_512px'

# %% visualization
# show3Dslider(np.log10(bk_mean+1))
# show3Dslider(np.log10(bk_std+1))
# show3Dslider(np.log10(ptychogram+1))
# show3Dslider(np.log10(bk+1))
show3Dslider(np.log10(ptychogram + 1))
show3Dslider(np.log10(bk + 1))

# finds center of mass of ptychogram
# PSD = ptychogram.mean(axis=0)
# cy, cx = ndi.center_of_mass(PSD)
# print(f'{cy},{cx}')
# # Centers the ptychogram according to center of mass of its PSD
# for i in range(ptychogram.shape[0]):
#     ptychogram[i,...] = re_center_ptychogram(ptychogram[i,...], center_coord=np.array([cy, cx]))

pty = ptychogram.copy()

if c_crop_center == 1:
    pty = cropCenter(pty, Nd)
    save_appendix += f'_cc'

# distance object to detector, optimized 10.05.2024
zo = 45.6e-3
# wavlength of the illumination
wavelength = 632.0e-9

# beam_diameter = 0.2e-3
# Number of photons after bkg substraction
# print(f'Total NPhotons:\n'
#       f'{np.sum(pty):.4E}\n'
#       f'log10: {np.log10(np.sum(pty))}\n'
#       f'log2: {np.log2(np.sum(pty))}')

# flips ptychogram to match coordinates of the setup, since the camera saves it flipped
# flips left to right
pty = np.fliplr(pty)
# pty = np.flipud(pty)
# pty = np.rot90(pty, k=1, axes=(-2,-1))
# encoder[:,-1] *=-1
show3Dslider(np.log10(pty + 1))
# show3Dslider(np.log10(ptychogram + 1))
# Save data
with h5py.File(f'{result_data_path}/{filename}_pp{save_appendix}.hdf5', 'w') as hf:
    hf.create_dataset('ptychogram', data=ptychogram, dtype='f')
    hf.create_dataset('encoder', data=encoder, dtype='f')
    hf.create_dataset('binningFactor', data=(1,), dtype='i')
    hf.create_dataset('dxd', data=(dxd,), dtype='f')
    hf.create_dataset('Nd', data=(Nd,), dtype='i')
    hf.create_dataset('zo', data=(zo,), dtype='f')
    hf.create_dataset('entrancePupilDiameter', data=(beam_diameter,), dtype='f')
    hf.create_dataset('wavelength', data=(wavelength,), dtype='f')
    hf.create_dataset('orientation', data=(0,), dtype='i')
    hf.close()
print('file: ...' + save_appendix + ' saved')
