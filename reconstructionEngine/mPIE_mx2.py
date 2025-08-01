import numpy as np
from matplotlib import pyplot as plt

try:
    import cupy as cp
except ImportError:
    print("Cupy not available, will not be able to run GPU based computation")
    # Still define the name, we'll take care of it later but in this way it's still possible
    # to see that gPIE exists for example.
    cp = None

# PtyLab imports
from PtyLab.Reconstruction.Reconstruction import Reconstruction
from PtyLab.Engines.BaseEngine import BaseEngine
from PtyLab.ExperimentalData.ExperimentalData import ExperimentalData
from PtyLab.Params.Params import Params
from PtyLab.utils.gpuUtils import getArrayModule, asNumpyArray
from PtyLab.Monitor.Monitor import Monitor
from PtyLab.utils.utils import fft2c, ifft2c
import logging
import tqdm
import sys


class mPIE_mx2(BaseEngine):
    def __init__(
        self,
        reconstruction: Reconstruction,
        experimentalData: ExperimentalData,
        params: Params,
        monitor: Monitor,
    ):
        # This contains reconstruction parameters that are specific to the reconstruction
        # but not necessarily to ePIE reconstruction
        super().__init__(reconstruction, experimentalData, params, monitor)
        self.logger = logging.getLogger("mPIE")
        self.logger.info("Sucesfully created mPIE mPIE_engine")
        self.logger.info("Wavelength attribute: %s", self.reconstruction.wavelength)
        # initialize mPIE Params
        self.initializeReconstructionParams()
        self.params.momentumAcceleration = True
        self.name = "mPIE"

    @property
    def keepPatches(self):
        """Wether or not to keep track of the individual object update patches.

        This strongly increases the amount of memory required, only use when absolutely required.

        """
        return hasattr(self, "patches")

    @keepPatches.setter
    def keepPatches(self, keep_them):

        if keep_them:
            self.logger.info("Keeping patches!")
            self.patches = np.zeros(
                (
                    self.experimentalData.ptychogram.shape[0],
                    *self.reconstruction.shape_O,
                ),
                np.complex64,
            )
        else:
            self.logger.info("Not keeping patches")
            if hasattr(self, "patches"):
                del self.patches

    def initializeReconstructionParams(self):
        """
        Set parameters that are specific to the mPIE settings.
        :return:
        """
        # self.eswUpdate = self.reconstruction.esw.copy()
        self.betaProbe = 0.25
        self.betaObject = 0.25
        self.alphaProbe = 0.1  # probe regularization
        self.alphaObject = 0.1  # object regularization
        self.feedbackM = 0.3  # feedback
        self.frictionM = 0.7  # friction
        self.numIterations = 50

        # initialize momentum
        self.reconstruction.initializeObjectMomentum()
        self.reconstruction.initializeProbeMomentum()
        # set object and probe buffers
        # self.reconstruction.object = self.reconstruction.object[[0],...] # remove all other wavelength dimensions
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()
        self.reconstruction.probeWindow = np.abs(self.reconstruction.probe)

    def reconstruct(self, shift_row, shift_col, experimentalData=None, reconstruction=None, ):
        """Reconstruct object. If experimentalData is given, it replaces the current data. Idem for reconstruction."""

        self.changeExperimentalData(experimentalData)
        self.changeOptimizable(reconstruction)

        self._prepareReconstruction()
        # set object and probe buffers, in case object and probe are changed in the _prepareReconstruction() step
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()
        objectPatches = self.reconstruction.probe * 0.0
        objectPatches = objectPatches[:, :, [0], ...]

        # actual reconstruction MPIE_engine
        self.pbar = tqdm.trange(
            self.numIterations, desc="mPIE MX2", file=sys.stdout, leave=True
        )
        for loop in self.pbar:
            # set position order
            self.setPositionOrder()
            self.reconstruction.esw = np.zeros_like(self.reconstruction.probe)

            # print(f'object_patches_shape {objectPatches.shape}')

            self.pbar_pos = tqdm.tqdm(self.positionIndices, leave=False, desc='ptychogram', file=sys.stdout)
            for positionLoop, positionIndex in enumerate(self.pbar_pos):
                # print(f'Loop: {positionLoop}')
                # print(f'index: {positionIndex}')
                # get object patch, stored as self.probe
                # self.reconstruction.make_probe(positionIndex)

                # row, col = self.reconstruction.positions[positionIndex]
                # sy = slice(row, row + self.reconstruction.Np)
                # sx = slice(col, col + self.reconstruction.Np)
                # # note that object patch has size of probe array
                # objectPatch = self.reconstruction.object[..., sy, sx].copy()
                #
                # # make exit surface wave
                # self.reconstruction.esw = objectPatch * self.reconstruction.probe

                # get object patch, shifted for each probe mode
                for i in range(self.reconstruction.probe.shape[0]):
                    row, col = self.reconstruction.positions[positionIndex]
                    row += shift_row[i]
                    col += shift_col[i]
                    sy = slice(row, row + self.reconstruction.Np)
                    sx = slice(col, col + self.reconstruction.Np)
                    # note that object patch has size of probe array
                    temp = self.reconstruction.object[..., sy, sx].copy()
                    # print(f'temp_shape: {temp.shape}')
                    # print(temp.shape)
                    objectPatches[i] = temp[0]  # .reshape(512, 512)
                    # objectPatches[i, 0, 1, 0, ...] = temp  # .reshape(512, 512)
                    # make exit surface wave
                    self.reconstruction.esw[i] = objectPatches[i] * self.reconstruction.probe[i]

                # propagate to camera, intensityProjection, propagate back to object
                self.intensityProjection(positionIndex)

                # difference term
                DELTA = self.reconstruction.eswUpdate - self.reconstruction.esw
                # self.viewer.layers['update'].data[positionIndex] = abs(DELTA ** 2).get()
                # import pyqtgraph as pg
                # pg.QtGui.QGuiApplication.processEvents()

                # object update
                if self.params.objectTVregSwitch and loop % self.params.objectTVfreq == 0:
                    object_patches = self.objectPatchUpdate_TV(objectPatches, DELTA)
                else:
                    object_patches = self.objectPatchUpdate(objectPatches, DELTA)
                # print(f'object_patches_shape {objectPatches.shape}')

                # object update shifting back the corresponding position
                # npsm_range = np.arange(self.reconstruction.npsm)
                # np.random.shuffle(npsm_range)
                for i in range(self.reconstruction.probe.shape[0]):
                    row, col = self.reconstruction.positions[positionIndex]
                    row += shift_row[i]
                    col += shift_col[i]
                    sy = slice(row, row + self.reconstruction.Np)
                    sx = slice(col, col + self.reconstruction.Np)
                    self.reconstruction.object[..., sy, sx] = object_patches[i]#,0,0,0,...]

                # probe update
                weight = 1
                if self.params.weigh_probe_updates_by_intensity:
                    weight = self.experimentalData.relative_intensity(positionIndex)
                    # print(f'for position {positionIndex}, using weight {weight}')

                self.reconstruction.probe = self.probeUpdate(objectPatches, DELTA, weight)
                # self.reconstruction.push_probe_update(self.reconstruction.probe, positionIndex, self.experimentalData.ptychogram.shape[0])

                # if self.params.positionCorrectionSwitch:
                #     shifter = self.positionCorrection(objectPatches, positionIndex, sy, sx)
                    #self.pbar_pos.write(f'Corr: {shifter[0]*1e6:.2f} um x {shifter[1]*1e6:.2f} um')

                # momentum updates
                if np.random.rand(1) > 0.95:
                    self.objectMomentumUpdate()
                    self.probeMomentumUpdate()
                # yield positionLoop, positionIndex

                # show reconstruction
                # self.showReconstruction(0)

            # get error metric
            self.getErrorMetrics()
            # yield 1,1

            # apply Constraints
            self.applyConstraints(loop)
            # yield 1, 1
            print(f'object_shape: {self.reconstruction.object.shape}')
            # show reconstruction
            self.showReconstruction(loop)

        if self.params.gpuFlag:
            self.logger.info("switch to cpu")
            self._move_data_to_cpu()
            self.params.gpuFlag = 0

            # todo clearMemory implementation

    def objectMomentumUpdate(self):
        """
        momentum update object, save updated objectMomentum and objectBuffer.
        :return:
        """
        gradient = self.reconstruction.objectBuffer - self.reconstruction.object
        self.reconstruction.objectMomentum = (
            gradient + self.frictionM * self.reconstruction.objectMomentum
        )
        self.reconstruction.object = (
            self.reconstruction.object
            - self.feedbackM * self.reconstruction.objectMomentum
        )
        self.reconstruction.objectBuffer = self.reconstruction.object.copy()

    def probeMomentumUpdate(self):
        """
        momentum update probe, save updated probeMomentum and probeBuffer.
        :return:
        """
        gradient = self.reconstruction.probeBuffer - self.reconstruction.probe
        self.reconstruction.probeMomentum = (
            gradient + self.frictionM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probe = (
            self.reconstruction.probe
            - self.feedbackM * self.reconstruction.probeMomentum
        )
        self.reconstruction.probeBuffer = self.reconstruction.probe.copy()

    def objectPatchUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absP2 = xp.abs(self.reconstruction.probe) ** 2
        Pmax = xp.max(xp.sum(absP2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        if self.experimentalData.operationMode == "FPM":
            frac = (
                abs(self.reconstruction.probe)
                / Pmax
                * self.reconstruction.probe.conj()
                / (self.alphaObject * Pmax + (1 - self.alphaObject) * absP2)
            )
        else:
            frac = self.reconstruction.probe.conj() / (
                self.alphaObject * Pmax + (1 - self.alphaObject) * absP2
            )

        return objectPatch + self.betaObject * xp.sum(
            frac * DELTA, axis=2, keepdims=True
        )

    def probeUpdate(self, objectPatch: np.ndarray, DELTA: np.ndarray, weight: float):
        """
        Todo add docstring
        :param objectPatch:
        :param DELTA:
        :return:
        """
        # find out which array module to use, numpy or cupy (or other...)
        xp = getArrayModule(objectPatch)
        absO2 = xp.abs(objectPatch) ** 2
        Omax = xp.max(xp.sum(absO2, axis=(0, 1, 2, 3)), axis=(-1, -2))
        frac = objectPatch.conj() / (
            self.alphaProbe * Omax + (1 - self.alphaProbe) * absO2
        )
        r = self.reconstruction.probe + weight * self.betaProbe * xp.sum(
            frac * DELTA, axis=1, keepdims=True
        )
        return r
