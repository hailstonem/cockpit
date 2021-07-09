#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cockpit import depot
from cockpit import events
from cockpit.experiment import immediateMode
import cockpit.util.threads
import cockpit.interfaces.imager as imager
import cockpit.interfaces.stageMover

# import cockpit.util.user
import wx
import os
import time
import numpy as np
import imageio
import json
import cockpit.util.userConfig as Config

import tensorflow as tf

## Provided so the UI knows what to call this experiment.
EXPERIMENT_NAME = "MLCorrection"
CAMERA_TIMEOUT = 2.0


class MLCorrection(immediateMode.ImmediateModeExperiment):
    """Create BiasImageDatasetExperiment from parent class (the ImmediateModeExperiment)
    Potentially we should directly create an Experiment, for ActionTable efficiency reasons,
    but this should be a good starting point
    We are using a composite AO device for applying a sequence of aberrations"""

    def __init__(
        self,
        *args,
        bias_modes=(4, 5, 6, 7, 10),
        abb_magnitude=3,
        applied_modes=(4, 5, 6, 7, 10),
        numReps=1,
        initial_abb=None,
        model=10,
        repDuration=4,
        imagesPerRep=1,
        saveprefix="BIDE_",
        savePath="",
        **kwargs,
    ):
        areas = numReps
        self.bias_modes = bias_modes  # 0-indexed-Noll
        print(bias_modes)
        applied_steps = [-abb_magnitude, abb_magnitude]

        self.model = model
        self.iterations = 3

        self.abb_generator = self.generateAbb(
            bias_modes, applied_modes, applied_steps, areas
        )  # Not going to be thread safe
        try:
            self.saveBasePath = os.path.join(cockpit.util.userConfig.getValue("data-dir"), saveprefix)
        except:
            self.saveBasePath = os.path.join(os.path.expanduser("~"), saveprefix)

        self.numReps = len(applied_steps) * len(applied_modes) * areas

        # Set savepath to '' to prevent saving images. We will save our own images,
        # because setting filenames using Experiment/DataSaver look like it is
        # going to require a bunch of complicated subclassing
        super().__init__(self.numReps, repDuration, imagesPerRep, savePath="")

        # override cameraToImageCount so bias images are correctly saved to individual files
        # (necessary if using DataSaver)
        self.cameraToImageCount = 2 * len(bias_modes) + 1

        self.table = None  # Apparently we need this, even though we're not using it? suspect there is a problem with ImmediateModeExperiment
        self.time_start = time.time()

    def is_running(self):
        # HACK: Imager won't collect images if an experiment is running... Catch 22 here... So just breaking this for now
        return False

    # @cockpit.util.threads.callInMainThread
    def executeRep(self, repNum):
        t_elapsed = time.time() - self.time_start
        print(
            f"Started rep {repNum+1}/{self.numReps} Time Elapsed: {t_elapsed:.1f}s Time Remaining: {(numReps-repNum)*t_elapsed/repNum/60:.1f} min"
        )
        # Assume correct camera already active
        activeCams = depot.getActiveCameras()
        camera = activeCams[0]

        aodev = depot.getDeviceWithName("ao")  # IS THIS THE CORRECT DEVICE NAME?

        try:
            # offset = aodev.proxy.get_system_flat()  # assumes the correction for flat has already been done.
            # offset = Config.getValue('dm_sys_flat'
            offset = np.copy(aodev.actuator_offset)
        except:
            print("Failed to Get system flat")
            offset = None
        print(offset)

        biaslist, fprefix, newarea = self.abb_generator.__next__()

        imlist = []
        dm_set_failure = False

        for abb in biaslist:
            for i in iterations:
                try:
                    aodev.proxy.set_phase(abb, offset)
                except:
                    dm_set_failure = True

                # Do we need a pause here?
                time.sleep(0.1)
                # Collect image
                # takeimage = wx.GetApp().Imager.takeImage # this allows for blocking
                takeimage = depot.getHandlerWithName("dsp imager").takeImage
                result = events.executeAndWaitForOrTimeout(
                    events.NEW_IMAGE % camera.name,
                    takeimage,
                    camera.getExposureTime() / 1000 + CAMERA_TIMEOUT,
                    # shouldBlock=True,
                )
                model = ModelWrapper(self, self.model, quadratic_metric=False)
                if result is not None:
                    imlist.append(result[0])
                else:
                    raise TimeoutError("Image capture returned None")

        if dm_set_failure:
            print("Didn't set the aberration on the DM")
        # Save image - would be better to use existing DataSaver, but doesn't seem to allow custom naming scheme
        filename = f"{self.saveBasePath}{fprefix}.tif"

        imageio.mimwrite(filename, imlist, format="tif")

        # Get the current stage position; positions are in microns.
        curX, curY, curZ = cockpit.interfaces.stageMover.getPosition()

        # reset phase
        aodev.proxy.set_phase([0], offset)

        if self.numReps == repNum + 1:
            print("-----Experiment Complete-----")

    def makeBiasPolytope(self, start_aberrations, offset_axes, nk, steps=(1,)):
        """Return list of list of zernike amplitudes ('betas') for generating cross-polytope pattern of psfs
        """
        # beta (diffraction-limited), N_beta = cpsf.czern.nk
        beta = np.zeros(nk, dtype=np.float32)
        beta[:] = start_aberrations

        # add offsets to beta

        betas = []
        betas.append(tuple(beta))
        for axis in offset_axes:
            for step in steps:
                plus_offset = beta.copy()
                plus_offset[axis] += 1 * step
                betas.append(tuple(plus_offset))
            for step in steps:
                minus_offset = beta.copy()
                minus_offset[axis] -= 1 * step
                betas.append(tuple(minus_offset))

        return betas

    def generateAbb(self, bias_modes, applied_modes, applied_steps, areas):
        """Returns each list of bias aberrations for AO device to apply"""
        start_aberrations = np.zeros(np.max((np.max(bias_modes), (np.max(applied_modes)))) + 1)
        newarea = False
        for area in range(areas):
            if area:
                newarea = True

            for applied_abb in applied_modes:
                for step in applied_steps:
                    start_aberrations[applied_abb] = step
                    biaslist = self.makeBiasPolytope(start_aberrations, bias_modes, len(start_aberrations))
                    fprefix = f"R{areas}A{area}A{applied_abb}S{step:.1f}_"
                    yield biaslist, fprefix, newarea
                    newarea = False


class ModelWrapper:
    """Stores model specific parameters and applied preprocessing before prediction"""

    def __init__(self, model_no=1, quadratic_metric=False):
        self.model = None
        self.bias_magnitude = 1
        self.model, self.subtract, self.return_modes = self.load_model(model_no)
        print(f"Model {model_no} loaded: return modes: {self.return_modes}")
        self.bias_modes = [4, 5, 6, 7, 10]  ### Bias modes
        # override normal ml prediction and use equivalent conventional 2n+1 correction
        self.quadratic_metric = quadratic_metric
        if quadratic_metric:
            self.return_modes = self.bias_modes

    def load_model(self, model_no):
        print("loading model")
        with open("./models/model_config.json", "r") as modelfile:
            model_dict = json.load(modelfile)
            print(model_dict[str(model_no)])
        model_name = "./models/" + model_dict[str(model_no)][0] + "_savedmodel.h5"
        print("model_name")
        model = tf.keras.models.load_model(model_name, compile=False,)
        subtract = model_dict[str(model_no)][1] == "S"
        self.return_modes = [int(x) for x in model_dict[str(model_no)][2]]
        if len(self.return_modes) == 0:
            self.return_modes = [
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                15,
                16,
                21,
            ]
        self.bias_magnitude = [float(x) for x in model_dict[str(model_no)][3]]
        print(self.bias_magnitude)
        if self.bias_magnitude == []:
            self.bias_magnitude = 1
        return model, subtract, self.return_modes

    def predict(self, stack, rot90=False, split=False):
        def rotate(stack, rot90):
            return np.rot90(stack, k=rot90, axes=[1, 2])

        if rot90:
            stack = rotate(stack, rot90)
        stack = (stack.astype("float") - stack.mean()) / max(
            stack.astype("float").std(), 10e-20
        )  # prevent div/0

        if self.quadratic_metric:
            return self.single_shot_quadratic(
                stack, len(self.bias_modes), self.bias_magnitude, self.quadratic_metric
            )

        if self.subtract:
            stack = stack[:, :, :, 1:] - stack[:, :, :, 0:1]

        if split is False:
            pred = list(self.model.predict(stack)[0])
        else:
            pred = np.mean(
                [
                    self.model.predict(stack[:, 0 : stack.shape[1] * 3 // 4, 0 : stack.shape[2] * 3 // 4, :])[
                        0
                    ],
                    self.model.predict(stack[:, stack.shape[1] // 4 :, 0 : stack.shape[2] * 3 // 4, :])[0],
                    self.model.predict(stack[:, 0 : stack.shape[1] * 3 // 4, stack.shape[2] // 4 :, :])[0],
                    self.model.predict(stack[:, stack.shape[1] // 4 :, stack.shape[2] // 4 :, :])[0],
                ],
                axis=0,
                keepdims=False,
            )
        if len(pred) != len(self.return_modes):
            print(
                f"Warning: Mismatch in returned modes: predicted:{len(pred)}, expected: {len(self.return_modes)}"
            )
        return pred

    @staticmethod
    def single_shot_quadratic(image, num_bias, bias_mag, metric):
        """Returns quadratic fit estimate in same format as 2n+1 MLAO"""

        estimate = np.zeros(num_bias)
        for b in range(num_bias):
            b_indices = [0, 2 * b + 1, 2 * b + 2]

            if isinstance(bias_mag, list) and len(bias_mag) == 1:
                coeffarray = [0, bias_mag[0], -bias_mag[0]]
            elif isinstance(bias_mag, list) and len(bias_mag) > 1:
                raise NotImplementedError("fitting for multiple bias aberrations not implemented")
            else:
                coeffarray = [0, bias_mag, -bias_mag]

            metric_interface = MetricInterface(metric, image[0, :, :, b_indices[0]])
            intensities = [
                metric_interface.eval(image[0, :, :, b_indices[0]]),
                metric_interface.eval(image[0, :, :, b_indices[1]]),
                metric_interface.eval(image[0, :, :, b_indices[2]]),
            ]
            estimate[b] = optimisation(coeffarray, intensities)
        print(f"SSQ{estimate}")
        return -estimate


EXPERIMENT_CLASS = MLCorrection  # Don't know what the point of this is but is required by GUI
