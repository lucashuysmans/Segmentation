#!/usr/bin/env python3

# We want to implement
# - the data_fitting term without penalty
# - gradient descent routine with clipping and c1/c2-updating

import numpy as np
import matplotlib.pylot as plt
import ChanVese as cv
import torch


class DeepSegmentation:
    def __init__(
        self, image, regulariser, u_init=None, segmentation_threshold=0.5, c=None
    ):
        self._image_arr = torch.Tensor(np.array(image, dtype=float) / 255)
        self.image_shape = self._image_arr.shape
        self.channels = len(image.getbands())
        if self.channels > 1:
            self.image_shape = self.image_shape[:-1]
        self._dim = len(self.image_shape)
        self.segmentation_threshold = segmentation_threshold
        self.c = (0, 1) if c is None else c

        self.regulariser = regulariser

        if u_init is None:
            self.u = torch.rand(size=self.image_shape)
        else:
            self.u = torch.Tensor(u_init)
            self.c = cv.get_segmentation_mean_colours(
                self.u, self.image_arr, self.segmentation_threshold
            )

    def show_segmentation(self):
        """Plots and shows the image with its segmentation contour superimposed."""
        plt.imshow(self._image_arr.numpy(), cmap="gray", vmin=0, vmax=1)
        plt.contour(
            np.clip(self.u.numpy(), self.segmentation_threshold, 1), [0], colors="red"
        )
        plt.show()

    def update_c(self):
        """Update the average colours in the segmentation domain and its complement. See
        'get_segmentation_mean_colours' for more information.
        """
        self.c = cv.get_segmentation_mean_colours(
            self.u, self._image_arr, self.segmentation_threshold
        )

    def single_step(self, lmb_reg=1, epsilon=0.1):
        self.u.requires_grad = True
        data_fitting = cv.CEN_data_fitting_energy(
            self.u, self.c[0], self.c[1], self._image_arr
        )
        error = data_fitting + lmb_reg * self.regulariser(u.unsqueeze(0).unsqueeze(0))
        gradients = torch.autograd.grad(error, self.u)[0]
        self.u = (self.u - epsilon * gradients).detach()