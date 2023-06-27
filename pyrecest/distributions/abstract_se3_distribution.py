import time
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import quaternion

from .cart_prod.abstract_lin_bounded_cart_prod_distribution import (
    AbstractLinBoundedCartProdDistribution,
)


class AbstractSE3Distribution(AbstractLinBoundedCartProdDistribution):
    def __init__(self):
        AbstractLinBoundedCartProdDistribution.__init__(self, 3, 3)  # 3-sphere and R^3

    @property
    def input_dim(self):
        return self.dim + 1

    @abstractmethod
    def mode(self):
        pass

    def plot_mode(self):
        mode = self.mode()
        h = AbstractSE3Distribution.plot_point(mode)
        return h

    def plot_state(
        self,
        orientationSamples: int | np.int32 | np.int64 = 10,
        showMarginalized: bool = True,
    ):
        samples = self.sample(orientationSamples)
        mode = self.mode()
        h = []
        if showMarginalized:
            h.append(self.marginalize_periodic().plot_state())
        for i in range(samples.shape[1]):
            if showMarginalized:
                linearPart = mode[4:]
            else:
                linearPart = samples[4:, i]
            h.append(
                AbstractSE3Distribution.plot_point(
                    np.concatenate((samples[:4, i], linearPart), axis=0)
                )
            )
        return h

    @staticmethod
    def plot_point(se3point): # pylint: disable=too-many-locals
        """Visualize just a point in the SE(3) domain (no uncertainties are considered)"""
        q = np.quaternion(*se3point[:4])
        rotMat = quaternion.as_rotation_matrix(q)

        pos = se3point[4:]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        h1 = ax.quiver(
            pos[0], pos[1], pos[2], rotMat[0, 0], rotMat[0, 1], rotMat[0, 2], color="r"
        )
        h2 = ax.quiver(
            pos[0], pos[1], pos[2], rotMat[1, 0], rotMat[1, 1], rotMat[1, 2], color="g"
        )
        h3 = ax.quiver(
            pos[0], pos[1], pos[2], rotMat[2, 0], rotMat[2, 1], rotMat[2, 2], color="b"
        )
        h = [h1, h2, h3]
        relevant_coords = np.concatenate((pos.reshape(-1, 1), pos + rotMat), axis=1)
        needed_boundaries = np.column_stack(
            (np.min(relevant_coords, axis=1), np.max(relevant_coords, axis=1))
        )

        # Get current axis limits
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        zl = ax.get_zlim()

        # Calculate optimal axis
        optAx = (pos // 5) * 5

        # Adjust axis limits if necessary
        if xl[0] < needed_boundaries[0, 0] or xl[1] > needed_boundaries[0, 1]:
            ax.set_xlim([optAx[0] - 5, optAx[0] + 5])
        if yl[0] < needed_boundaries[1, 0] or yl[1] > needed_boundaries[1, 1]:
            ax.set_ylim([optAx[1] - 5, optAx[1] + 5])
        if zl[0] < needed_boundaries[2, 0] or zl[1] > needed_boundaries[2, 1]:
            ax.set_zlim([optAx[2] - 5, optAx[2] + 5])

        return h

    @staticmethod
    def plot_trajectory(periodicStates, linStates, animate=False, delay=0.05):
        assert periodicStates.shape[1] == linStates.shape[1]
        h = []
        for i in range(periodicStates.shape[1]):
            h.append(
                AbstractSE3Distribution.plot_point(
                    np.concatenate((periodicStates[:, i], linStates[:, i]), axis=0)
                )
            )
            if animate:
                time.sleep(delay)
        return h

    def get_manifold_size(self):
        return np.inf
