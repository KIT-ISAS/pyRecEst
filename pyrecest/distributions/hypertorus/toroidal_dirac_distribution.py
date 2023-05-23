import numpy as np
from typing import Optional

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


class ToroidalDiracDistribution(
    HypertoroidalDiracDistribution, AbstractToroidalDistribution
):
    def __init__(self, d: np.ndarray, w: Optional[np.ndarray] = None):
        """
        Initialize ToroidalDiracDistribution.

        :param d: Array of positions.
        :param w: Array of weights.
        """
        AbstractToroidalDistribution.__init__(self)
        HypertoroidalDiracDistribution.__init__(self, d, w)

    def circular_correlation_jammalamadaka(self) -> float:
        """
        Calculate the circular correlation according to Jammalamadaka's definition

        :returns: Correlation coefficient.
        """
        m = self.mean_direction()

        x = np.sum(self.w * np.sin(self.d[0, :] - m[0]) * np.sin(self.d[1, :] - m[1]))
        y = np.sqrt(
            np.sum(self.w * np.sin(self.d[0, :] - m[0]) ** 2)
            * np.sum(self.w * np.sin(self.d[1, :] - m[1]) ** 2)
        )
        rhoc = x / y
        return rhoc

    def covariance_4D(self) -> np.ndarray:
        """
        Compute the 4D covariance matrix.

        :returns: 4D covariance matrix.
        """
        dbar = np.column_stack(
            [
                np.cos(self.d[0, :]),
                np.sin(self.d[0, :]),
                np.cos(self.d[1, :]),
                np.sin(self.d[1, :]),
            ]
        )
        mu = np.dot(self.w, dbar)
        n = len(self.d)
        C = (dbar - np.tile(mu, (n, 1))).T @ (
            np.diag(self.w) @ (dbar - np.tile(mu, (n, 1)))
        )
        return C
