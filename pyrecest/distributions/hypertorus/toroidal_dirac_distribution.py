from pyrecest.backend import tile
from pyrecest.backend import sum
from pyrecest.backend import sqrt
from pyrecest.backend import sin
from pyrecest.backend import dot
from pyrecest.backend import cos
import numpy as np

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


class ToroidalDiracDistribution(
    HypertoroidalDiracDistribution, AbstractToroidalDistribution
):
    def __init__(self, d: np.ndarray, w: np.ndarray | None = None):
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

        x = sum(self.w * sin(self.d[0, :] - m[0]) * sin(self.d[1, :] - m[1]))
        y = sqrt(
            sum(self.w * sin(self.d[0, :] - m[0]) ** 2)
            * sum(self.w * sin(self.d[1, :] - m[1]) ** 2)
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
                cos(self.d[0, :]),
                sin(self.d[0, :]),
                cos(self.d[1, :]),
                sin(self.d[1, :]),
            ]
        )
        mu = dot(self.w, dbar)
        n = len(self.d)
        C = (dbar - tile(mu, (n, 1))).T @ (
            np.diag(self.w) @ (dbar - tile(mu, (n, 1)))
        )
        return C