# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import column_stack, cos, sin, sqrt, sum

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


class ToroidalDiracDistribution(
    HypertoroidalDiracDistribution, AbstractToroidalDistribution
):
    def __init__(self, d, w):
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

        first_sines = sin(self.d[:, 0] - m[0])
        second_sines = sin(self.d[:, 1] - m[1])
        x = sum(self.w * first_sines * second_sines)
        y = sqrt(sum(self.w * first_sines**2) * sum(self.w * second_sines**2))
        rhoc = x / y
        return rhoc

    def covariance_4D(self):
        """
        Compute the 4D covariance matrix.

        :returns: 4D covariance matrix.
        """
        dbar = column_stack(
            [
                cos(self.d[:, 0]),
                sin(self.d[:, 0]),
                cos(self.d[:, 1]),
                sin(self.d[:, 1]),
            ]
        )
        mu = sum(self.w[:, None] * dbar, axis=0)
        centered = dbar - mu
        C = centered.T @ (self.w[:, None] * centered)
        return C
