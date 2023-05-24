import numpy as np
from numpy import cos, exp, sin

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)


class ToroidalWrappedNormalDistribution(
    HypertoroidalWrappedNormalDistribution, AbstractToroidalDistribution
):
    """
    Toroidal Wrapped Normal Distribution.
    """

    def mean_4D(self) -> np.array:
        """
        Compute the 4D mean of the distribution.

        Returns:
            np.array: The 4D mean.
        """
        s = self.mu
        mu = np.array(
            [
                cos(s[0, :]) * exp(-self.C[0, 0] / 2),
                sin(s[0, :]) * exp(-self.C[0, 0] / 2),
                cos(s[1, :]) * exp(-self.C[1, 1] / 2),
                sin(s[1, :]) * exp(-self.C[1, 1] / 2),
            ]
        )
        return mu

    def covariance_4D(self) -> np.array:
        """
        Compute the 4D covariance of the distribution.

        Returns:
            np.array: The 4D covariance.
        """
        C = np.zeros((4, 4))
        # jscpd:ignore-start
        C[0, 0] = (
            1
            / 2
            * (1 - exp(-self.C[0, 0]))
            * (1 - exp(-self.C[0, 0]) * cos(2 * self.mu[0]))
        )
        C[0, 1] = (
            -1 / 2 * (1 - exp(-self.C[0, 0])) * exp(-self.C[0, 0]) * sin(2 * self.mu[0])
        )
        C[1, 0] = C[0, 1]
        C[1, 1] = (
            1
            / 2
            * (1 - exp(-self.C[0, 0]))
            * (1 + exp(-self.C[0, 0]) * cos(2 * self.mu[0]))
        )

        C[2, 2] = (
            1
            / 2
            * (1 - exp(-self.C[1, 1]))
            * (1 - exp(-self.C[1, 1]) * cos(2 * self.mu[1]))
        )
        C[2, 3] = (
            -1 / 2 * (1 - exp(-self.C[1, 1])) * exp(-self.C[1, 1]) * sin(2 * self.mu[1])
        )
        C[3, 2] = C[2, 3]
        C[3, 3] = (
            1
            / 2
            * (1 - exp(-self.C[1, 1]))
            * (1 + exp(-self.C[1, 1]) * cos(2 * self.mu[1]))
        )

        C[0, 2] = (
            1
            / 2
            * exp(-self.C[0, 0] / 2 - self.C[1, 1] / 2)
            * (
                exp(-self.C[0, 1]) * cos(self.mu[0] + self.mu[1])
                + exp(self.C[0, 1]) * cos(self.mu[0] - self.mu[1])
                - 2 * cos(self.mu[0]) * cos(self.mu[1])
            )
        )
        C[0, 3] = (
            1
            / 2
            * exp(-self.C[0, 0] / 2 - self.C[1, 1] / 2)
            * (
                exp(-self.C[0, 1]) * sin(self.mu[0] + self.mu[1])
                - exp(self.C[0, 1]) * sin(self.mu[0] - self.mu[1])
                - 2 * cos(self.mu[0]) * sin(self.mu[1])
            )
        )

        C[1, 2] = (
            1
            / 2
            * exp(-self.C[0, 0] / 2 - self.C[1, 1] / 2)
            * (
                exp(-self.C[0, 1]) * sin(self.mu[0] + self.mu[1])
                - exp(self.C[0, 1]) * sin(self.mu[0] - self.mu[1])
                + 2 * sin(self.mu[0]) * cos(self.mu[1])
            )
        )
        C[1, 3] = (
            1
            / 2
            * exp(-self.C[0, 0] / 2 - self.C[1, 1] / 2)
            * (
                exp(-self.C[0, 1]) * cos(self.mu[0] + self.mu[1])
                + exp(self.C[0, 1]) * cos(self.mu[0] - self.mu[1])
                - 2 * sin(self.mu[0]) * sin(self.mu[1])
            )
        )

        C[2, 0] = C[0, 2]
        C[3, 0] = C[0, 3]
        C[2, 1] = C[1, 2]
        C[3, 1] = C[1, 3]
        # jscpd:ignore-end
        return C
