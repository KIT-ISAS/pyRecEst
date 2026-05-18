# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, exp, sin

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)


class ToroidalWrappedNormalDistribution(
    HypertoroidalWrappedNormalDistribution, AbstractToroidalDistribution
):
    """Bivariate wrapped normal distribution on the torus.

    References
    ----------
    Kurz, G., Gilitschenski, I., Dolgov, M., & Hanebeck, U. D. (2014).
    Bivariate Angular Estimation Under Consideration of Dependencies Using
    Directional Statistics. Proceedings of the 53rd IEEE Conference on
    Decision and Control.
    """

    def mean_4D(self):
        """
        Compute the 4D mean of the distribution.

        Returns:
            array: The 4D mean.
        """
        mu = array(
            [
                cos(self.mu[0]) * exp(-self.C[0, 0] / 2),
                sin(self.mu[0]) * exp(-self.C[0, 0] / 2),
                cos(self.mu[1]) * exp(-self.C[1, 1] / 2),
                sin(self.mu[1]) * exp(-self.C[1, 1] / 2),
            ]
        )
        return mu

    def covariance_4D(self):
        """
        Compute the 4D covariance of the distribution.

        Returns:
            array: The 4D covariance.
        """
        mu0 = self.mu[0]
        mu1 = self.mu[1]
        c00 = self.C[0, 0]
        c01 = self.C[0, 1]
        c11 = self.C[1, 1]
        common_scale = exp(-c00 / 2 - c11 / 2)

        # jscpd:ignore-start
        C00 = 1 / 2 * (1 - exp(-c00)) * (1 - exp(-c00) * cos(2 * mu0))
        C01 = -1 / 2 * (1 - exp(-c00)) * exp(-c00) * sin(2 * mu0)
        C11 = 1 / 2 * (1 - exp(-c00)) * (1 + exp(-c00) * cos(2 * mu0))

        C22 = 1 / 2 * (1 - exp(-c11)) * (1 - exp(-c11) * cos(2 * mu1))
        C23 = -1 / 2 * (1 - exp(-c11)) * exp(-c11) * sin(2 * mu1)
        C33 = 1 / 2 * (1 - exp(-c11)) * (1 + exp(-c11) * cos(2 * mu1))

        C02 = (
            1
            / 2
            * common_scale
            * (
                exp(-c01) * cos(mu0 + mu1)
                + exp(c01) * cos(mu0 - mu1)
                - 2 * cos(mu0) * cos(mu1)
            )
        )
        C03 = (
            1
            / 2
            * common_scale
            * (
                exp(-c01) * sin(mu0 + mu1)
                - exp(c01) * sin(mu0 - mu1)
                - 2 * cos(mu0) * sin(mu1)
            )
        )

        C12 = (
            1
            / 2
            * common_scale
            * (
                exp(-c01) * sin(mu0 + mu1)
                + exp(c01) * sin(mu0 - mu1)
                - 2 * sin(mu0) * cos(mu1)
            )
        )
        C13 = (
            1
            / 2
            * common_scale
            * (
                exp(c01) * cos(mu0 - mu1)
                - exp(-c01) * cos(mu0 + mu1)
                - 2 * sin(mu0) * sin(mu1)
            )
        )

        C = array(
            [
                [C00, C01, C02, C03],
                [C01, C11, C12, C13],
                [C02, C12, C22, C23],
                [C03, C13, C23, C33],
            ]
        )
        # jscpd:ignore-end
        return C
