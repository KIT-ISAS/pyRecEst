import numpy as np

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


class ToroidalDiracDistribution(
    HypertoroidalDiracDistribution, AbstractToroidalDistribution
):
    def __init__(self, d, w=None):
        AbstractToroidalDistribution.__init__(self)
        HypertoroidalDiracDistribution.__init__(self, d, w)

    def integrate(self, left=None, right=None):
        left, right = self.prepare_integral_arguments(left, right)

        assert 0 <= left[0] <= 2 * np.pi
        assert 0 <= right[0] <= 2 * np.pi
        assert 0 <= left[1] <= 2 * np.pi
        assert 0 <= right[1] <= 2 * np.pi

        if right[0] < left[0]:
            # swap l1 and r1
            result = -self.integrate(
                np.array([right[0], left[1]]), np.array([left[0], right[1]])
            )
        elif right[1] < left[1]:
            # swap l2 and r2
            result = -self.integrate(
                np.array([left[0], right[1]]), np.array([right[0], left[1]])
            )
        else:
            # now we can guarantee l1 <= r1 and l2 <= r2
            result = np.sum(
                self.w[
                    (self.d[0, :] >= left[0])
                    & (self.d[0, :] < right[0])
                    & (self.d[1, :] >= left[1])
                    & (self.d[1, :] < right[1])
                ]
            )

        return result

    def circular_correlation_jammalamadaka(self):
        m = self.mean_direction()

        x = np.sum(self.w * np.sin(self.d[0, :] - m[0]) * np.sin(self.d[1, :] - m[1]))
        y = np.sqrt(
            np.sum(self.w * np.sin(self.d[0, :] - m[0]) ** 2)
            * np.sum(self.w * np.sin(self.d[1, :] - m[1]) ** 2)
        )
        rhoc = x / y
        return rhoc

    def covariance_4D(self):
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
