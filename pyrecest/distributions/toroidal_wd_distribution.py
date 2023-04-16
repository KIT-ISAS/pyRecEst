import numpy as np

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_wd_distribution import HypertoroidalWDDistribution


class ToroidalWDDistribution(HypertoroidalWDDistribution, AbstractToroidalDistribution):
    def __init__(self, d, w=None):
        HypertoroidalWDDistribution.__init__(self, d, w)

    def integrate(self, left=None, right=None):
        left, right = self.prepare_integral_arguments(left, right)

        # TODO: Handle case where [l, r] spans more than 2*pi
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
