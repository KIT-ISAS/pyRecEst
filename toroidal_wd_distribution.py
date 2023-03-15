import numpy as np
from hypertoroidal_wd_distribution import HypertoroidalWDDistribution
from abstract_toroidal_distribution import AbstractToroidalDistribution

class ToroidalWDDistribution(HypertoroidalWDDistribution, AbstractToroidalDistribution):
    def __init__(self, d, w=None):
        if w is None:
            w = np.ones(d.shape[1]) / d.shape[1]

        super().__init__(d, w)

    def integral(self, l=None, r=None):
        if l is None:
            l = np.array([0, 0])

        if r is None:
            r = np.array([2 * np.pi, 2 * np.pi])

        assert l.shape == (self.dim, )
        assert r.shape == (self.dim, )

        # TODO: Handle case where [l, r] spans more than 2*pi
        assert 0 <= l[0] <= 2 * np.pi
        assert 0 <= r[0] <= 2 * np.pi
        assert 0 <= l[1] <= 2 * np.pi
        assert 0 <= r[1] <= 2 * np.pi

        if r[0] < l[0]:
            result = -self.integral(np.array([r[0], l[1]]), np.array([l[0], r[1]]))  # swap l1 and r1
        elif r[1] < l[1]:
            result = -self.integral(np.array([l[0], r[1]]), np.array([r[0], l[1]]))  # swap l2 and r2
        else:
            # now we can guarantee l1 <= r1 and l2 <= r2
            result = np.sum(self.w[(self.d[0, :] >= l[0]) & (self.d[0, :] < r[0]) & (self.d[1, :] >= l[1]) & (self.d[1, :] < r[1])])

        return result
