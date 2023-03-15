import numpy as np
from scipy.integrate import quad
from abc import ABC, abstractmethod
from AbstractHypertoroidalDistribution import AbstractHypertoroidalDistribution

class AbstractCircularDistribution(AbstractHypertoroidalDistribution):
    def __init__(self):
        super().__init__()
        self.dim = 1

    def cdf_numerical(self, xa, starting_point=0):
        xa = np.asarray(xa)
        assert xa.ndim == 1, "xa must be a 1D array"

        def cdf_single(x):
            starting_point_mod = starting_point % (2 * np.pi)
            x_mod = x % (2 * np.pi)

            if x_mod < starting_point_mod:
                return 1 - self.integral_numerical(x_mod, starting_point_mod)
            else:
                return self.integral_numerical(starting_point_mod, x_mod)

        return np.array([cdf_single(x) for x in xa])

    def sample_metropolis_hastings(self, n, proposal=None, start_point=None, burn_in=10, skipping=5):
        if proposal is None:
            wn = WNDistribution.from_moment(self.trigonometric_moment(1))
            wn.mu = 0
            proposal = lambda x: (x + wn.sample(1)) % (2 * np.pi)

        if start_point is None:
            start_point = self.mean_direction()

        s = super().sample_metropolis_hastings(n, proposal, start_point, burn_in, skipping)
        return s
