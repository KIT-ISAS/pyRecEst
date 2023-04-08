import numpy as np
import scipy.special
from .abstract_circular_distribution import AbstractCircularDistribution

class VMDistribution(AbstractCircularDistribution):
    def __init__(self, mu, kappa, norm_const=None):
        assert kappa >= 0
        super().__init__()
        self.mu = mu
        self.kappa = kappa
        self.norm_const = norm_const

    def calculate_norm_const(self):
        self.norm_const = 2 * np.pi * scipy.special.iv(0, self.kappa)
        return self.norm_const

    def get_params(self):
        return self.mu, self.kappa

    def pdf(self, xs):
        if self.norm_const is None:
            self.calculate_norm_const()
        p = np.exp(self.kappa * np.cos(xs - self.mu)) / self.norm_const
        return p
