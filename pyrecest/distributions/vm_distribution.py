import numpy as np
from scipy.optimize import fsolve
from scipy.special import iv

from .abstract_circular_distribution import AbstractCircularDistribution


class VMDistribution(AbstractCircularDistribution):
    def __init__(self, mu, kappa, norm_const=None):
        assert kappa >= 0
        super().__init__()
        self.mu = mu
        self.kappa = kappa
        self.norm_const = norm_const

    def calculate_norm_const(self):
        self.norm_const = 2 * np.pi * iv(0, self.kappa)
        return self.norm_const

    def get_params(self):
        return self.mu, self.kappa

    def pdf(self, xs):
        if self.norm_const is None:
            self.calculate_norm_const()
        p = np.exp(self.kappa * np.cos(xs - self.mu)) / self.norm_const
        return p

    @staticmethod
    def besselratio(nu, kappa):
        return iv(nu + 1, kappa) / iv(nu, kappa)

    @staticmethod
    def besselratio_inverse(v, x):
        def f(t):
            return VMDistribution.besselratio(v, t) - x

        start = 1
        (kappa,) = fsolve(f, start, xtol=1e-40)
        return kappa

    def multiply(self, vm2):
        C = self.kappa * np.cos(self.mu) + vm2.kappa * np.cos(vm2.mu)
        S = self.kappa * np.sin(self.mu) + vm2.kappa * np.sin(vm2.mu)
        mu_ = np.mod(np.arctan2(S, C), 2 * np.pi)
        kappa_ = np.sqrt(C**2 + S**2)
        return VMDistribution(mu_, kappa_)

    def convolve(self, vm2):
        mu_ = np.mod(self.mu + vm2.mu, 2 * np.pi)
        t = VMDistribution.besselratio(0, self.kappa) * VMDistribution.besselratio(
            0, vm2.kappa
        )
        kappa_ = VMDistribution.besselratio_inverse(0, t)
        return VMDistribution(mu_, kappa_)

    def entropy(self):
        result = -self.kappa * VMDistribution.besselratio(0, self.kappa) + np.log(
            2 * np.pi * iv(0, self.kappa)
        )
        return result

    @staticmethod
    def from_moment(m):
        """
        Obtain a VM distribution from a given first trigonometric moment.

        Parameters:
            m (scalar): First trigonometric moment (complex number).

        Returns:
            vm (VMDistribution): VM distribution obtained by moment matching.
        """
        mu_ = np.mod(np.arctan2(np.imag(m), np.real(m)), 2 * np.pi)
        kappa_ = VMDistribution.besselratio_inverse(0, np.abs(m))
        vm = VMDistribution(mu_, kappa_)
        return vm
