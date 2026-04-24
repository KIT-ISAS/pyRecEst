# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import all, cos, exp, mod, pi

from .abstract_toroidal_distribution import AbstractToroidalDistribution


class AbstractToroidalBivarVMDistribution(AbstractToroidalDistribution):
    """Abstract base for bivariate von Mises distributions on the torus.

    Subclasses share the same ``pdf`` structure:

        C * exp(kappa1*cos(x1 - mu1) + kappa2*cos(x2 - mu2) + coupling_term)

    and must implement :meth:`_coupling_term`.
    """

    def __init__(self, mu, kappa):
        AbstractToroidalDistribution.__init__(self)
        assert mu.shape == (2,)
        assert kappa.shape == (2,)
        assert all(kappa >= 0.0)
        self.mu = mod(mu, 2.0 * pi)
        self.kappa = kappa

    def _coupling_term(self, xs):
        """Return the distribution-specific coupling term for ``pdf``."""
        raise NotImplementedError

    def pdf(self, xs):
        assert xs.shape[-1] == 2
        return self.C * exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self._coupling_term(xs)
        )
