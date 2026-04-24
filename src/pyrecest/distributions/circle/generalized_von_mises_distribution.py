# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import arange, array, cos, exp, sum

from .abstract_circular_distribution import AbstractCircularDistribution


class GvMDistribution(AbstractCircularDistribution):
    """
    Generalized von Mises distribution of arbitrary order.

    See Riccardo Gatto, Sreenivasa Rao Jammalamadaka,
    "The Generalized von Mises Distribution",
    Statistical Methodology, 2007.

    Parameters
    ----------
    mu : array_like, shape (k,)
        Location parameters.
    kappa : array_like, shape (k,)
        Concentration parameters, all must be positive.
    """

    def __init__(self, mu, kappa):
        mu = array(mu, dtype=float)
        kappa = array(kappa, dtype=float)
        assert mu.ndim == 1, "mu must be a 1D array"
        assert mu.shape == kappa.shape, "mu and kappa must have the same shape"
        assert (kappa > 0).all(), "all kappa values must be positive"
        AbstractCircularDistribution.__init__(self)
        self.mu = mu
        self.kappa = kappa
        self._norm_const = None

    @property
    def norm_const(self):
        if self._norm_const is None:
            self._norm_const = self.integrate_fun_over_domain(
                lambda *args: self.pdf_unnormalized(array(args)), self.dim
            )
        return self._norm_const

    def pdf_unnormalized(self, xs):
        """
        Evaluate the unnormalized pdf at each point in xs.

        Parameters
        ----------
        xs : array_like, shape (n,)
            Points where the pdf is evaluated.

        Returns
        -------
        p : array, shape (n,)
            Unnormalized pdf values.
        """
        xs = array(xs)
        # j = [1, 2, ..., k], shape (k,)
        j = arange(1, self.mu.shape[0] + 1, dtype=float)
        # Broadcast: (k, 1) * ((1, n) - (k, 1)) → (k, n)
        arg = j[:, None] * (xs[None, :] - self.mu[:, None])
        return exp(sum(self.kappa[:, None] * cos(arg), axis=0))

    def pdf(self, xs):
        """
        Evaluate the pdf at each point in xs.

        Parameters
        ----------
        xs : array_like, shape (n,)
            Points where the pdf is evaluated.

        Returns
        -------
        p : array, shape (n,)
            Pdf values.
        """
        return self.pdf_unnormalized(xs) / self.norm_const
