import collections
import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import shape, sum

from ..hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from .abstract_circular_distribution import AbstractCircularDistribution
from .circular_dirac_distribution import CircularDiracDistribution
from .circular_fourier_distribution import CircularFourierDistribution


class CircularMixture(AbstractCircularDistribution, HypertoroidalMixture):
    def __init__(
        self,
        dists: collections.abc.Sequence[AbstractCircularDistribution],
        w,
    ):
        """
        Creates a new circular mixture.

        Args:
            dists: The list of distributions.
            w: The weights of the distributions. They must have the same shape as 'dists'
                and the sum of all weights must be 1.
        """
        HypertoroidalMixture.__init__(self, dists, w)
        AbstractCircularDistribution.__init__(self)
        if not all(isinstance(cd, AbstractCircularDistribution) for cd in dists):
            raise TypeError(
                "All elements of 'dists' must be of type AbstractCircularDistribution."
            )

        if shape(dists) != shape(w):
            raise ValueError("'dists' and 'w' must have the same shape.")

        if all(isinstance(cd, CircularFourierDistribution) for cd in dists):
            warnings.warn(
                "Warning: Mixtures of Fourier distributions can be built by combining the Fourier coefficients so using a mixture may not be necessary"
            )
        elif all(isinstance(cd, CircularDiracDistribution) for cd in dists):
            warnings.warn(
                "Warning: Mixtures of WDDistributions can usually be combined into one WDDistribution."
            )

        self.dists = dists
        self.w = w / sum(w)
