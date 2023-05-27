import warnings
from typing import List

import numpy as np
from abstract_circular_distribution import AbstractCircularDistribution

from ..hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from .circular_dirac_distribution import CircularDiracDistribution
from .circular_fourier_distribution import CircularFourierDistribution
from beartype import beartype

class CircularMixture(AbstractCircularDistribution, HypertoroidalMixture):
    @beartype
    def __init__(self, dists: List[AbstractCircularDistribution], w: np.ndarray):
        """
        Creates a new circular mixture.

        Args:
            dists: The list of distributions.
            w: The weights of the distributions. They must have the same shape as 'dists'
                and the sum of all weights must be 1.
        """
        super().__init__(dists, w)
        if not all(isinstance(cd, AbstractCircularDistribution) for cd in dists):
            raise TypeError(
                "All elements of 'dists' must be of type AbstractCircularDistribution."
            )

        if np.shape(dists) != np.shape(w):
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
        self.w = w / np.sum(w)
