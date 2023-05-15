from typing import List
import numpy as np
from abstract_circular_distribution import AbstractCircularDistribution
from ..hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from .circular_fourier_distribution import CircularFourierDistribution
from .circular_dirac_distribution import CircularDiracDistribution


class CircularMixture(AbstractCircularDistribution, HypertoroidalMixture):
    def __init__(self, dists: List[AbstractCircularDistribution], w: np.ndarray):
        super().__init__(dists, w)
        assert all(isinstance(cd, AbstractCircularDistribution) for cd in dists), \
            "dists must be a list of circular distributions"
        assert np.all(np.shape(dists) == np.shape(w)), "size of dists and w must be equal"

        if all(isinstance(cd, CircularFourierDistribution) for cd in dists):
            print("Warning: Mixtures of Fourier distributions can be built by combining the Fourier coefficients so using a mixture may not be necessary")
        elif all(isinstance(cd, CircularDiracDistribution) for cd in dists):
            print("Warning: Mixtures of WDDistributions can usually be combined into one WDDistribution.")

        self.dists = dists
        self.w = w / np.sum(w)
