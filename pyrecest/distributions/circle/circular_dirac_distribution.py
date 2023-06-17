import numpy as np
from beartype import beartype

from ..hypertorus.hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution
from .abstract_circular_distribution import AbstractCircularDistribution


class CircularDiracDistribution(
    HypertoroidalDiracDistribution, AbstractCircularDistribution
):

    def plot_interpolated(self, _):
        """
        Raises an exception since interpolation is not available for CircularDiracDistribution.
        """
        raise NotImplementedError("No interpolation available for CircularDiracDistribution.")
