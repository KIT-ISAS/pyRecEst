import numpy as np
from beartype import beartype

from ..hypertorus.hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution
from .abstract_circular_distribution import AbstractCircularDistribution


class CircularDiracDistribution(
    HypertoroidalDiracDistribution, AbstractCircularDistribution
):
    @beartype
    def __init__(self, d: np.ndarray, w: np.ndarray | None = None):
        """
        Initializes a CircularDiracDistribution instance.

        Args:
            d (np.ndarray): The Dirac locations.
            w (Optional[np.ndarray]): The weights for each Dirac location.
        """
        super().__init__(d, w, dim=1)
        assert self.d.shape[0] == self.w.shape[0], "The shapes of d and w should match."

    def plot_interpolated(self, _):
        """
        Raises an exception since interpolation is not available for WDDistribution.
        """
        raise NotImplementedError("No interpolation available for WDDistribution.")
