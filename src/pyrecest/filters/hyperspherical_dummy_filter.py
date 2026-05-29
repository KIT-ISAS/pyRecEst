from numbers import Integral

from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)

from .abstract_dummy_filter import AbstractDummyFilter
from .manifold_mixins import HypersphericalFilterMixin


class HypersphericalDummyFilter(AbstractDummyFilter, HypersphericalFilterMixin):
    """Hyperspherical dummy filter initialized with a uniform distribution.

    This filter does nothing on predictions and updates, always returning
    samples from the initial uniform distribution as point estimates.
    """

    def __init__(self, dim):
        """Initialize HypersphericalDummyFilter.

        Parameters:
            dim (int >= 2): Manifold dimension of the hypersphere (e.g. 2 for S^2).
        """
        if isinstance(dim, bool) or not isinstance(dim, Integral):
            raise ValueError("dim must be an integer at least 2.")
        dim = int(dim)
        if dim < 2:
            raise ValueError("dim must be an integer at least 2.")
        HypersphericalFilterMixin.__init__(self)
        AbstractDummyFilter.__init__(self, HypersphericalUniformDistribution(dim))

    def get_point_estimate(self):
        return AbstractDummyFilter.get_point_estimate(self)
