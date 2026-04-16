from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)

from .abstract_dummy_filter import AbstractDummyFilter
from .manifold_mixins import HypertoroidalFilterMixin


class HypertoroidalDummyFilter(AbstractDummyFilter, HypertoroidalFilterMixin):
    """Hypertoroidal dummy filter initialized with a uniform distribution.

    This filter does nothing on predictions and updates, always returning
    samples from the initial uniform distribution as point estimates.
    """

    def __init__(self, dim):
        """Initialize HypertoroidalDummyFilter.

        Parameters:
            dim (int >= 1): Manifold dimension of the hypertorus (e.g. 1 for T^1).
        """
        assert dim >= 1, "dim must be at least 1"
        HypertoroidalFilterMixin.__init__(self)
        AbstractDummyFilter.__init__(self, HypertoroidalUniformDistribution(dim))

    def get_point_estimate(self):
        return AbstractDummyFilter.get_point_estimate(self)
