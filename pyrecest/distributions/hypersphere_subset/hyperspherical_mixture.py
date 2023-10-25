from ..abstract_mixture import AbstractMixture
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalMixture(AbstractMixture, AbstractHypersphericalDistribution):
    """
    A class used to represent a mixture of hyperspherical distributions.
    """

    def __init__(self, dists: list[AbstractHypersphericalDistribution], w):
        """
        Initializes the HypersphericalMixture with a list of distributions and weights.

        Args:
            dists (List[AbstractHypersphericalDistribution]): The list of hyperspherical distributions.
            w (List[float]): The list of weights for each distribution.
        """
        AbstractHypersphericalDistribution.__init__(self, dim=dists[0].dim)
        assert all(
            isinstance(dist, AbstractHypersphericalDistribution) for dist in dists
        ), "dists must be a list of hyperspherical distributions"

        AbstractMixture.__init__(self, dists, w)
