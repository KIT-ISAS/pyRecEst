from abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution
from abstract_uniform_distribution import AbstractUniformDistribution
from hyperspherical_uniform_distribution import HypersphericalUniformDistribution

class HyperhemisphericalUniformDistribution(AbstractHyperhemisphericalDistribution, AbstractUniformDistribution):
    def __init__(self, dim_):
        assert isinstance(dim_, int) and dim_ >= 1, "dim_ must be an integer greater than or equal to 1"
        self.dim = dim_

    def sample(self, n):
        s = HypersphericalUniformDistribution(self.dim).sample(n)
        # Mirror ones with negative on the last dimension up for hemisphere. This
        # may give a disadvantage to ones with exactly zero at the first dimension but
        # since this is hit with quasi probability zero, we neglect
        # this.
        s = (1 - 2 * (s[-1, :] < 0)) * s
        return s
