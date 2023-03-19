from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from abstract_mixture import AbstractMixture

class HypersphericalMixture(AbstractMixture, AbstractHypersphericalDistribution):
    def __init__(self, dists, w):
        assert all(isinstance(dist, AbstractHypersphericalDistribution) for dist in dists), \
            'dists must be a list of hyperspherical distributions'
        #if all(isinstance(dist, AbstractSphericalHarmonicDistribution) for dist in dists):
        #    print('Warning: Creating a mixture of Spherical Harmonics may not be necessary.')

        self.dists = dists
        self.w = w
        AbstractMixture.__init__(self, dists, w)
