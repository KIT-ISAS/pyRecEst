import numpy as np
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from abstract_uniform_distribution import AbstractUniformDistribution

class HypersphericalUniformDistribution (AbstractUniformDistribution, AbstractHypersphericalDistribution):
    def __init__(self, dim_):
        assert isinstance(dim_, int) and dim_ >= 2, "dim_ must be an integer greater than or equal to 2"
        self.dim = dim_
        
    def sample(self, n):
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        
        if self.dim == 3:
            s = np.empty((self.dim, n))
            phi = 2 * np.pi * np.random.rand(1, n)
            s[2, :] = np.random.rand(n) * 2 - 1
            r = np.sqrt(1 - s[2, :]**2)
            s[0, :] = r * np.cos(phi)
            s[1, :] = r * np.sin(phi)
        else:
            samples_unnorm = np.random.randn(self.dim, n)
            s = samples_unnorm / np.linalg.norm(samples_unnorm, axis=0)
        return s
    
    def get_manifold_size(self):
        return AbstractHypersphericalDistribution.get_manifold_size(self)
