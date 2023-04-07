import numpy as np
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from abstract_hypersphere_subset_uniform_distribution import AbstractHypersphereSubsetUniformDistribution

class HypersphericalUniformDistribution(AbstractHypersphericalDistribution, AbstractHypersphereSubsetUniformDistribution):
    def __init__(self, dim_):
        assert isinstance(dim_, int) and dim_ >= 1, "dim_ must be an integer greater than or equal to 2"
        self.dim = dim_
        
    def pdf(self, xa):
        p = (1 / self.get_manifold_size()) * np.ones(xa.size//(self.dim + 1))
        return p    
    
    def sample(self, n):
        assert isinstance(n, int) and n > 0, "n must be a positive integer"
        
        if self.dim == 2:
            s = np.empty((n, self.dim + 1,))
            phi = 2 * np.pi * np.random.rand(n)
            s[:, 2] = np.random.rand(n) * 2 - 1
            r = np.sqrt(1 - s[:, 2]**2)
            s[:, 0] = r * np.cos(phi)
            s[:, 1] = r * np.sin(phi)
        else:
            samples_unnorm = np.random.randn(n, self.dim + 1)
            s = samples_unnorm / np.linalg.norm(samples_unnorm, axis=1, keepdims=True)
        return s
    
    def get_manifold_size(self):
        return AbstractHypersphericalDistribution.get_manifold_size(self)
