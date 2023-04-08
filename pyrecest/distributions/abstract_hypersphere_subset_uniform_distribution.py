import numpy as np
from .abstract_hypersphere_subset_distribution import AbstractHypersphereSubsetDistribution
from .abstract_uniform_distribution import AbstractUniformDistribution

class AbstractHypersphereSubsetUniformDistribution (AbstractHypersphereSubsetDistribution, AbstractUniformDistribution):
    def __init__(self, dim_):
        assert isinstance(dim_, int) and dim_ >= 1, "dim_ must be an integer greater than or equal to 2"
        self.dim = dim_
        
    def pdf(self, xa):
        p = (1 / self.get_manifold_size()) * np.ones(xa.size//(self.dim + 1))
        return p    
