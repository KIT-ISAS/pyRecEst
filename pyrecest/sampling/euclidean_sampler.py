from .abstract_sampler import AbstractSampler
import numpy as np
from pyrecest.distributions import GaussianDistribution

class AbstractEuclideanSampler(AbstractSampler):
    pass

class AbstractGaussianSampler(AbstractEuclideanSampler):
    def sample_stochastic(self, n_samples: int, dim: int) -> np.ndarray:
        return GaussianDistribution(np.zeros(dim), np.eye(dim)).sample(n_samples)
