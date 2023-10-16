from pyrecest.backend import eye
from pyrecest.backend import zeros
import numpy as np
from pyrecest.distributions import GaussianDistribution

from .abstract_sampler import AbstractSampler


class AbstractEuclideanSampler(AbstractSampler):
    pass


class GaussianSampler(AbstractEuclideanSampler):
    def sample_stochastic(self, n_samples: int, dim: int) -> np.ndarray:
        return GaussianDistribution(zeros(dim), eye(dim)).sample(n_samples)
