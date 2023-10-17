from math import pi
from pyrecest.backend import linspace
import numpy as np
from beartype import beartype
from pyrecest.distributions import CircularUniformDistribution

from .abstract_sampler import AbstractSampler


class AbstractHypertoroidalSampler(AbstractSampler):
    pass


class AbstractCircularSampler(AbstractHypertoroidalSampler):
    pass


class CircularUniformSampler(AbstractCircularSampler):
    def sample_stochastic(self, n_samples: int):
        return CircularUniformDistribution().sample(n_samples)

    def get_grid(self, grid_density_parameter: int) -> np.ndarray:
        """
        Returns an equidistant grid of points on the circle [0,2*pi).
        """
        points = linspace(0, 2 * pi, grid_density_parameter, endpoint=False)
        # Set it to the middle of the interval instead of the start
        points += (2 * pi / grid_density_parameter) / 2
        return points