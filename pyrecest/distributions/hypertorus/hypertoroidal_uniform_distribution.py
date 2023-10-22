from math import pi
from pyrecest.backend import random
from typing import Union
from pyrecest.backend import prod
from pyrecest.backend import ones
from pyrecest.backend import ndim
from pyrecest.backend import log
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import zeros


from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalUniformDistribution(
    AbstractUniformDistribution, AbstractHypertoroidalDistribution
):
    def pdf(self, xs):
        """
        Returns the Probability Density Function evaluated at xs

        :param xs: Values at which to evaluate the PDF
        :returns: PDF evaluated at xs
        """
        if xs.ndim == 0:
            assert self.dim==1
            n_inputs = 1
        elif xs.ndim == 1 and self.dim==1:
            n_inputs = xs.shape[0]
        elif xs.ndim == 1:
            assert self.dim==xs.shape[0]
            n_inputs = 1
        else:
            n_inputs = xs.shape[0]
        
        return 1.0 / self.get_manifold_size() * ones(n_inputs)

    def trigonometric_moment(self, n: Union[int, int32, int64]):
        """
        Returns the n-th trigonometric moment

        :param n: Moment order
        :returns: n-th trigonometric moment
        """
        if n == 0:
            return ones(self.dim)

        return zeros(self.dim)

    def entropy(self) -> float:
        """
        Returns the entropy of the distribution

        :returns: Entropy
        """
        return self.dim * log(2.0 * pi)

    def mean_direction(self):
        """
        Returns the mean of the circular uniform distribution.
        Since it doesn't have a unique mean, this function always raises a ValueError.

        :raises ValueError: Circular uniform distribution does not have a unique mean
        """
        raise ValueError(
            "Hypertoroidal uniform distributions do not have a unique mean"
        )

    def sample(self, n: Union[int, int32, int64]):
        """
        Returns a sample of size n from the distribution

        :param n: Sample size
        :returns: Sample of size n
        """
        return 2.0 * pi * random.rand(n, self.dim)

    def shift(self, shift_by) -> "HypertoroidalUniformDistribution":
        """
        Shifts the distribution by shift_angles.
        Since this is a uniform distribution, the shift does not change the distribution.

        :param shift_angles: Angles to shift by
        :returns: Shifted distribution
        """
        assert shift_by.shape == (self.dim,)
        return self

    def integrate(
        self, integration_boundaries=None
    ) -> float:
        """
        Returns the integral of the distribution over the specified boundaries

        :param integration_boundaries: Optional boundaries for integration.
            If None, uses the entire distribution support.
        :returns: Integral over the specified boundaries
        """
        if integration_boundaries is None:
            left = zeros((self.dim,))
            right = 2.0 * pi * ones((self.dim,))
        else:
            left, right = integration_boundaries
        assert ndim(left) == 0 and self.dim == 1 or left.shape == (self.dim,)
        assert ndim(right) == 0 and self.dim == 1 or right.shape == (self.dim,)

        volume = prod(right - left)
        return 1.0 / (2.0 * pi) ** self.dim * volume