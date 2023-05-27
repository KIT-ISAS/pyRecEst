from typing import List, Union

import numpy as np

from ..abstract_mixture import AbstractMixture
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalMixture(AbstractMixture, AbstractHypertoroidalDistribution):
    def __init__(
        self, dists: List[AbstractHypertoroidalDistribution], w: np.ndarray = None
    ):
        """
        Constructor

        :param dists: list of hypertoroidal distributions
        :param w: list of weights
        """
        assert all(
            isinstance(dist, AbstractHypertoroidalDistribution) for dist in dists
        ), "dists must be a list of hypertoroidal distributions"

        AbstractHypertoroidalDistribution.__init__(self, dim=dists[0].dim)
        AbstractMixture.__init__(self, dists, w)

    def trigonometric_moment(self, n: Union[int, np.int64]) -> np.ndarray:
        """
        Calculate n-th trigonometric moment

        :param n: number of moment
        :returns: n-th trigonometric moment (complex number)
        """
        m = np.zeros(self.dim, dtype=np.complex128)
        for i in range(len(self.dists)):
            m += self.w[i] * self.dists[i].trigonometric_moment(
                n
            )  # Calculate moments using moments of each component
        return m

    def shift(self, shift_angles: np.ndarray):
        """
        Shifts the distribution by shift_angles.

        :param shift_angles: angles to shift by
        :returns: shifted distribution
        """
        assert len(shift_angles) == self.dim
        hd = HypertoroidalMixture(self.dists, self.w)
        hd.dists = [dist.shift(shift_angles) for dist in hd.dists]
        return hd

    def to_circular_mixture(self):
        """
        Convert to a circular mixture (only in 1D case)

        :returns: CircularMixture with same parameters
        """
        assert self.dim == 1
        from ..circle.circular_mixture import CircularMixture

        return CircularMixture(self.dists, self.w)

    def to_toroidal_mixture(self):
        """
        Convert to a toroidal mixture (only in 2D case)

        :returns: ToroidalMixture with same parameters
        """
        assert self.dim == 2
        from toroidal_mixture import ToroidalMixture

        return ToroidalMixture(self.dists, self.w)

    @property
    def input_dim(self):
        return AbstractHypertoroidalDistribution.input_dim.fget(self)
