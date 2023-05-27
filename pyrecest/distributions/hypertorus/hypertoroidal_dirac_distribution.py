import copy
from typing import Optional, Union

import numpy as np

from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from beartype import beartype

class HypertoroidalDiracDistribution(
    AbstractDiracDistribution, AbstractHypertoroidalDistribution
):
    def __init__(
        self, d: np.ndarray, w: Optional[np.ndarray] = None, dim: Optional[int] = None
    ):
        """Can set dim manually to tell apart number of samples vs dimension for 1-D arrays."""
        if dim is None:
            if d.ndim > 1:
                dim = d.shape[-1]
            elif w is not None:
                dim = np.size(d) // np.size(w)
            else:
                raise ValueError("Cannot determine dimension.")

        AbstractHypertoroidalDistribution.__init__(self, dim)
        AbstractDiracDistribution.__init__(self, np.mod(d, 2 * np.pi), w=w)

    def plot(self, *args, **kwargs):
        raise NotImplementedError("Plotting is not implemented")

    def mean_direction(self):
        """
        Calculate the mean direction of the HypertoroidalDiracDistribution.

        :param self: HypertoroidalDiracDistribution instance
        :return: Mean direction
        """
        a = self.trigonometric_moment(1)
        m = np.mod(np.arctan2(np.imag(a), np.real(a)), 2 * np.pi)
        return m

    @beartype
    def trigonometric_moment(self, n: Union[int, np.int64]):
        """
        Calculate the trigonometric moment of the HypertoroidalDiracDistribution.

        :param self: HypertoroidalDiracDistribution instance
        :param n: Integer moment order
        :return: Trigonometric moment
        """
        return np.sum(
            np.exp(1j * n * self.d.T) * np.tile(self.w, (self.dim, 1)), axis=1
        )

    def apply_function(self, f):
        dist = super().apply_function(f)
        dist.d = np.mod(dist.d, 2 * np.pi)
        return dist

    def to_toroidal_wd(self):
        from .toroidal_dirac_distribution import ToroidalDiracDistribution

        assert self.dim == 2, "The dimension must be 2"
        twd = ToroidalDiracDistribution(self.d, self.w)
        return twd

    def marginalize_to_1D(self, dimension):
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        return CircularDiracDistribution(self.d[:, dimension], self.w)

    def marginalize_out(self, dimensions: Union[int, list[int]]):
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        remaining_dims = list(range(self.dim))
        remaining_dims = [dim for dim in remaining_dims if dim != dimensions]
        return CircularDiracDistribution(self.d[:, remaining_dims], self.w)

    def shift(self, shift_angles):
        assert shift_angles.shape[-1] == self.dim
        hd = copy.copy(self)
        hd.d = np.mod(self.d + np.reshape(shift_angles, (1, -1)), 2 * np.pi)
        return hd

    def entropy(self):
        # Implement the entropy calculation here.
        raise NotImplementedError("Entropy calculation is not implemented")

    def to_wd(self):
        assert self.dim == 1
        from ..circle.circular_dirac_distribution import CircularDiracDistribution

        return CircularDiracDistribution(self.d, self.w)
