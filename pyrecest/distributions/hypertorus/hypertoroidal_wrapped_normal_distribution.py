import copy

import numpy as np
from scipy.stats import multivariate_normal

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalWrappedNormalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self, mu_, C_):
        AbstractHypertoroidalDistribution.__init__(self, mu_.shape[0])
        assert C_.shape[0] == C_.shape[1], "C must be dim x dim"
        assert np.allclose(C_, C_.T, atol=1e-8), "C must be symmetric"
        assert np.all(np.linalg.eigvals(C_) > 0), "C must be positive definite"
        assert mu_.shape[0] == C_.shape[1], "mu must be dim x 1"

        self.mu = np.mod(mu_, 2 * np.pi)
        self.C = C_

    def pdf(self, xs, m=3):
        xs = np.reshape(xs, (-1, self.dim))
        dim = self.mu.shape[0]

        # Generate all combinations of offsets for each dimension
        offsets = [np.arange(-m, m + 1) * 2 * np.pi for _ in range(dim)]
        offset_combinations = np.array(np.meshgrid(*offsets)).T.reshape(-1, dim)

        # Calculate the PDF values by considering all combinations of offsets
        pdf_values = np.zeros(xs.shape[0])
        for offset in offset_combinations:
            shifted_xa = xs + offset[np.newaxis, :]
            pdf_values += multivariate_normal.pdf(
                shifted_xa, mean=self.mu.flatten(), cov=self.C
            )

        return pdf_values

    def shift(self, shift_angles):
        # Shift distribution by the given angles
        #
        # Parameters:
        #   shiftAngles (self.dim,) vector
        #       angles to shift by
        # Returns:
        #   hd (HypertoroidalWrappedNormalDistribution)
        #       shifted distribution
        assert shift_angles.shape == (self.dim,)

        hd = self
        hd.mu = np.mod(self.mu + shift_angles, 2 * np.pi)
        return hd

    def sample(self, n):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")

        s = np.random.multivariate_normal(self.mu, self.C, n)
        s = np.mod(s, 2 * np.pi)  # wrap the samples
        return s

    def set_mode(self, m):
        """
        Set the mode of the distribution.

        Parameters:
        m (numpy array): The new mode.

        Returns:
        HypertoroidalWNDistribution: A new instance of the distribution with the updated mode.
        """
        dist = copy.deepcopy(self)
        dist.mu = m
        return dist

    def trigonometric_moment(self, n):
        """
        Calculate the trigonometric moment of the HypertoroidalWNDistribution.

        :param self: HypertoroidalWNDistribution instance
        :param n: Integer moment order
        :return: Trigonometric moment
        """
        assert isinstance(n, int), "n must be an integer"

        m = np.exp(
            [1j * n * self.mu[i] - n**2 * self.C[i, i] / 2 for i in range(self.dim)]
        )

        return m

    def mode(self):
        # Determines the mode of the distribution, i.e., the point
        # where the pdf is largest.
        #
        # Returns:
        #   m (vector)
        #       the mode
        return self.mu
