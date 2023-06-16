import copy

import numpy as np
from scipy.stats import multivariate_normal

from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from ..nonperiodic.gaussian_distribution import GaussianDistribution
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


class PartiallyWrappedNormalDistribution(
    AbstractHypercylindricalDistribution
):
    def __init__(
        self, mu: np.ndarray, C: np.ndarray, bound_dim: int | np.int32 | np.int64
    ):
        assert bound_dim >= 0, "bound_dim must be non-negative"
        assert np.ndim(mu) == 1, "mu must be a 1-dimensional array"
        assert C.shape == (np.size(mu), np.size(mu)), "C must match size of mu"
        assert np.allclose(C, C.T), "C must be symmetric"
        assert np.all(np.linalg.eigvals(C) > 0), "C must be positive definite"
        assert bound_dim <= np.size(mu)
        assert np.ndim(mu) == 1
        if bound_dim > 0:  # This decreases the need for many wrappings
            mu[:bound_dim] = np.mod(mu[:bound_dim], 2 * np.pi)

        AbstractHypercylindricalDistribution.__init__(
            self, bound_dim=bound_dim, lin_dim=mu.shape[0] - bound_dim
        )

        self.mu = mu
        self.mu[:bound_dim] = np.mod(self.mu[:bound_dim], 2 * np.pi)
        self.C = C

    def pdf(self, xs: np.ndarray, m: int | np.int32 | np.int64 = 3):
        xs = np.atleast_2d(xs)
        if self.bound_dim > 0:
            xs[:, : self.bound_dim] = np.mod(xs[:, : self.bound_dim], 2 * np.pi)

        assert xs.shape[-1] == self.input_dim

        # generate multiples for wrapping
        multiples = np.array(range(-m, m + 1)) * 2 * np.pi

        # create meshgrid for all combinations of multiples
        mesh = np.array(np.meshgrid(*[multiples] * self.bound_dim)).reshape(
            -1, self.bound_dim
        )

        # reshape xs for broadcasting
        xs_reshaped = np.tile(xs[:, : self.bound_dim], (mesh.shape[0], 1))  # noqa: E203

        # prepare data for wrapping (not applied to linear dimensions)
        xs_wrapped = xs_reshaped + np.repeat(mesh, xs.shape[0], axis=0)
        xs_wrapped = np.concatenate(
            [
                xs_wrapped,
                np.tile(xs[:, self.bound_dim :], (mesh.shape[0], 1)),  # noqa: E203
            ],
            axis=1,
        )

        # evaluate normal for all xs_wrapped
        mvn = multivariate_normal(self.mu, self.C)
        evals = mvn.pdf(xs_wrapped)

        # sum evaluations for the wrapped dimensions
        summed_evals = np.sum(evals.reshape(-1, (2 * m + 1) ** self.bound_dim), axis=1)

        return summed_evals

    def mode(self):
        """
        Determines the mode of the distribution, i.e., the point where the pdf is largest.
        Returns:
            m (lin_dim + bound_dim,) vector: the mode
        """
        return self.mu

    def set_mode(self, new_mode: np.ndarray):
        self.mu = copy.copy(new_mode)
        return self

    def hybrid_moment(self):
        """
        Calculates mean of [x1, x2, .., x_lin_dim, cos(x_(linD+1), sin(x_(linD+1)), ..., cos(x_(linD+boundD), sin(x_(lin_dim+bound_dim))]
        Returns:
            mu (linD+2 x 1): expectation value of [x1, x2, .., x_lin_dim, cos(x_(lin_dim+1), sin(x_(lin_dim+1)), ..., cos(x_(lin_dim+bound_dim), sin(x_(lin_dim+bound_dim))]
        """
        mu = np.NaN * np.zeros((2 * self.bound_dim + self.lin_dim, 1))
        mu[2 * self.bound_dim :, :] = self.mu[self.bound_dim :]  # noqa: E203
        for i in range(self.bound_dim):
            mu[2 * i, :] = np.cos(self.mu[i]) * np.exp(-self.C[i, i] / 2)  # noqa: E203
            mu[2 * i + 1, :] = np.sin(self.mu[i]) * np.exp(  # noqa: E203
                -self.C[i, i] / 2
            )
        return mu

    def hybrid_mean(self):
        return self.mu

    def linear_mean(self):
        return self.mu[-self.lin_dim :]  # noqa: E203

    def periodic_mean(self):
        return self.mu[: self.bound_dim]

    def sample(self, n: int):
        """
        Sample n points from the distribution
        Parameters:
            n (int): number of points to sample
        """
        assert n > 0, "n must be positive"
        s = np.random.multivariate_normal(self.mu, self.C, n)
        s[:, : self.bound_dim] = np.mod(s[:, : self.bound_dim], 2 * np.pi)  # noqa: E203
        return s

    def to_gaussian(self):
        return GaussianDistribution(self.mu, self.C)

    def linear_covariance(self):
        return self.C[self.bound_dim :, self.bound_dim :]  # noqa: E203

    def marginalize_periodic(self):
        return GaussianDistribution(
            self.mu[self.bound_dim :],  # noqa: E203
            self.C[self.bound_dim :, self.bound_dim :],  # noqa: E203
        )

    def marginalize_linear(self):
        return HypertoroidalWrappedNormalDistribution(
            self.mu[: self.bound_dim],
            self.C[: self.bound_dim, : self.bound_dim],  # noqa: E203
        )
