# pylint: disable=redefined-builtin,no-name-in-module,no-member
import numpy as np

from pyrecest.backend import (
    abs,
    allclose,
    array,
    conj,
    exp,
    gammaln,
    linalg,
    log,
    pi,
    random,
    real,
    sqrt,
    sum,
    transpose,
)


class ComplexAngularCentralGaussianDistribution:
    """Complex Angular Central Gaussian distribution on the complex unit hypersphere.

    This distribution is defined on the complex unit sphere in C^d (equivalently S^{2d-1}
    in R^{2d}). It is parameterized by a Hermitian positive definite matrix C of shape (d, d).

    Reference:
        Ported from ComplexAngularCentralGaussian.m in libDirectional:
        https://github.com/libDirectional/libDirectional/blob/master/lib/distributions/complexHypersphere/ComplexAngularCentralGaussian.m
    """

    def __init__(self, C):
        """Initialize the distribution.

        Parameters
        ----------
        C : array-like of shape (d, d)
            Hermitian positive definite parameter matrix.
        """
        assert allclose(C, conj(transpose(C))), "C must be Hermitian"
        self.C = C
        self.dim = C.shape[0]

    def pdf(self, za):
        """Evaluate the pdf at each row of za.

        Parameters
        ----------
        za : array-like of shape (n, d) or (d,)
            Points on the complex unit sphere. Each row is a complex unit vector.

        Returns
        -------
        p : array-like of shape (n,) or scalar
            PDF values at each row of za.
        """
        single = za.ndim == 1
        if single:
            za = za.reshape(1, -1)

        # Solve C * X = za.T to get C^{-1} * za.T, shape (d, n)
        C_inv_z = linalg.solve(self.C, transpose(za))
        # Hermitian quadratic form: inner[i] = za[i]^H C^{-1} za[i]
        inner = sum(conj(transpose(za)) * C_inv_z, axis=0)  # shape (n,)

        d = self.dim
        # gamma(d) / (2 * pi^d) in log space: gammaln(d) - log(2) - d*log(pi)
        log_normalizer = gammaln(array(float(d))) - log(2.0) - d * log(array(pi))
        p = exp(log_normalizer) * abs(inner) ** (-d) / abs(linalg.det(self.C))

        if single:
            return p[0]
        return p

    def sample(self, n):
        """Sample n points from the distribution.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        Z : array-like of shape (n, d)
            Complex unit vectors sampled from the distribution.
        """
        # Lower Cholesky factor: C = L @ L^H
        L = linalg.cholesky(self.C)
        a = random.normal(size=(n, self.dim))
        b = random.normal(size=(n, self.dim))
        # Each row of (a + 1j*b) is CN(0, 2*I); transform by L^T to get CN(0, 2*C)
        # Using regular transpose (not conjugate) so each row maps as z -> L @ z (column form)
        z = (a + 1j * b) @ transpose(L)
        norms = sqrt(real(sum(z * conj(z), axis=-1)))
        return z / norms.reshape(-1, 1)

    @staticmethod
    def fit(Z, n_iterations=100):
        """Fit distribution to data Z using fixed-point iterations.

        Parameters
        ----------
        Z : array-like of shape (n, d)
            Complex unit vectors (each row is a sample).
        n_iterations : int, optional
            Number of fixed-point iterations (default 100).

        Returns
        -------
        dist : ComplexAngularCentralGaussianDistribution
        """
        C = ComplexAngularCentralGaussianDistribution.estimate_parameter_matrix(
            Z, n_iterations
        )
        return ComplexAngularCentralGaussianDistribution(C)

    @staticmethod
    def estimate_parameter_matrix(Z, n_iterations=100):
        """Estimate the parameter matrix from data using fixed-point iterations.

        Parameters
        ----------
        Z : array-like of shape (n, d)
            Complex unit vectors (each row is a sample).
        n_iterations : int, optional
            Number of iterations (default 100).

        Returns
        -------
        C : array-like of shape (d, d)
            Estimated Hermitian parameter matrix.
        """
        N = Z.shape[0]
        D = Z.shape[1]
        C = array(np.eye(D, dtype=complex))

        for _ in range(n_iterations):
            # Solve C * X = Z.T to get C^{-1} * Z.T, shape (d, n)
            C_inv_Z = linalg.solve(C, transpose(Z))
            # Hermitian quadratic forms: inner[k] = Z[k]^H C^{-1} Z[k]
            inner = sum(conj(transpose(Z)) * C_inv_Z, axis=0)  # shape (n,)
            weights = (D - 1) / abs(inner)  # shape (n,)
            # C = (1/N) * sum_k weights[k] * z_k z_k^H
            # = Z.T @ diag(weights) @ conj(Z) / N
            C = transpose(Z) @ (weights.reshape(-1, 1) * conj(Z)) / N

        return C
