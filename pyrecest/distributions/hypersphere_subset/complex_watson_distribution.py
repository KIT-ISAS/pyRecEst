# pylint: disable=redefined-builtin,no-name-in-module,no-member
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

import mpmath


class ComplexWatsonDistribution:
    """Complex Watson distribution on the complex unit sphere C^D.

    References:
        Mardia, K. V. & Dryden, I. L.
        The Complex Watson Distribution and Shape Analysis
        Journal of the Royal Statistical Society: Series B
        (Statistical Methodology), Blackwell Publishers Ltd., 1999, 61, 913-926
    """

    EPSILON = 1e-6

    def __init__(self, mu, kappa):
        """
        Parameters:
            mu: 1-D complex array of shape (D,), must be a unit vector
            kappa: scalar concentration parameter
        """
        mu = np.asarray(mu, dtype=complex)
        assert mu.ndim == 1, "mu must be a 1-D vector"
        assert (
            abs(np.linalg.norm(mu) - 1.0) < self.EPSILON
        ), "mu must be a unit vector"

        self.mu = mu
        self.kappa = float(kappa)
        self.dim = mu.shape[0]  # D: length of the complex feature vector
        self._log_norm_const = None

    def mean(self):
        """Return the mean direction (mode) of the distribution."""
        return self.mu

    @property
    def log_norm_const(self):
        """Log normalization constant log(C_D(kappa))."""
        if self._log_norm_const is None:
            self._log_norm_const = ComplexWatsonDistribution.log_norm(
                self.dim, self.kappa
            )
        return self._log_norm_const

    def pdf(self, za):
        """Evaluate the pdf at each row of za.

        Parameters:
            za: complex array of shape (N, D) or (D,) for a single point

        Returns:
            Real-valued array of shape (N,) or a scalar
        """
        za = np.asarray(za, dtype=complex)
        inner = za @ self.mu.conj()  # complex inner product, shape (N,) or scalar
        return np.exp(self.log_norm_const + self.kappa * np.abs(inner) ** 2).real

    def sample(self, n):
        """Sample from the complex Watson distribution.

        Uses the complex Bingham distribution sampling algorithm described in
        Mardia, K. V. & Jupp, P. E., Directional Statistics, Wiley, 2009, p. 336.

        Parameters:
            n: number of samples

        Returns:
            complex array of shape (n, D)
        """
        D = self.dim

        # Compute B = -kappa * (I - mu * mu^H)
        B = -self.kappa * (np.eye(D) - np.outer(self.mu, self.mu.conj()))

        if B.shape[0] != B.shape[1]:
            raise ValueError("Matrix B is not square.")
        if D < 2:
            raise ValueError("Matrix B is too small.")

        # Force Hermitian
        B = 0.5 * (B + B.conj().T)

        # Eigendecompose -B; eigh returns eigenvalues in ascending order
        Lambda, V = np.linalg.eigh(-B)

        # Reorder to descending
        idx = np.argsort(Lambda)[::-1]
        Lambda = Lambda[idx].real
        V = V[:, idx]

        # Shift so the last eigenvalue is zero
        Lambda = Lambda - Lambda[-1]

        temp1 = -1.0 / Lambda[:-1]
        temp2 = 1.0 - np.exp(-Lambda[:-1])

        samples = np.zeros((n, D), dtype=complex)

        for k in range(n):
            S = np.zeros(D)
            while True:
                U = np.random.uniform(size=D - 1)
                # For small Lambda the truncated-exponential is nearly uniform.
                small = Lambda[:-1] < 0.03
                S[:-1] = np.where(small, U, temp1 * np.log(1.0 - U * temp2))
                if np.sum(S[:-1]) < 1.0:
                    break

            S[-1] = 1.0 - np.sum(S[:-1])

            # Independent random phase angles
            theta = 2.0 * np.pi * np.random.uniform(size=D)
            W = np.sqrt(np.maximum(S, 0.0)) * np.exp(1j * theta)

            samples[k] = V @ W

        return samples

    @classmethod
    def fit(cls, Z, weights=None):
        """Fit a complex Watson distribution to data using MLE.

        Parameters:
            Z: complex array of shape (N, D)
            weights: optional 1-D real array of shape (N,)

        Returns:
            ComplexWatsonDistribution
        """
        mu_hat, kappa_hat = cls.estimate_parameters(Z, weights)
        return cls(mu_hat, kappa_hat)

    @staticmethod
    def estimate_parameters(Z, weights=None):
        """Estimate parameters of the complex Watson distribution via MLE.

        Following Mardia & Dryden (1999).

        Parameters:
            Z: complex array of shape (N, D)
            weights: optional 1-D real array of shape (N,)

        Returns:
            tuple (mu_hat, kappa_hat)
        """
        Z = np.asarray(Z, dtype=complex)
        N, D = Z.shape

        # Build the Hermitian scatter matrix S (D x D)
        if weights is None:
            S = Z.conj().T @ Z
        else:
            weights = np.asarray(weights, dtype=float).ravel()
            assert weights.shape[0] == N, "dimensions of Z and weights mismatch"
            assert weights.ndim == 1, "weights must be a 1-D vector"
            # Weighted scatter: Σ_i w_i·z_i·z_i^H, normalised so uniform
            # weights (w_i=1) reproduce the unweighted result Z^H Z.
            S = (Z * weights[:, None]).conj().T @ Z * (N / np.sum(weights))

        # Force Hermitian
        S = 0.5 * (S + S.conj().T)

        eigvals, eigvecs = np.linalg.eigh(S)

        # Reorder to descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx].real
        eigvecs = eigvecs[:, idx]

        assert np.all(eigvals > 0), (
            "All eigenvalues of the scatter matrix must be positive; "
            "this may indicate that fewer samples than features were provided."
        )

        mu_hat = eigvecs[:, 0]

        # High-concentration approximation used as initial / fallback estimate
        kappa_hat = N * (D - 1) / (N - eigvals[0])

        approximation_threshold = 200.0
        if kappa_hat < approximation_threshold:
            concentration_max = 1000.0
            normed_lambda = eigvals[0] / N
            kappa_hat = ComplexWatsonDistribution._hypergeometric_ratio_inverse(
                normed_lambda, D, concentration_max
            )

        return mu_hat, float(kappa_hat)

    @staticmethod
    def _hypergeometric_ratio_inverse(rho, D, kappa_max):
        """Find kappa satisfying the MLE score equation for complex Watson.

        Solves for kappa in:
            1F1(2; D+1; kappa) / (D * 1F1(1; D; kappa)) = rho

        which equals E[|mu^H z|^2] = rho under the distribution.

        Parameters:
            rho: target ratio (= observed lambda_1 / N)
            D: feature-space dimension
            kappa_max: upper bracket for the root search

        Returns:
            kappa (float)
        """

        def score(kappa):
            if kappa <= 0:
                return 1.0 / D - rho
            kappa_mp = mpmath.mpf(kappa)
            h1 = mpmath.hyp1f1(1, D, kappa_mp)
            # d/dkappa 1F1(1;D;kappa) = (1/D) * 1F1(2;D+1;kappa)
            dh = mpmath.hyp1f1(2, D + 1, kappa_mp) / D
            # Compute ratio in mpmath to avoid float overflow for large kappa
            return float(dh / h1) - rho

        try:
            return brentq(score, 0.0, kappa_max)
        except ValueError:
            return kappa_max

    @staticmethod
    def log_norm(D, kappa):
        """Compute the log normalization constant log(C_D(kappa)).

        Uses three numerical regimes for stability (Mardia & Dryden 1999).

        Parameters:
            D: feature-space dimension (integer >= 1)
            kappa: concentration parameter – scalar or array-like

        Returns:
            log(C_D(kappa)); same shape as kappa (scalar if scalar input)
        """
        scalar_input = np.isscalar(kappa)
        kappa_arr = np.atleast_1d(np.asarray(kappa, dtype=float)).ravel()
        log_c = np.zeros_like(kappa_arr)

        with np.errstate(divide="ignore", invalid="ignore"):
            # ----------------------------------------------------------
            # High-kappa asymptotic formula (Mardia1999Watson, p. 917)
            #   log(1/C) ≈ log(2) + D·log(π) + (1-D)·log(κ) + κ
            # ----------------------------------------------------------
            log_c_high = (
                np.log(2)
                + D * np.log(np.pi)
                + (1 - D) * np.log(kappa_arr)
                + kappa_arr
            )

            # ----------------------------------------------------------
            # Medium-kappa formula (Mardia1999Watson, Eq. 3)
            #   correction = exp(-κ) · Σ_{i=0}^{D-2} κ^i / i!
            #   log(1/C) = log_c_high + log(1 - correction)
            # ----------------------------------------------------------
            if D > 1:
                i_arr = np.arange(D - 1, dtype=float)  # 0, 1, ..., D-2
                correction = np.exp(-kappa_arr) * np.sum(
                    kappa_arr[:, None] ** i_arr / np.exp(gammaln(i_arr + 1.0)),
                    axis=1,
                )
                log_c_medium = log_c_high + np.log(
                    np.maximum(1.0 - correction, 1e-300)
                )
            else:
                log_c_medium = log_c_high  # D=1: empty sum → correction = 0

            # ----------------------------------------------------------
            # Low-kappa Taylor series (Mardia1999Watson, Eq. 4, 10 terms)
            #   1F1(1;D;κ) ≈ 1 + Σ_{i=1}^{10} κ^i / (D·(D+1)·…·(D+i-1))
            # ----------------------------------------------------------
            ratios = kappa_arr[:, None] / np.arange(D, D + 10, dtype=float)
            cumprods = np.cumprod(ratios, axis=1)
            hyp1f1_approx = 1.0 + np.sum(cumprods, axis=1)
            log_fact_Dm1 = gammaln(float(D))  # log Γ(D) = log (D-1)!
            log_c_low = (
                np.log(2)
                + D * np.log(np.pi)
                - log_fact_Dm1
                + np.log(hyp1f1_approx)
            )

        # Select the appropriate regime
        mask_low = kappa_arr < (1.0 / D)
        mask_high = kappa_arr >= 100.0
        mask_medium = ~mask_low & ~mask_high

        log_c[mask_low] = log_c_low[mask_low]
        log_c[mask_medium] = log_c_medium[mask_medium]
        log_c[mask_high] = log_c_high[mask_high]

        # Negate: the three formulas compute log(1/C); we want log(C)
        log_c = -log_c

        if scalar_input:
            return float(log_c[0])
        return log_c
