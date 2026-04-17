# pylint: disable=redefined-builtin,no-name-in-module,no-member
"""
Complex Watson distribution on the complex unit sphere in C^D.

Reference:
    Mardia, K. V. & Dryden, I. L.
    The Complex Watson Distribution and Shape Analysis
    Journal of the Royal Statistical Society: Series B
    (Statistical Methodology), Blackwell Publishers Ltd., 1999, 61, 913-926.
"""

import pyrecest.backend
from scipy.optimize import brentq

from pyrecest.backend import (  # pylint: disable=no-name-in-module,no-member
    abs,
    any,
    argmax,
    argsort,
    asarray,
    atleast_1d,
    conj,
    exp,
    eye,
    gammaln,
    linalg,
    log,
    maximum,
    ones_like,
    outer,
    pi,
    random,
    real,
    sqrt,
    sum,
    where,
    zeros,
    zeros_like,
)


class ComplexWatsonDistribution:
    """
    Complex Watson distribution on the complex unit sphere in C^D.

    The PDF is: f(z; mu, kappa) = C(D, kappa)^{-1} * exp(kappa * |mu^H z|^2)

    where z in C^D is a unit vector (|z|=1), mu in C^D is the mode (unit vector),
    kappa >= 0 is the concentration parameter, and C(D, kappa) is the normalization.
    """

    EPSILON = 1e-6

    def __init__(self, mu, kappa):
        """
        Initializes the ComplexWatsonDistribution.

        Args:
            mu: D-dimensional complex unit vector (the mode direction).
            kappa (float): Concentration parameter (>= 0).
        """
        mu = asarray(mu, dtype=complex)
        assert mu.ndim == 1, "mu must be a 1-D vector"
        assert (
            abs(linalg.norm(mu) - 1.0) < self.EPSILON
        ), "mu must be normalized (|mu|=1)"

        self.mu = mu
        self.kappa = float(kappa)
        self.dim = len(mu)  # D: dimension of the complex space C^D
        self._log_c = ComplexWatsonDistribution.log_norm(self.dim, self.kappa)

    @staticmethod
    def log_norm(D, kappa):  # pylint: disable=too-many-locals
        """
        Compute the log normalization constant for the complex Watson distribution.

        Returns -log(C(D, kappa)) where
            log C(D, kappa) = log(2) + D*log(pi) - log(Gamma(D)) + log(1F1(1; D; kappa))

        Three regimes are used for numerical stability:
            - Low kappa  (kappa < 1/D): Taylor series
            - Medium kappa (1/D <= kappa < 100): intermediate correction
            - High kappa (kappa >= 100): asymptotic approximation

        Args:
            D (int): Dimension of the complex space.
            kappa: Concentration parameter(s) — scalar or array.

        Returns:
            float or ndarray: -log(C(D, kappa)), same shape as kappa.
        """
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError("log_norm is not supported on the JAX backend.")
        kappa_as_arr = asarray(kappa, dtype=float)
        scalar_input = kappa_as_arr.ndim == 0
        kappa = atleast_1d(kappa_as_arr).ravel()
        log_c = zeros_like(kappa)

        # Asymptotic formula for high kappa
        # log C ~ log(2) + D*log(pi) + (1-D)*log(kappa) + kappa
        # log_c_high is evaluated for all kappa before masking; clip to avoid log(0) warning
        log_c_high = (
            log(2.0) + D * log(pi)
            + (1 - D) * log(maximum(kappa, 1e-300)) + kappa
        )

        # Intermediate formula (Mardia1999 Eq. 3):
        # log C = log_c_high + log(1 - sum_{j=0}^{D-2} kappa^j * exp(-kappa) / j!)
        running = exp(-kappa)
        correction_sum = running.copy()
        for j in range(1, D - 1):
            running = running * kappa / j
            correction_sum = correction_sum + running
        if D >= 2:
            log_c_medium = log_c_high + log(
                maximum(1.0 - correction_sum, 1e-300)
            )
        else:
            log_c_medium = log_c_high

        # Taylor series for low kappa (Mardia1999 Eq. 4):
        # 1F1(1; D; kappa) = 1 + kappa/D + kappa^2/(D*(D+1)) + ...
        running_prod = ones_like(kappa)
        series_sum = ones_like(kappa)
        for j in range(10):
            running_prod = running_prod * kappa / (D + j)
            series_sum = series_sum + running_prod
        log_c_low = (
            log(2.0) + D * log(pi) - gammaln(D) + log(series_sum)
        )

        mask_low = kappa < 1.0 / D
        mask_high = kappa >= 100.0
        mask_medium = ~mask_low & ~mask_high

        log_c[mask_low] = log_c_low[mask_low]
        log_c[mask_medium] = log_c_medium[mask_medium]
        log_c[mask_high] = log_c_high[mask_high]

        result = -log_c
        if scalar_input:
            return float(result[0])
        return result

    def pdf(self, Z):
        """
        Evaluate the PDF at the columns of Z.

        Args:
            Z: D x N complex matrix where each column is a unit vector in C^D.
               May also be a 1-D array (single vector).

        Returns:
            ndarray: N-dimensional real array of PDF values.
        """
        Z = asarray(Z, dtype=complex)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        # |mu^H z|^2 for each column
        inner = abs(conj(self.mu) @ Z) ** 2
        return real(exp(self._log_c + self.kappa * inner))

    def sample(self, n):
        """
        Draw n unit vectors from the complex Watson distribution.

        Uses the complex Bingham representation:
            B = -kappa * (I - mu mu^H)

        Args:
            n (int): Number of samples.

        Returns:
            ndarray: D x n complex matrix of samples.
        """
        B = -self.kappa * (
            eye(self.dim, dtype=complex) - outer(self.mu, conj(self.mu))
        )
        B = 0.5 * (B + B.conj().T)
        return _sample_complex_bingham(B, n)

    @staticmethod
    def fit(Z, weights=None):
        """
        Fit a ComplexWatsonDistribution to data using MLE.

        Args:
            Z: D x N complex matrix of observations (unit vectors).
            weights: Optional 1 x N or (N,) real weight array.

        Returns:
            ComplexWatsonDistribution: Fitted distribution.
        """
        mu_hat, kappa_hat = ComplexWatsonDistribution.estimate_parameters(Z, weights)
        return ComplexWatsonDistribution(mu_hat, kappa_hat)

    @staticmethod
    def estimate_parameters(Z, weights=None):
        """
        MLE estimation of complex Watson parameters.

        Method: Mardia & Dryden (1999), Section 4.

        Args:
            Z: D x N complex matrix of observations (unit vectors).
            weights: Optional 1 x N or (N,) real weight array.

        Returns:
            tuple: (mu_hat, kappa_hat) — complex unit mode vector and concentration.
        """
        Z = asarray(Z, dtype=complex)
        D, N = Z.shape

        if weights is None:
            S = Z @ conj(Z).T
        else:
            weights = asarray(weights, dtype=float).ravel()
            assert len(weights) == N, "weights length must match number of samples"
            S = (Z * weights) @ conj(Z).T * N / sum(weights)

        S = 0.5 * (S + conj(S).T)  # enforce Hermitian

        eigenvalues, eigenvectors = linalg.eigh(S)
        idx = argmax(eigenvalues)
        mu_hat = eigenvectors[:, idx]
        lambda_max = float(real(eigenvalues[idx]))

        normed_lambda = lambda_max / N

        # High-concentration approximation (Mardia & Dryden 1999)
        kappa_approx = N * (D - 1) / max(N - lambda_max, 1e-300)
        if kappa_approx < 200:
            kappa_hat = ComplexWatsonDistribution.hypergeometric_ratio_inverse(
                normed_lambda, D, concentration_max=1000
            )
        else:
            kappa_hat = kappa_approx

        return mu_hat, kappa_hat

    @staticmethod
    def hypergeometric_ratio(kappa, D):
        """
        Compute E[|mu^H z|^2] = d(log C)/d(kappa) for the complex Watson distribution.

        This equals (1/D) * 1F1(2; D+1; kappa) / 1F1(1; D; kappa).

        Args:
            kappa (float): Concentration.
            D (int): Dimension of the complex space.

        Returns:
            float: Expected value of |mu^H z|^2, in [1/D, 1).
        """
        kappa = float(kappa)
        if kappa < 1e-10:
            return 1.0 / D
        # For large kappa use asymptotic: ratio ~ 1 - (D-1)/kappa
        # (from d/dkappa [kappa + (1-D)*log(kappa) + ...] = 1 + (1-D)/kappa)
        if kappa >= 100.0:
            return 1.0 - float(D - 1) / kappa
        # Differentiate log_norm numerically to avoid hyp1f1 overflow
        eps = max(kappa * 1e-4, 1e-7)
        log_c_plus = ComplexWatsonDistribution.log_norm(D, kappa + eps)
        log_c_minus = ComplexWatsonDistribution.log_norm(D, max(kappa - eps, 0.0))
        # ratio = d(log C)/d(kappa) = -d(log_norm)/d(kappa)  [log_norm = -log C]
        return -(log_c_plus - log_c_minus) / (2.0 * eps)

    @staticmethod
    def hypergeometric_ratio_inverse(r, D, concentration_max=500):
        """
        Find kappa such that hypergeometric_ratio(kappa, D) == r.

        Args:
            r (float): Target ratio, should be in (1/D, 1).
            D (int): Dimension of the complex space.
            concentration_max (float): Upper bound for bracket search.

        Returns:
            float: kappa value.
        """
        r = float(r)
        lower = 1.0 / D
        if r <= lower + 1e-10:
            return 0.0
        if r >= 1.0 - 1e-10:
            return float(concentration_max)

        def objective(k):
            return ComplexWatsonDistribution.hypergeometric_ratio(k, D) - r

        return brentq(objective, 0.0, float(concentration_max), xtol=1e-8)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _sample_complex_bingham(B, n):
    """
    Sample n unit vectors from a complex Bingham distribution with parameter B.

    Implements the algorithm from:
        Mardia, K. V. & Jupp, P. E. Directional Statistics, Wiley, 2009, p. 336.

    Args:
        B: D x D complex Hermitian matrix (concentration matrix).
        n (int): Number of samples.

    Returns:
        ndarray: D x n complex matrix of samples (each column is a unit vector).
    """
    B = asarray(B, dtype=complex)
    D = B.shape[0]
    B = 0.5 * (B + conj(B).T)

    # Eigendecompose -B (so eigenvalues are non-negative in descending order)
    eigenvalues, V = linalg.eigh(-B)
    idx = argsort(eigenvalues)[::-1]
    Lambda = real(eigenvalues[idx])
    V = V[:, idx]

    # Shift so smallest eigenvalue is 0 (doesn't change the distribution)
    Lambda = Lambda - Lambda[-1]

    Z = zeros((D, n), dtype=complex)
    for i in range(n):
        s = _sample_diagonal_complex_bingham_magnitudes(Lambda, D)
        theta = 2.0 * pi * random.uniform(size=(D,))
        w = sqrt(s) * exp(1j * theta)
        Z[:, i] = V @ w

    return Z


def _sample_diagonal_complex_bingham_magnitudes(Lambda, D):
    """
    Sample squared magnitudes |z_i|^2 from a diagonal complex Bingham distribution.

    The resulting vector s satisfies s_i >= 0 and sum(s) = 1.

    Args:
        Lambda: D-dimensional non-negative eigenvalue vector (largest first, last=0).
        D (int): Dimension.

    Returns:
        ndarray: D-dimensional vector of squared magnitudes.
    """
    Lambda_pos = Lambda[: D - 1]  # first D-1 (positive) eigenvalues

    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "_sample_diagonal_complex_bingham_magnitudes is not supported on the JAX backend."
        )

    # Precompute for the truncated exponential inverse CDF
    large = Lambda_pos >= 0.03
    safe_lambda = where(large, Lambda_pos, 1.0)
    temp1 = where(large, -1.0 / safe_lambda, 0.0)
    temp2 = where(large, 1.0 - exp(-Lambda_pos), 0.0)

    s = zeros(D)
    while True:
        U = random.uniform(size=(D - 1,))
        if any(large):
            s[: D - 1][large] = temp1[large] * log(1.0 - U[large] * temp2[large])
        if any(~large):
            s[: D - 1][~large] = U[~large]

        if sum(s[: D - 1]) < 1.0:
            break

    s[D - 1] = 1.0 - sum(s[: D - 1])
    return s
