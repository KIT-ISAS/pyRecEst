import math
import numpy as np
from scipy.optimize import least_squares

from .abstract_complex_hyperspherical_distribution import (
    AbstractComplexHypersphericalDistribution,
)


class ComplexBinghamDistribution(AbstractComplexHypersphericalDistribution):
    """
    Complex Bingham distribution on the complex unit hypersphere in C^d.

    The pdf is
        f(z) = exp(log_c + Re(z^H B z)),  ||z|| = 1
    where B is a Hermitian d×d parameter matrix and log_c = -log C(B) is the
    (negative) log of the normalization integral C(B).

    References
    ----------
    Kent, J. T.
        The Complex Bingham Distribution and Shape Analysis.
        J. Royal Statistical Society B, 56(2):285-299, 1994.

    Kent, J. T.; Constable, P. D. L. & Er, F.
        Simulation for the complex Bingham distribution.
        Statistics and Computing, 14:53-57, 2004.
    """

    def __init__(self, B):
        """
        Parameters
        ----------
        B : array_like, shape (d, d), complex or real
            Hermitian parameter matrix.  The complex dimension d is inferred
            from B.
        """
        B = np.asarray(B, dtype=complex)
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise ValueError("B must be a square matrix.")
        if not np.allclose(B, B.conj().T, atol=1e-10):
            raise ValueError("B must be Hermitian (B == B^H).")

        d = B.shape[0]
        super().__init__(d)

        self.B = B
        self.log_c = ComplexBinghamDistribution.log_norm(self.B)

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def pdf(self, za):
        """
        Evaluate the pdf at the given point(s) on the complex unit sphere.

        Parameters
        ----------
        za : array_like, shape (d,) or (d, n)
            Each column is a unit vector in C^d.

        Returns
        -------
        p : ndarray, shape () or (n,)
            Real-valued probability densities.
        """
        za = np.asarray(za, dtype=complex)
        single = za.ndim == 1
        if single:
            za = za[:, np.newaxis]

        # z^H B z is real for Hermitian B and unit vectors z; abs removes
        # tiny imaginary residuals from floating-point arithmetic
        val = np.einsum("id,ij,jd->d", za.conj(), self.B, za).real
        p = np.exp(self.log_c + val)
        return p[0] if single else p

    # ------------------------------------------------------------------
    # Sampling  (Kent, Constable & Er, 2004)
    # ------------------------------------------------------------------

    def sample(self, n):
        """
        Draw n i.i.d. samples from the complex Bingham distribution.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        Z : ndarray, shape (d, n), complex
            Columns are unit vectors in C^d.
        """
        d = self.complex_dim
        if d < 2:
            raise ValueError("B must be at least 2×2 for sampling.")

        # Eigendecomposition of -B; eigh returns ascending eigenvalues
        eigenvalues, V = np.linalg.eigh(-self.B)

        # Sort eigenvalues in *descending* order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]

        # Eigenvalue shift so that the smallest (last) eigenvalue is 0
        eigenvalues = eigenvalues - eigenvalues[-1]

        # Pre-compute constants for the truncated-exponential sampler
        lam = eigenvalues[:-1]  # length d-1, all >= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            temp1 = np.where(lam > 0, -1.0 / lam, 0.0)
            temp2 = np.where(lam > 0, 1.0 - np.exp(-lam), 0.0)

        Z = np.empty((d, n), dtype=complex)
        rng = np.random.default_rng()

        for k in range(n):
            # Sample simplex coordinate S via rejection
            while True:
                U = rng.uniform(0.0, 1.0, d - 1)
                S = np.where(
                    lam < 0.03,
                    U,
                    temp1 * np.log(1.0 - U * temp2),
                )
                if S.sum() < 1.0:
                    break

            S_full = np.empty(d)
            S_full[:-1] = S
            S_full[-1] = 1.0 - S.sum()

            # Independent uniform phases
            theta = 2.0 * np.pi * rng.uniform(0.0, 1.0, d)
            W = np.sqrt(S_full) * np.exp(1j * theta)

            Z[:, k] = V @ W

        return Z

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def fit(Z):
        """
        Fit a ComplexBinghamDistribution to complex samples.

        Parameters
        ----------
        Z : array_like, shape (d, N), complex
            N sample unit vectors in C^d.

        Returns
        -------
        ComplexBinghamDistribution
        """
        Z = np.asarray(Z, dtype=complex)
        N = Z.shape[1]
        S = Z @ Z.conj().T / N
        B = ComplexBinghamDistribution.estimate_parameter_matrix(S)
        return ComplexBinghamDistribution(B)

    @staticmethod
    def estimate_parameter_matrix(S):
        """
        Maximum-likelihood estimate of B from a scatter matrix S.

        The ML eigenvectors of B are the eigenvectors of S.  The eigenvalues
        κ_1 ≤ ... ≤ κ_{d-1} < κ_d = 0 are found by solving the score
        equations  E_κ[|z_k|²] = s_k  (where s_k = eigenvalue_k(S)),
        i.e. the expected scatter equals the sample scatter.

        Parameters
        ----------
        S : array_like, shape (d, d), Hermitian positive semi-definite
            Sample scatter matrix  S = (1/N) Σ z_i z_i^H.

        Returns
        -------
        B : ndarray, shape (d, d), complex Hermitian
        """
        S = np.asarray(S, dtype=complex)
        d = S.shape[0]

        # Eigendecompose S (eigh gives ascending order, real eigenvalues)
        s_vals, V_S = np.linalg.eigh(S)
        # s_vals are in ascending order; target proportions
        s_total = s_vals.sum()
        target = s_vals / s_total if s_total > 0 else np.ones(d) / d

        def _expected_scatter(kappa):
            """E_κ[|z_k|²] via numerical derivative of log-integral."""
            h = 1e-5
            grad = np.zeros(d)
            for i in range(d):
                kp = kappa.copy()
                km = kappa.copy()
                kp[i] += h
                km[i] -= h
                B_p = V_S @ np.diag(kp.astype(complex)) @ V_S.conj().T
                B_m = V_S @ np.diag(km.astype(complex)) @ V_S.conj().T
                # log_norm returns -log(I), so -log_norm = log(I)
                log_I_p = -ComplexBinghamDistribution.log_norm(B_p)
                log_I_m = -ComplexBinghamDistribution.log_norm(B_m)
                grad[i] = (log_I_p - log_I_m) / (2 * h)
            return grad

        def residual(kappa_free):
            kappa = np.append(kappa_free, 0.0)
            expected = _expected_scatter(kappa)
            exp_total = expected.sum()
            if exp_total > 0:
                expected = expected / exp_total
            return expected[:-1] - target[:-1]

        # Initial guess: push smaller eigenvalues to more negative values
        x0 = -(target[-1] - target[:-1]) * 10.0

        try:
            result = least_squares(
                residual,
                x0,
                bounds=(np.full(d - 1, -1000.0), np.full(d - 1, -1e-2)),
                method="trf",
                ftol=1e-12,
                xtol=1e-12,
                max_nfev=int(1e5),
            )
            kappa = np.append(result.x, 0.0)
        except Exception:  # pylint: disable=broad-except
            kappa = np.zeros(d)

        # Sort and construct B
        idx = np.argsort(kappa)
        kappa = kappa[idx]
        V = V_S[:, idx]
        B = V @ np.diag(kappa.astype(complex)) @ V.conj().T
        B = 0.5 * (B + B.conj().T)
        return B

    @staticmethod
    def log_norm(B, variant="analytical", n_mc=100_000):
        """
        Compute the (negative) log normalization constant.

        ``log_c = -log C(B)``  where
        ``C(B) = ∫_{||z||=1} exp(z^H B z) dz``.

        The pdf satisfies  f(z) = exp(log_c + Re(z^H B z)).

        Parameters
        ----------
        B : array_like, shape (d, d), Hermitian
        variant : {'analytical', 'monte_carlo'}
            'analytical' (default): uses the divided-differences formula
            (exact for distinct eigenvalues; near-duplicate eigenvalues are
            perturbed slightly).
            'monte_carlo': Monte Carlo estimate.
        n_mc : int
            Number of Monte Carlo samples (only used when variant='monte_carlo').

        Returns
        -------
        float
            Negative log of the normalization integral.
        """
        B = np.asarray(B, dtype=complex)
        d = B.shape[0]

        if variant == "monte_carlo":
            log_int = ComplexBinghamDistribution._log_norm_monte_carlo(B, n_mc)
            return -log_int

        # Eigenvalue shift for numerical stability
        eigenvalues = np.linalg.eigvalsh(B).real
        shift = float(eigenvalues.max())
        eigenvalues = eigenvalues - shift

        # Nearly uniform case: all eigenvalues ≈ 0
        if np.all(np.abs(eigenvalues) < 1e-3):
            log_int = np.log(2.0 * np.pi**d / math.factorial(d - 1))
            return -(log_int + shift)

        eigenvalues = ComplexBinghamDistribution._make_distinct(eigenvalues)
        log_int_shifted = ComplexBinghamDistribution._log_norm_analytical(
            eigenvalues, d
        )
        # Correct for eigenvalue shift: C(B) = exp(shift) * C(B - shift*I)
        log_int = log_int_shifted + shift
        return -log_int

    # ------------------------------------------------------------------
    # Internal helpers for log_norm
    # ------------------------------------------------------------------

    @staticmethod
    def _log_norm_analytical(eigenvalues, d):
        """
        log C from the divided-differences (partial-fractions) formula:

            C(κ) = 2π^d · Σ_k exp(κ_k) / ∏_{j≠k} (κ_k - κ_j)

        All eigenvalues must be distinct.
        """
        total = 0.0
        for k in range(d):
            denom = 1.0
            for j in range(d):
                if j != k:
                    denom *= eigenvalues[k] - eigenvalues[j]
            total += np.exp(float(eigenvalues[k])) / denom

        log_c = np.log(2.0 * np.pi**d * total)
        return float(log_c)

    @staticmethod
    def _make_distinct(eigenvalues, min_gap=1e-2):
        """
        Perturb eigenvalues (in descending order) so that consecutive pairs
        differ by at least ``min_gap``.  Mirrors the MATLAB workaround
        ``makeSureEigenvaluesAreNotTooClose``.
        """
        lam = np.sort(eigenvalues)[::-1].copy()  # descending
        for i in range(len(lam) - 1):
            if lam[i] - lam[i + 1] < min_gap:
                lam[i + 1] = lam[i] - min_gap
        return lam  # keep descending order (sign of products handled in formula)

    @staticmethod
    def _log_norm_monte_carlo(B, n=100_000):
        """Monte Carlo estimate of log C(B)."""
        d = B.shape[0]
        volume = 2.0 * np.pi**d / math.factorial(d - 1)
        rng = np.random.default_rng()
        Z = rng.standard_normal((d, n)) + 1j * rng.standard_normal((d, n))
        Z /= np.linalg.norm(Z, axis=0, keepdims=True)
        val = np.einsum("id,ij,jd->d", Z.conj(), B, Z).real
        return float(np.log(np.mean(np.exp(val)) * volume))

    # ------------------------------------------------------------------
    # Cauchy-Schwarz divergence
    # ------------------------------------------------------------------

    @staticmethod
    def cauchy_schwarz_divergence(cB1, cB2):
        """
        Cauchy-Schwarz divergence between two complex Bingham distributions.

        D_CS = log_c(B1+B2) - ½ (log_c(2·B1) + log_c(2·B2))

        where log_c uses the sign convention log_c = -log C(B).

        Parameters
        ----------
        cB1, cB2 : ComplexBinghamDistribution or array_like
            Distributions (or parameter matrices) to compare.

        Returns
        -------
        float
        """
        if isinstance(cB1, ComplexBinghamDistribution):
            B1, B2 = cB1.B, cB2.B
        else:
            B1 = np.asarray(cB1, dtype=complex)
            B2 = np.asarray(cB2, dtype=complex)

        log_c1 = ComplexBinghamDistribution.log_norm(2.0 * B1)
        log_c2 = ComplexBinghamDistribution.log_norm(2.0 * B2)
        log_c3 = ComplexBinghamDistribution.log_norm(B1 + B2)

        return float(log_c3 - 0.5 * (log_c1 + log_c2))
