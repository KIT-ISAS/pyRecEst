# pylint: disable=no-name-in-module
"""Complex Bingham Distribution.

Ported from the MATLAB libDirectional library:
  ComplexBinghamDistribution.m  (libDirectional/lib/distributions/complexHypersphere/)

Reference:
  Kent, J. T. "The Complex Bingham Distribution and Shape Analysis."
  Journal of the Royal Statistical Society. Series B (Methodological), 1994, 285-299.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares


class ComplexBinghamDistribution:
    """Complex Bingham distribution on the complex unit hypersphere.

    The distribution is defined on the complex unit sphere
    S^{2d-1} = {z ∈ C^d : ‖z‖ = 1} with pdf

        p(z) ∝ exp(z^H B z),

    where B is a d×d Hermitian parameter matrix.

    Attributes
    ----------
    B : numpy.ndarray, shape (d, d), dtype complex128
        Hermitian parameter matrix.
    dim : int
        Complex dimension d.
    log_norm_const : float
        Negative log normalization constant (i.e. −log C(B) where C(B) is
        the normalising constant), stored so that
        pdf(z) = exp(log_norm_const + Re(z^H B z)).
    """

    def __init__(self, B):
        """Construct a ComplexBinghamDistribution.

        Parameters
        ----------
        B : array_like, shape (d, d)
            Hermitian parameter matrix (B == B^H must hold).
        """
        B = np.asarray(B, dtype=complex)
        assert np.allclose(B, B.conj().T, atol=1e-10), "B must be Hermitian"
        self.B = B
        self.dim = B.shape[0]
        self.log_norm_const = ComplexBinghamDistribution.log_norm(B)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def pdf(self, z):
        """Evaluate the pdf at one or more points on the complex unit sphere.

        Parameters
        ----------
        z : array_like, shape (d,) or (d, n)
            Each column (or the single vector) is a point on the complex unit
            sphere.

        Returns
        -------
        numpy.ndarray, shape (n,) or float
            Pdf value(s).
        """
        z = np.asarray(z, dtype=complex)
        single = z.ndim == 1
        if single:
            z = z[:, np.newaxis]
        # Re(z^H B z) for each column
        Bz = self.B @ z  # (d, n)
        vals = np.real(np.einsum("ij,ij->j", z.conj(), Bz))  # shape (n,)
        p = np.exp(self.log_norm_const + vals)
        return float(p[0]) if single else p

    def sample(self, n):
        """Draw samples from the complex Bingham distribution.

        Uses the rejection-sampling algorithm of
        Kent, Constable & Er (2004) "Simulation for the complex Bingham
        distribution", *Statistics and Computing*, 14, 53-57.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        numpy.ndarray, shape (d, n), dtype complex128
            Sampled unit vectors (each column has unit norm).
        """
        d = self.dim
        if d < 2:
            raise ValueError("Sampling requires d >= 2.")

        # Eigendecomposition of -B (eigenvectors V, eigenvalues Λ of -B)
        eigenvalues_neg, V = eigh(-self.B)  # sorted ascending by scipy
        # Sort descending
        idx = np.argsort(eigenvalues_neg)[::-1]
        eigenvalues_neg = eigenvalues_neg[idx]
        V = V[:, idx]

        # Shift so the last (smallest) eigenvalue of -B becomes 0
        Lambda = eigenvalues_neg - eigenvalues_neg[-1]  # Λ ≥ 0, Λ[-1] = 0

        # Precompute for truncated-exponential CDF inversion
        Lam = Lambda[:-1]  # shape (d-1,)

        samples = np.zeros((d, n), dtype=complex)
        for k in range(n):
            # Rejection loop
            while True:
                S = np.zeros(d)
                U = np.random.uniform(size=d - 1)
                for i in range(d - 1):
                    if Lambda[i] < 0.03:
                        # Nearly-uniform truncated exponential
                        S[i] = U[i]
                    else:
                        S[i] = -(1.0 / Lam[i]) * np.log(
                            1.0 - U[i] * (1.0 - np.exp(-Lam[i]))
                        )
                if S[:-1].sum() < 1.0:
                    break
            S[-1] = 1.0 - S[:-1].sum()

            # Random phases
            theta = 2.0 * np.pi * np.random.uniform(size=d)
            W = np.sqrt(S) * np.exp(1j * theta)
            samples[:, k] = V @ W

        return samples

    # ------------------------------------------------------------------
    # Static / class methods
    # ------------------------------------------------------------------

    @staticmethod
    def log_norm(B):
        """Compute the *negative* log normalization constant −log C(B).

        The formula used is (Kent 1994, eq. 2.3) after eigenvalue shift:

            C(λ) = 2π^d · Σ_j exp(λ_j) / Π_{k≠j}(λ_j − λ_k)

        For nearly-equal eigenvalues a small perturbation is applied so that
        the Vandermonde denominators remain well-conditioned, matching the
        approach in the original MATLAB implementation.

        Parameters
        ----------
        B : array_like, shape (d, d)
            Hermitian parameter matrix.

        Returns
        -------
        float
            Negative log normalization constant.
        """
        B = np.asarray(B, dtype=complex)
        d = B.shape[0]

        # Real eigenvalues of a Hermitian matrix
        eigenvalues = np.linalg.eigvalsh(B)  # sorted ascending

        # Shift so the maximum eigenvalue is 0
        eigenvalue_shift = float(eigenvalues[-1])
        eigenvalues = eigenvalues - eigenvalue_shift

        # Perturb near-equal eigenvalues for numerical stability
        eigenvalues = ComplexBinghamDistribution._perturb_eigenvalues(eigenvalues)

        if np.all(np.abs(eigenvalues) < 1e-3):
            # All eigenvalues near zero: use limiting uniform value C = 2π^d / (d-1)!
            log_C_shifted = np.log(2.0) + d * np.log(np.pi) - float(
                np.sum(np.log(np.arange(1, d)))
            )
        else:
            # Analytical formula: C = 2π^d · Σ_j exp(λ_j) / Π_{k≠j}(λ_j - λ_k)
            log_C_shifted = ComplexBinghamDistribution._log_norm_from_eigenvalues(
                eigenvalues
            )

        # Apply eigenvalue-shift correction: C(λ) = exp(shift) · C(λ - shift)
        log_C = log_C_shifted + eigenvalue_shift

        return -log_C  # inverted convention: log_norm_const = -log C

    @staticmethod
    def _log_norm_from_eigenvalues(eigenvalues):
        """Log normalization for shifted eigenvalues (max = 0) via partial fractions.

        Computes log(2π^d · Σ_j exp(λ_j) / Π_{k≠j}(λ_j - λ_k)).
        """
        d = len(eigenvalues)
        log_prefix = np.log(2.0) + d * np.log(np.pi)

        # For each j compute sign_j * exp(log_term_j) where
        #   log_term_j = λ_j - Σ_{k≠j} log|λ_j - λ_k|
        log_terms = np.empty(d)
        signs = np.empty(d)
        for j in range(d):
            diffs = eigenvalues[j] - np.delete(eigenvalues, j)
            signs[j] = np.prod(np.sign(diffs))
            log_terms[j] = eigenvalues[j] - np.sum(np.log(np.abs(diffs)))

        # log(|Σ_j sign_j · exp(log_term_j)|) via log-sum-exp
        max_log = np.max(log_terms)
        scaled = np.sum(signs * np.exp(log_terms - max_log))
        log_sum = max_log + np.log(np.abs(scaled))

        return log_prefix + log_sum

    @staticmethod
    def _perturb_eigenvalues(eigenvalues):
        """Sort eigenvalues descending and enforce minimum spacing of 0.01.

        Mirrors MATLAB's ``makeSureEigenvaluesAreNotTooClose``.
        """
        lam = np.sort(eigenvalues)[::-1].copy()
        diffs = np.diff(lam)  # non-positive for sorted-descending
        diffs = np.minimum(diffs, -0.01)  # enforce gap ≥ 0.01
        lam[1:] = lam[0] + np.cumsum(diffs)
        return lam

    @classmethod
    def fit(cls, Z):
        """Maximum-likelihood fit of a complex Bingham distribution to data.

        Parameters
        ----------
        Z : array_like, shape (d, n)
            Complex unit vectors (columns); it is assumed that ‖z_i‖ = 1.

        Returns
        -------
        ComplexBinghamDistribution
        """
        Z = np.asarray(Z, dtype=complex)
        n = Z.shape[1]
        S = Z @ Z.conj().T / n  # sample scatter matrix
        B = cls._estimate_parameter_matrix(S)
        return cls(B)

    @staticmethod
    def _estimate_parameter_matrix(S):
        """Compute the ML estimate of B from the scatter matrix S.

        The eigenvectors of B equal those of S.  The eigenvalues of B are
        found by solving the moment-matching equations

            ∂ log C(λ) / ∂λ_k  =  s_k,  k = 1, …, d

        where s_k are the eigenvalues of S, using a least-squares solver
        with finite-difference gradients.

        Parameters
        ----------
        S : numpy.ndarray, shape (d, d), complex Hermitian
            Sample scatter matrix (E[z z^H] estimate).

        Returns
        -------
        numpy.ndarray, shape (d, d), complex Hermitian
            Estimated parameter matrix B.
        """
        d = S.shape[0]
        eigenvalues_S, V = eigh(S)  # ascending

        def grad_log_C(lam):
            """Numerical gradient of log C via forward finite differences."""
            eps = 1e-6
            B_diag = np.diag(lam.astype(complex))
            log_c0 = ComplexBinghamDistribution.log_norm(B_diag)
            grad = np.empty(d)
            for i in range(d):
                lam_p = lam.copy()
                lam_p[i] += eps
                log_cp = ComplexBinghamDistribution.log_norm(
                    np.diag(lam_p.astype(complex))
                )
                # log_norm_const = -log C, so d(log C)/dλ_i = -d(log_norm_const)/dλ_i
                grad[i] = (-log_cp - (-log_c0)) / eps
            return grad

        # Initial guess: spread eigenvalues below zero
        x0 = np.linspace(-(d - 1) * 10, -10, d - 1)

        def residuals(x):
            lam = np.append(x, 0.0)
            return grad_log_C(lam) - eigenvalues_S

        result = least_squares(
            residuals,
            x0,
            method="lm",
            ftol=1e-15,
            xtol=1e-10,
            max_nfev=int(1e4),
        )
        lam_B = np.append(result.x, 0.0)
        B = V @ np.diag(lam_B.astype(complex)) @ V.conj().T
        B = 0.5 * (B + B.conj().T)  # enforce exact Hermitian symmetry
        return B

    @staticmethod
    def cauchy_schwarz_divergence(cB1, cB2):
        """Cauchy-Schwarz divergence between two complex Bingham distributions.

        D_CS(p, q) = log C(B₁+B₂) − ½[log C(2B₁) + log C(2B₂)]

        Parameters
        ----------
        cB1, cB2 : ComplexBinghamDistribution or numpy.ndarray
            Distributions or parameter matrices.

        Returns
        -------
        float
        """
        if isinstance(cB1, ComplexBinghamDistribution):
            B1 = cB1.B
        else:
            B1 = np.asarray(cB1, dtype=complex)
        if isinstance(cB2, ComplexBinghamDistribution):
            B2 = cB2.B
        else:
            B2 = np.asarray(cB2, dtype=complex)

        assert np.allclose(B1, B1.conj().T, atol=1e-10), "B1 must be Hermitian"
        assert np.allclose(B2, B2.conj().T, atol=1e-10), "B2 must be Hermitian"

        log_c1 = ComplexBinghamDistribution.log_norm(2.0 * B1)
        log_c2 = ComplexBinghamDistribution.log_norm(2.0 * B2)
        log_c3 = ComplexBinghamDistribution.log_norm(B1 + B2)

        return log_c3 - 0.5 * (log_c1 + log_c2)
