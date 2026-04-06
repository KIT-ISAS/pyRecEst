import numpy as np
from scipy.special import factorial

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class ComplexBinghamDistribution(AbstractHypersphericalDistribution):
    """Complex Bingham distribution on the complex unit hypersphere.

    Reference:
        Kent, J. T.
        The Complex Bingham Distribution and Shape Analysis.
        Journal of the Royal Statistical Society. Series B (Methodological),
        JSTOR, 1994, 285-299.
    """

    def __init__(self, B):
        """
        Parameters
        ----------
        B : array_like, shape (d, d)
            Hermitian parameter matrix.
        """
        B = np.asarray(B, dtype=complex)
        if not np.allclose(B, B.conj().T):
            raise ValueError("B must be Hermitian")

        d = B.shape[0]
        # The complex unit sphere in C^d is the real (2d-1)-dimensional sphere
        # S^{2d-1}.  AbstractHypersphericalDistribution expects the dimension of
        # the sphere (i.e., 2d-2 for S^{2d-1}).
        AbstractHypersphericalDistribution.__init__(self, 2 * (d - 1))

        self.B = B
        self.d = d  # complex dimension
        self.log_norm_const = ComplexBinghamDistribution.log_norm(B)
        self.norm_const = np.exp(-self.log_norm_const)  # = integral = normalisation denominator

    # ------------------------------------------------------------------
    # Core density
    # ------------------------------------------------------------------

    def pdf(self, za):
        """Evaluate the pdf at complex unit vectors.

        Parameters
        ----------
        za : array_like, shape (d,) or (d, n)
            Complex unit vectors (each column / the vector should have unit
            norm).

        Returns
        -------
        p : ndarray, shape () or (n,)
            Probability density values.
        """
        za = np.asarray(za, dtype=complex)
        if za.ndim == 1:
            val = float(np.real(za.conj() @ self.B @ za))
            return float(np.exp(self.log_norm_const + val))
        # za shape: (d, n)
        vals = np.real(np.einsum("in,ij,jn->n", za.conj(), self.B, za))
        return np.exp(self.log_norm_const + vals)

    # ------------------------------------------------------------------
    # Sampling  (Kent, Constable & Er 2004)
    # ------------------------------------------------------------------

    def sample(self, n):
        """Sample from the complex Bingham distribution.

        Implements the algorithm from:
            Kent, J. T.; Constable, P. D. L. & Er, F.
            Simulation for the complex Bingham distribution.
            Statistics and Computing, Springer, 2004, 14, 53-57.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        Z : ndarray, shape (d, n), dtype complex
            Sampled complex unit vectors arranged as columns.
        """
        P = self.d
        if P < 2:
            raise ValueError("Matrix B is too small (need d >= 2).")

        # Eigendecompose -B to get non-negative eigenvalues (mode has lam=0)
        eigenvalues, V = np.linalg.eigh(-self.B)
        # Sort eigenvalues descending and permute eigenvectors accordingly
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]

        # Shift so the smallest eigenvalue becomes 0
        eigenvalues = eigenvalues - eigenvalues[-1]

        # Pre-compute helpers for truncated-exponential inversion
        lam = eigenvalues[:-1]  # length P-1
        small = np.abs(lam) < 0.03  # use uniform approximation for tiny lambda
        with np.errstate(divide="ignore", invalid="ignore"):
            temp1 = np.where(small, np.ones_like(lam), -1.0 / lam)
            temp2 = np.where(small, np.ones_like(lam), 1.0 - np.exp(-lam))

        samples = np.zeros((P, n), dtype=complex)
        rng = np.random.default_rng()

        for k in range(n):
            while True:
                U = rng.random(P - 1)
                S = np.zeros(P)
                for i in range(P - 1):
                    if small[i]:
                        S[i] = U[i]
                    else:
                        S[i] = temp1[i] * np.log(1.0 - U[i] * temp2[i])
                if S[: P - 1].sum() < 1.0:
                    break
            S[P - 1] = 1.0 - S[: P - 1].sum()

            theta = 2.0 * np.pi * rng.random(P)
            W = np.sqrt(np.maximum(S, 0.0)) * np.exp(1j * theta)
            samples[:, k] = V @ W

        return samples

    # ------------------------------------------------------------------
    # Normalization constant
    # ------------------------------------------------------------------

    @staticmethod
    def log_norm(B):
        """Compute the log-normalization constant log C(B).

        Uses the eigenvalue-shift method from Kent (1994), eq. (2.3).
        For the near-uniform case (all |eigenvalues| < 1e-3) the surface-area
        formula is used directly.

        The sign convention matches the MATLAB source: the returned value is
        the *positive* log of the normalising integral, so the pdf can be
        written as  p(z) = exp(log_norm_const + Re(z^H B z)).

        Parameters
        ----------
        B : array_like, shape (d, d) or (d, d, K)
            Hermitian parameter matrix or stack of K matrices.

        Returns
        -------
        log_c : float or ndarray, shape (K,)
        """
        B = np.asarray(B, dtype=complex)
        if B.ndim == 2:
            single = True
            B = B[:, :, np.newaxis]
        else:
            single = False

        D = B.shape[0]
        K = B.shape[2]
        log_c = np.zeros(K)

        surface_area = 2.0 * np.pi**D / float(factorial(D - 1))

        for k in range(K):
            Bk = B[:, :, k]
            eigenvalues = np.linalg.eigvalsh(Bk)
            shift = float(eigenvalues.max())
            eigenvalues = np.sort(eigenvalues) - shift  # max is now 0

            if np.all(np.abs(eigenvalues) < 1e-3):
                log_c[k] = np.log(surface_area)
            else:
                eigenvalues = ComplexBinghamDistribution._fix_eigenvalues(eigenvalues)
                val = ComplexBinghamDistribution._norm_integrand(eigenvalues, D)
                log_c[k] = np.log(float(val))

            # Undo the eigenvalue shift (Kent 1994 p. 286)
            log_c[k] += shift

        # Negate: the convention is to store -log(integral) so that
        # pdf can be written as exp(log_norm_const + Re(z^H B z)).
        if single:
            return float(-log_c[0])
        return -log_c

    @staticmethod
    def _norm_integrand(eigenvalues, D):
        """Evaluate the normalization integral for (shifted) eigenvalues.

        Computes  ∫_{S^{2D-1}} exp(∑_j λ_j |z_j|²) dσ(z)
        using the exact formula (valid for distinct eigenvalues):

            c(λ) = 2π^D · Σ_j  exp(λ_j) / Π_{k≠j}(λ_j − λ_k)

        which follows from the Dirichlet(1,…,1) marginal of uniform z on S^{2D-1}
        and the partial-fraction decomposition of the simplex integral.
        """
        lam = np.asarray(eigenvalues, dtype=float)
        prefix = 2.0 * np.pi**D
        total = 0.0
        for j in range(D):
            denom = 1.0
            for k in range(D):
                if k != j:
                    denom *= lam[j] - lam[k]
            total += np.exp(lam[j]) / denom
        return float(prefix * total)

    @staticmethod
    def _fix_eigenvalues(eigenvalues):
        """Ensure adjacent eigenvalues differ by at least 1e-2 (avoids singularities)."""
        eigenvalues = np.sort(eigenvalues)[::-1].copy()  # descending
        for i in range(len(eigenvalues) - 1):
            if eigenvalues[i + 1] - eigenvalues[i] > -1e-2:
                eigenvalues[i + 1] = eigenvalues[i] - 1e-2
        return eigenvalues

    # ------------------------------------------------------------------
    # Utility / static methods
    # ------------------------------------------------------------------

    @staticmethod
    def unit_sphere_surface(d):
        """Surface area of the unit complex sphere in C^d, i.e., S^{2d-1}.

        Parameters
        ----------
        d : int
            Complex dimension.

        Returns
        -------
        float
        """
        return 2.0 * float(np.pi**d) / float(factorial(d - 1))

    def integral(self, n_samples=100_000):
        """Estimate the normalisation integral via Monte Carlo (should be 1).

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        float
        """
        rng = np.random.default_rng()
        Z = rng.standard_normal((self.d, n_samples)) + 1j * rng.standard_normal(
            (self.d, n_samples)
        )
        Z /= np.sqrt(np.sum(np.abs(Z) ** 2, axis=0, keepdims=True))
        p = self.pdf(Z)
        surface = ComplexBinghamDistribution.unit_sphere_surface(self.d)
        return float(np.mean(p) * surface)

    @staticmethod
    def cauchy_schwarz_divergence(cB1, cB2):
        """Cauchy-Schwarz divergence between two complex Bingham distributions.

        D_{CS}(f_1 || f_2) = log C(B_1+B_2) - 1/2*(log C(2B_1) + log C(2B_2))

        Parameters
        ----------
        cB1, cB2 : ComplexBinghamDistribution or array_like
            Distributions or Hermitian parameter matrices.

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

        if not np.allclose(B1, B1.conj().T):
            raise ValueError("B1 must be Hermitian")
        if not np.allclose(B2, B2.conj().T):
            raise ValueError("B2 must be Hermitian")

        log_c1 = ComplexBinghamDistribution.log_norm(2.0 * B1)
        log_c2 = ComplexBinghamDistribution.log_norm(2.0 * B2)
        log_c3 = ComplexBinghamDistribution.log_norm(B1 + B2)
        return float(log_c3 - 0.5 * (log_c1 + log_c2))

    @staticmethod
    def fit(Z):
        """Fit a ComplexBinghamDistribution to data via maximum likelihood.

        Parameters
        ----------
        Z : array_like, shape (d, n)
            Complex unit vectors arranged as columns.

        Returns
        -------
        ComplexBinghamDistribution
        """
        Z = np.asarray(Z, dtype=complex)
        n = Z.shape[1]
        S = Z @ Z.conj().T / n
        B = ComplexBinghamDistribution.estimate_parameter_matrix(S)
        return ComplexBinghamDistribution(B)

    @staticmethod
    def estimate_parameter_matrix(S):
        """ML estimate of the parameter matrix B from the scatter matrix S.

        The eigenvectors of S become the eigenvectors of B.  The eigenvalues
        of B are found numerically by matching the gradient of log C(B) to the
        eigenvalues of S (ML condition, Kent 1994 eq. 3.3).

        Parameters
        ----------
        S : array_like, shape (d, d)
            Positive semi-definite Hermitian scatter matrix (e.g. Z Z^H / N).

        Returns
        -------
        B : ndarray, shape (d, d), dtype complex
        """
        from scipy.optimize import least_squares  # pylint: disable=import-error

        S = np.asarray(S, dtype=complex)
        D = S.shape[0]

        eigenvalues_S, eigenvectors_S = np.linalg.eigh(S)
        target = eigenvalues_S.real  # ML target

        def residual(x):
            lam = np.append(x, 0.0)
            try:
                val = ComplexBinghamDistribution._log_norm_gradient(lam)
            except Exception:  # pylint: disable=broad-except
                return np.ones(D - 1) * 1e6
            return val[:-1] - target[:-1]

        x0 = 100.0 * (-D + 1 + np.arange(D - 1)) / D
        result = least_squares(residual, x0, bounds=(-1e3, -1e-2))
        eigenvalues_B = np.append(result.x, 0.0)

        B = eigenvectors_S @ np.diag(eigenvalues_B) @ eigenvectors_S.conj().T
        B = 0.5 * (B + B.conj().T)  # enforce Hermitian
        return B

    @staticmethod
    def _log_norm_gradient(eigenvalues):
        """Numerical gradient of log C w.r.t. eigenvalues (finite differences)."""
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        D = len(eigenvalues)
        grad = np.zeros(D)
        eps = 1e-3
        for i in range(D):
            ev_plus = eigenvalues.copy()
            ev_plus[i] += eps
            ev_minus = eigenvalues.copy()
            ev_minus[i] -= eps
            B_plus = np.diag(ev_plus).astype(complex)
            B_minus = np.diag(ev_minus).astype(complex)
            grad[i] = (
                ComplexBinghamDistribution.log_norm(B_plus)
                - ComplexBinghamDistribution.log_norm(B_minus)
            ) / (2.0 * eps)
        return grad
