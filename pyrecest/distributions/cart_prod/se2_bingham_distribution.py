# pylint: disable=redefined-builtin,no-name-in-module,no-member
import numpy as np
from scipy.integrate import quad
from scipy.special import iv

from pyrecest.backend import (
    array,
    column_stack,
    exp,
    pi,
    sum,
)

from ..abstract_se2_distribution import AbstractSE2Distribution
from ..hypersphere_subset.bingham_distribution import BinghamDistribution
from ..nonperiodic.custom_linear_distribution import CustomLinearDistribution


class SE2BinghamDistribution(AbstractSE2Distribution):
    """
    Distribution on SE(2) = S^1 x R^2.

    The density is f(x) = (1/NC) * exp(x^T * C * x) where x is the dual
    quaternion representation (first two components on S^1, last two
    components in R^2).

    C is a 4x4 symmetric matrix partitioned as::

        C = [ C1   C2^T ]
            [ C2   C3   ]

    where:
      - C1 (2x2): symmetric, controls the Bingham (rotational) part
      - C2 (2x2): coupling between rotation and translation
      - C3 (2x2): symmetric, negative-definite, controls the Gaussian (translational) part

    Reference:
    Igor Gilitschenski, Gerhard Kurz, Simon J. Julier, Uwe D. Hanebeck,
    "A New Probability Distribution for Simultaneous Representation of
    Uncertain Position and Orientation",
    Proceedings of the 17th International Conference on Information Fusion
    (Fusion 2014), Salamanca, Spain, July 2014.
    """

    def __init__(self, C, C2=None, C3=None):
        """
        Create an SE2BinghamDistribution.

        Parameters
        ----------
        C : array_like, shape (4, 4) or (2, 2)
            If C2 and C3 are not provided, this is the full 4x4 parameter
            matrix.  Otherwise it is the 2x2 Bingham (rotational) part C1.
        C2 : array_like, shape (2, 2), optional
            Coupling matrix between rotation and translation.
        C3 : array_like, shape (2, 2), optional
            Symmetric negative-definite matrix for the translational part.
        """
        AbstractSE2Distribution.__init__(self)

        assert (C2 is None) == (C3 is None), (
            "Either both C2 and C3 must be provided, or neither."
        )

        if C2 is None:
            assert C.shape == (4, 4), "C must be 4x4 when C2 and C3 are not provided."
            assert np.allclose(np.array(C), np.array(C).T), "Full C matrix must be symmetric."
            self.C = C
            self.C1 = C[:2, :2]
            self.C2 = C[2:, :2]
            self.C3 = C[2:, 2:]
        else:
            assert C.shape == (2, 2), "C1 must be 2x2."
            assert C2.shape == (2, 2), "C2 must be 2x2."
            assert C3.shape == (2, 2), "C3 must be 2x2."
            assert np.allclose(np.array(C), np.array(C).T), "C1 must be symmetric."
            assert np.allclose(np.array(C3), np.array(C3).T), "C3 must be symmetric."
            self.C1 = C
            self.C2 = C2
            self.C3 = C3
            self.C = column_stack(
                [
                    column_stack([self.C1, self.C2.T]).T,
                    column_stack([self.C2, self.C3]).T,
                ]
            ).T

        assert np.all(np.linalg.eigvalsh(np.array(self.C3)) <= 0), (
            "C3 must be negative semi-definite."
        )

        self._nc = None  # lazily computed

    @property
    def nc(self):
        """Normalization constant (lazily computed)."""
        if self._nc is None:
            self._nc = self._compute_nc()
        return self._nc

    def _compute_nc(self):
        """
        Compute the normalization constant.

        NC = 2*pi * sqrt(det(-0.5 * C3^{-1})) * F_bingham(Z_bm)

        where Z_bm are the eigenvalues of the Schur complement
        BM = C1 - C2^T * C3^{-1} * C2,
        and F_bingham is the 2D Bingham normalization constant
        F = 2*pi * exp((z1+z2)/2) * I_0((z2-z1)/2).
        """
        C1 = np.array(self.C1, dtype=float)
        C2 = np.array(self.C2, dtype=float)
        C3 = np.array(self.C3, dtype=float)
        C3_inv = np.linalg.inv(C3)
        bm = C1 - C2.T @ C3_inv @ C2
        z = np.sort(np.linalg.eigvalsh(bm))  # ascending
        # 2D Bingham normalization on S^1
        b_nc = 2.0 * np.pi * np.exp((z[0] + z[1]) / 2.0) * iv(0, (z[1] - z[0]) / 2.0)
        nc = 2.0 * np.pi * np.sqrt(np.linalg.det(-0.5 * C3_inv)) * b_nc
        return float(nc)

    def pdf(self, xa):
        """
        Evaluate the probability density at the given points.

        Parameters
        ----------
        xa : array_like, shape (N, 4) or (N, 3)
            Evaluation points in dual quaternion (N x 4) or angle-pos
            (N x 3) representation.

        Returns
        -------
        p : array, shape (N,)
            Density values.
        """
        xa = array(xa)
        if xa.ndim == 1:
            xa = xa.reshape(1, -1)
        if xa.shape[1] == 3:
            xa = AbstractSE2Distribution.angle_pos_to_dual_quaternion(xa)
        assert xa.shape[1] == 4, "Input must have 4 columns (dual quaternion)."
        return (1.0 / self.nc) * exp(sum(xa * (xa @ self.C.T), axis=1))

    def mode(self):
        """
        Compute one mode of the distribution.

        Because of antipodal symmetry, -mode is equally valid.

        Returns
        -------
        m : array, shape (4,)
            Mode in dual quaternion representation.
        """
        C1 = np.array(self.C1, dtype=float)
        C2 = np.array(self.C2, dtype=float)
        C3 = np.array(self.C3, dtype=float)
        C3_inv = np.linalg.inv(C3)
        bingham_c = C1 - C2.T @ C3_inv @ C2
        eigenvalues, eigenvectors = np.linalg.eigh(bingham_c)
        idx = int(np.argmax(eigenvalues))
        m_rot = eigenvectors[:, idx]
        m_lin = -C3_inv @ C2 @ m_rot
        return array(np.concatenate([m_rot, m_lin]))

    def sample(self, n):
        """
        Draw n samples from the distribution.

        Sampling uses a two-step procedure:
        1. Sample the rotational part from the Bingham marginal.
        2. Sample the translational part from the Gaussian conditional.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        s : array, shape (n, 4)
            Samples in dual quaternion representation.
        """
        assert n > 0, "n must be positive."
        C1 = np.array(self.C1, dtype=float)
        C2 = np.array(self.C2, dtype=float)
        C3 = np.array(self.C3, dtype=float)
        C3_inv = np.linalg.inv(C3)

        # Step 1: sample Bingham marginal via Schur complement eigendecomp
        bingham_c = C1 - C2.T @ C3_inv @ C2
        eigenvalues, eigenvectors = np.linalg.eigh(bingham_c)
        order = np.argsort(eigenvalues)  # ascending
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        # Shift so last entry is 0 (BinghamDistribution convention)
        z_shifted = array(eigenvalues - eigenvalues[-1])
        m_mat = array(eigenvectors)
        b = BinghamDistribution(z_shifted, m_mat)
        bingham_samples = b.sample(n)  # (n, 2)

        # Step 2: sample Gaussian conditional
        # mean_i = -C3^{-1} * C2 * x_rot_i
        means = (-C3_inv @ C2 @ np.array(bingham_samples).T).T  # (n, 2)
        # covariance = -0.5 * C3^{-1} (positive definite)
        cov = -0.5 * C3_inv
        lin_samples = array(
            np.array(means) + np.random.multivariate_normal(np.zeros(2), cov, size=n)
        )

        return column_stack([bingham_samples, lin_samples])

    def marginalize_linear(self):
        """
        Return the marginal distribution over the periodic (rotational) part.

        The marginal is the Bingham distribution corresponding to the Schur
        complement BM = C1 - C2^T * C3^{-1} * C2.

        Returns
        -------
        b : BinghamDistribution
            Marginal Bingham distribution on S^1.
        """
        C1 = np.array(self.C1, dtype=float)
        C2 = np.array(self.C2, dtype=float)
        C3 = np.array(self.C3, dtype=float)
        C3_inv = np.linalg.inv(C3)
        bm = C1 - C2.T @ C3_inv @ C2
        eigenvalues, eigenvectors = np.linalg.eigh(bm)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        z = array(eigenvalues - eigenvalues[-1])
        m = array(eigenvectors)
        return BinghamDistribution(z, m)

    def marginalize_periodic(self):
        """
        Return the marginal distribution over the linear (translational) part.

        The marginal is computed by numerically integrating out the rotational
        component.

        Returns
        -------
        dist : CustomLinearDistribution
            Marginal distribution over R^2.
        """
        C_np = np.array(self.C, dtype=float)
        nc = self.nc

        def _marginal_pdf(xs):
            xs = np.atleast_2d(xs)
            out = np.empty(xs.shape[0])
            for i, x_lin in enumerate(xs):
                # Integrate exp(x^T C x) over S^1 using the angle parametrisation
                def integrand(theta, xl=x_lin):
                    x_rot = np.array([np.cos(theta), np.sin(theta)])
                    x = np.concatenate([x_rot, xl])
                    return np.exp(float(x @ C_np @ x))

                val, _ = quad(integrand, 0.0, 2.0 * np.pi)
                out[i] = val / nc
            return array(out)

        return CustomLinearDistribution(_marginal_pdf, self.lin_dim)

    @staticmethod
    def fit(samples, weights=None):
        """
        Estimate SE2BinghamDistribution parameters from samples.

        Parameters
        ----------
        samples : array_like, shape (N, 4) or (N, 3)
            Samples in dual quaternion (N x 4) or angle-pos (N x 3) form.
        weights : array_like, shape (N,), optional
            Non-negative weights (need not sum to 1).  Defaults to uniform.

        Returns
        -------
        dist : SE2BinghamDistribution
            Fitted distribution.
        """
        samples = np.array(samples, dtype=float)
        if samples.shape[1] == 3:
            samples = np.array(
                AbstractSE2Distribution.angle_pos_to_dual_quaternion(array(samples))
            )
        assert samples.shape[1] == 4

        n = samples.shape[0]
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = np.array(weights, dtype=float)
            weights = weights / weights.sum()

        s_rot = samples[:, :2]  # (N, 2)
        s_lin = samples[:, 2:]  # (N, 2)

        # Weighted scatter matrix for rotational part
        w = weights[:, np.newaxis]
        scatter_rot = (s_rot * w).T @ s_rot  # (2, 2)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter_rot)
        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        # Bingham parameters from scatter: Z_i = log(eigenvalues_i) up to constant;
        # use the standard moment-based approach: C1 - C2'C3^{-1}C2 = M diag(Z) M'
        z_bingham = eigenvalues - eigenvalues[-1]
        tmp = eigenvectors @ np.diag(z_bingham) @ eigenvectors.T  # approx C1 - C2'C3^{-1}C2

        # Weighted regression for Gaussian part (Anderson 2003, Th. 8.2.1)
        reg_c = (s_lin * w).T @ s_rot  # (2, 2)
        reg_a = (s_rot * w).T @ s_rot  # (2, 2)
        # Use pinv for numerical stability in case reg_a is nearly singular (e.g. few samples)
        reg_beta = reg_c @ np.linalg.pinv(reg_a)  # (2, 2) = -C3^{-1} C2

        residuals = s_lin - s_rot @ reg_beta.T  # (N, 2)
        reg_cov = (residuals * w).T @ residuals  # (2, 2)

        # Use pinv here: reg_cov may be ill-conditioned when samples cluster on a subspace
        c3_est = np.linalg.pinv(-2.0 * reg_cov)
        c3_est = 0.5 * (c3_est + c3_est.T)
        c2_est = -c3_est @ reg_beta

        # c3_est is the estimated C3 (negative-definite); its inverse is well-conditioned
        c1_est = tmp + c2_est.T @ np.linalg.inv(c3_est) @ c2_est
        c1_est = 0.5 * (c1_est + c1_est.T)

        return SE2BinghamDistribution(array(c1_est), array(c2_est), array(c3_est))
