import numpy as _np
from scipy.integrate import nquad
from scipy.linalg import cholesky as _scipy_cholesky
from scipy.special import iv
from scipy.stats import multivariate_normal as _mvn
from scipy.stats import vonmises as _vonmises

from pyrecest.backend import (
    arccos,
    array,
    allclose,
    concatenate,
    cos,
    eye,
    exp,
    float64,
    hstack,
    imag,
    linalg,
    mod,
    pi,
    random,
    real,
    sin,
    sqrt,
    sum,
    vstack,
    zeros,
)

from ..nonperiodic.gaussian_distribution import GaussianDistribution
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


class GaussVonMisesDistribution(AbstractHypercylindricalDistribution):
    """Gauss von Mises Distribution on R^linD x S^1.

    Reference:
        Horwood, J. T. & Poore, A. B.
        Gauss von Mises Distribution for Improved Uncertainty Realism in Space
        Situational Awareness.
        SIAM/ASA Journal on Uncertainty Quantification, 2014, 2, 276-304
    """

    def __init__(self, mu, P, alpha, beta, Gamma, kappa):
        # Convert scalars/lists to arrays
        mu = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
        P = _np.atleast_2d(_np.asarray(P, dtype=_np.float64))
        beta = _np.atleast_1d(_np.asarray(beta, dtype=_np.float64))
        Gamma = _np.atleast_2d(_np.asarray(Gamma, dtype=_np.float64))

        n = mu.shape[0]

        # Validate parameters
        assert P.shape == (n, n), "P and mu must have matching size"
        assert _np.allclose(P, P.T), "P must be symmetric"
        _scipy_cholesky(P, lower=True)  # raises if not positive definite

        assert beta.shape == (n,), "size of beta must match size of mu"
        assert Gamma.shape == (n, n), "Gamma and mu must have matching size"
        assert _np.allclose(Gamma, Gamma.T), "Gamma must be symmetric"
        assert float(kappa) > 0, "kappa has to be a positive scalar"

        AbstractHypercylindricalDistribution.__init__(self, bound_dim=1, lin_dim=n)

        self.mu = array(mu)
        self.P = array(P)
        self.alpha = float(mod(array(float(alpha)), 2.0 * pi))
        self.beta = array(beta)
        self.Gamma = array(Gamma)
        self.kappa = float(kappa)
        # Lower triangular Cholesky factor of P
        self.A = array(_scipy_cholesky(P, lower=True))

    def get_theta(self, xa):
        """Compute the angle offset theta for each column of xa (linear part).

        Parameters
        ----------
        xa : array of shape (lin_dim, n)

        Returns
        -------
        theta : array of shape (n,)
        """
        if xa.ndim == 1:
            xa = xa.reshape(-1, 1)
        z = linalg.solve(self.A, xa - self.mu.reshape(-1, 1))  # (linD, n)
        # Quadratic form: 0.5 * z.T @ Gamma @ z for each column
        Gamma_z = self.Gamma @ z  # (linD, n)
        quad = 0.5 * sum(z * Gamma_z, axis=0)  # (n,)
        theta = self.alpha + self.beta @ z + quad  # (n,)
        return theta

    def pdf(self, xa):
        """Evaluate the pdf at each column of xa.

        Parameters
        ----------
        xa : array of shape (lin_dim + 1,) or (lin_dim + 1, n)
            First row/element is the periodic variable, the rest are linear.

        Returns
        -------
        p : scalar or array of shape (n,)
        """
        xa = _np.asarray(xa, dtype=_np.float64)
        single_point = xa.ndim == 1
        if single_point:
            xa = xa.reshape(-1, 1)

        assert xa.shape[0] == self.lin_dim + 1

        theta = self.get_theta(xa[1:, :])
        mvn_vals = _mvn.pdf(xa[1:, :].T, mean=_np.asarray(self.mu), cov=_np.asarray(self.P))
        p = mvn_vals * _np.exp(self.kappa * _np.cos(xa[0, :] - _np.asarray(theta))) / (
            2.0 * float(pi) * iv(0, self.kappa)
        )

        if single_point:
            return float(p[0])
        return array(p)

    def mode(self):
        """Return the mode [alpha, mu_1, ..., mu_linD]."""
        return concatenate([array([self.alpha]), self.mu])

    def hybrid_moment(self):
        """Analytic hybrid moment E[cos(theta), sin(theta), x_1, ..., x_linD].

        See Horwood, Section 4.3.
        """
        from ..circle.von_mises_distribution import VonMisesDistribution

        M = _np.eye(self.lin_dim) - 1j * _np.asarray(self.Gamma)
        eiphi = (
            1.0 / _np.sqrt(_np.linalg.det(M))
            * VonMisesDistribution.besselratio(0, self.kappa)
            * _np.exp(
                1j * self.alpha
                - 0.5 * _np.asarray(self.beta) @ _np.linalg.solve(M, _np.asarray(self.beta))
            )
        )
        return concatenate([array([float(_np.real(eiphi)), float(_np.imag(eiphi))]), self.mu])

    def hybrid_moment_numerical(self):
        """Numerical hybrid moment E[cos(theta), sin(theta), x_1, ..., x_linD]."""
        P_np = _np.asarray(self.P)
        mu_np = _np.asarray(self.mu)
        scale = 10.0

        bounds_periodic = [[0.0, 2.0 * _np.pi]]
        bounds_linear = [
            [float(mu_np[i]) - scale * float(_np.sqrt(P_np[i, i])),
             float(mu_np[i]) + scale * float(_np.sqrt(P_np[i, i]))]
            for i in range(self.lin_dim)
        ]
        bounds = bounds_periodic + bounds_linear

        if self.lin_dim == 1:
            e_cos = nquad(
                lambda theta, x: _np.cos(theta) * self.pdf(_np.array([theta, x])), bounds
            )[0]
            e_sin = nquad(
                lambda theta, x: _np.sin(theta) * self.pdf(_np.array([theta, x])), bounds
            )[0]
            e_x = nquad(
                lambda theta, x: x * self.pdf(_np.array([theta, x])), bounds
            )[0]
            return array([e_cos, e_sin, e_x])

        raise NotImplementedError("hybrid_moment_numerical not implemented for lin_dim > 1")

    def sample(self, n):
        """Draw n samples from the distribution.

        Parameters
        ----------
        n : int

        Returns
        -------
        s : array of shape (lin_dim + 1, n)
            First row is periodic (angle), remaining rows are linear.
        """
        s_gauss = random.multivariate_normal(
            mean=self.mu, cov=self.P, size=n
        ).T  # (linD, n)
        theta = self.get_theta(s_gauss)  # (n,)
        vm_samples = _vonmises.rvs(kappa=self.kappa, loc=0.0, size=n)
        vm_samples = _np.mod(vm_samples + _np.asarray(theta), 2.0 * _np.pi)
        return vstack([array(vm_samples).reshape(1, -1), s_gauss])

    def sample_deterministic_horwood(self):
        """Deterministic sigma-point approximation (Horwood, Section 5.1).

        Returns
        -------
        d : array of shape (lin_dim + 1, 2*lin_dim + 3)
            Sigma-point locations.
        w : array of shape (2*lin_dim + 3,)
            Sigma-point weights.
        """
        lin_dim = self.lin_dim

        def B(p_val, kappa_val):
            return 1.0 - iv(p_val, kappa_val) / iv(0, kappa_val)

        B1 = B(1, self.kappa)
        B2 = B(2, self.kappa)
        xi = float(sqrt(array(3.0)))
        eta = float(arccos(array(B2 / (2.0 * B1) - 1.0)))
        wxi0 = 1.0 / 6.0
        weta0 = B1 ** 2 / (4.0 * B1 - B2)
        w00 = 1.0 - 2.0 * weta0 - 2.0 * lin_dim * wxi0

        n_pts = 2 * lin_dim + 3
        d = _np.zeros((lin_dim + 1, n_pts))

        # Column 0: origin (all zeros) — N00
        # Columns 1-2: Neta0 (±eta on the periodic axis)
        d[lin_dim, 1] = -eta
        d[lin_dim, 2] = eta
        # Columns 3..n_pts-1: Nxi0 (±xi on each linear axis)
        for i in range(lin_dim):
            d[i, 3 + 2 * i] = -xi
            d[i, 3 + 2 * i + 1] = xi

        w = _np.full(n_pts, wxi0)
        w[0] = w00
        w[1] = weta0
        w[2] = weta0

        # Transform back to original parameterisation
        lin_part = d[1:, :]  # (linD, n_pts)
        theta_vals = _np.asarray(self.get_theta(array(lin_part)))
        d[0, :] = _np.mod(d[0, :] + theta_vals, 2.0 * _np.pi)
        d[1:, :] = _np.asarray(self.A) @ lin_part + _np.asarray(self.mu).reshape(-1, 1)

        return array(d), array(w)

    def integrate(self, integration_boundaries=None):
        """Numerically integrate the pdf over its domain."""
        if integration_boundaries is not None:
            return self.integrate_numerically(integration_boundaries)

        P_np = _np.asarray(self.P)
        mu_np = _np.asarray(self.mu)
        scale = 10.0
        bounds = [[0.0, 2.0 * _np.pi]] + [
            [float(mu_np[i]) - scale * float(_np.sqrt(P_np[i, i])),
             float(mu_np[i]) + scale * float(_np.sqrt(P_np[i, i]))]
            for i in range(self.lin_dim)
        ]
        return nquad(lambda *args: self.pdf(_np.array(args)), bounds)[0]

    def to_gaussian(self):
        """Approximate conversion to a Gaussian (valid for large kappa, small Gamma).

        See Horwood, Section 4.6.
        """
        mtmp = concatenate([array([self.alpha]), self.mu])
        top_row = hstack([array([[1.0 / sqrt(array(self.kappa))]]), self.beta.reshape(1, -1)])
        bot_rows = hstack([zeros((self.lin_dim, 1)), self.A])
        Atmp = vstack([top_row, bot_rows])
        Ptmp = Atmp @ Atmp.T
        return GaussianDistribution(mtmp, Ptmp)

    def linear_covariance(self, approximate_mean=None):
        """Return the covariance of the linear dimensions."""
        return self.P

    def marginalize_circular(self):
        """Marginalise out the circular dimension, returning a Gaussian."""
        return GaussianDistribution(self.mu, self.P)

    def marginalize_periodic(self):
        """Marginalise out the periodic dimension, returning a Gaussian.

        Since integral_0^{2pi} pdf(theta,x) dtheta = N(x; mu, P), this is exact.
        """
        return GaussianDistribution(self.mu, self.P)

    def marginalize_linear(self):
        """Marginalise out the linear dimensions, returning a custom circular distribution."""
        from ..hypertorus.custom_hypertoroidal_distribution import (
            CustomHypertoroidalDistribution,
        )

        P_np = _np.asarray(self.P)
        mu_np = _np.asarray(self.mu)
        scale = 10.0
        bounds_linear = [
            [float(mu_np[i]) - scale * float(_np.sqrt(P_np[i, i])),
             float(mu_np[i]) + scale * float(_np.sqrt(P_np[i, i]))]
            for i in range(self.lin_dim)
        ]

        if self.lin_dim == 1:
            def marginal_pdf(theta):
                theta = _np.atleast_1d(_np.asarray(theta))
                result = _np.zeros_like(theta, dtype=_np.float64)
                for idx, t in enumerate(theta.ravel()):
                    result.ravel()[idx] = nquad(
                        lambda x: self.pdf(_np.array([t, x])), bounds_linear
                    )[0]
                return result

            dist = CustomHypertoroidalDistribution(marginal_pdf, self.bound_dim)
        else:
            raise NotImplementedError(
                "marginalize_linear not implemented for lin_dim > 1"
            )

        return dist