"""
Implementations of Merwe-scaled and Julier sigma-point generators and an
Unscented Kalman Filter (UKF), replacing the former dependency on the
``bayesian_filters`` package.

The public API (class names, constructor signatures, attribute names, and
method signatures) is intentionally kept compatible with the ``filterpy``-style
API that ``bayesian_filters`` exposed, so that existing call sites require only
a one-line import change.
"""

from copy import deepcopy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    asarray,
    copy,
    einsum,
    empty,
    empty_like,
    expand_dims,
    eye,
    float64,
    full,
    linalg,
    reshape,
    transpose,
    zeros,
)


# ---------------------------------------------------------------------------
# Sigma-point generators
# ---------------------------------------------------------------------------


class MerweScaledSigmaPoints:
    """Merwe scaled sigma points (van der Merwe, 2004).

    Parameters
    ----------
    n:
        State dimension.
    alpha:
        Spread of sigma points around the mean (typically 1e-3).
    beta:
        Prior knowledge of the distribution (2 is optimal for Gaussians).
    kappa:
        Secondary scaling parameter (typically 0).
    """

    def __init__(self, n: int, alpha: float, beta: float, kappa: float):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self._compute_weights()

    # ------------------------------------------------------------------
    def _compute_weights(self):
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n

        self.Wm = full(2 * n + 1, 0.5 / (n + lam))
        self.Wm[0] = lam / (n + lam)

        self.Wc = copy(self.Wm)
        self.Wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)

    # ------------------------------------------------------------------
    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n

        x = reshape(asarray(x, dtype=float64), (-1,))
        P = asarray(P, dtype=float64)

        U = linalg.cholesky((n + lam) * P)  # lower-triangular

        sigmas = empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[n + i + 1] = x - U[:, i]
        return sigmas


class JulierSigmaPoints:
    """Julier sigma points (Julier & Uhlmann, 1997).

    Parameters
    ----------
    n:
        State dimension.
    kappa:
        Scaling parameter (``n + kappa`` should be non-zero).
    """

    def __init__(self, n: int, kappa: float = 0.0):
        self.n = n
        self.kappa = kappa
        self._compute_weights()

    # ------------------------------------------------------------------
    def _compute_weights(self):
        n = self.n
        k = n + self.kappa

        self.Wm = full(2 * n + 1, 0.5 / k)
        self.Wm[0] = self.kappa / k

        self.Wc = copy(self.Wm)

    # ------------------------------------------------------------------
    def sigma_points(self, x, P):
        """Return ``(2n+1, n)`` sigma-point matrix.

        Parameters
        ----------
        x:
            State mean, shape ``(n,)``.
        P:
            State covariance, shape ``(n, n)``.
        """
        n = self.n
        k = n + self.kappa

        x = reshape(asarray(x, dtype=float64), (-1,))
        P = asarray(P, dtype=float64)

        U = linalg.cholesky(k * P)  # lower-triangular

        sigmas = empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + U[:, i]
            sigmas[n + i + 1] = x - U[:, i]
        return sigmas


# ---------------------------------------------------------------------------
# Unscented Kalman Filter
# ---------------------------------------------------------------------------


class UnscentedKalmanFilter:
    """Minimal Unscented Kalman Filter compatible with the ``filterpy`` API.

    Parameters
    ----------
    dim_x:
        Dimension of the state vector.
    dim_z:
        Dimension of the measurement vector.
    dt:
        Time step (passed to *fx*).
    hx:
        Measurement function ``hx(x) -> z``.
    fx:
        State-transition function ``fx(x, dt, **kwargs) -> x_new``.
    points:
        Sigma-point generator (must expose ``sigma_points``, ``Wm``, ``Wc``).
    """

    def __init__(self, dim_x, dim_z, dt, hx, fx, points):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self._hx = hx
        self._fx = fx
        self._points = points

        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self.R = eye(dim_z)

        # Populated after predict()
        self.x_prior = copy(self.x)
        self.P_prior = copy(self.P)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, fx=None, dt=None, **fx_args):
        """UKF predict step.

        Parameters
        ----------
        fx:
            Override the default state-transition function.
        dt:
            Override the default time step.
        """
        if fx is None:
            fx = self._fx
        if dt is None:
            dt = self.dt

        sigmas = self._points.sigma_points(self.x, self.P)
        n_sigmas = sigmas.shape[0]

        sigmas_f = empty_like(sigmas)
        for i in range(n_sigmas):
            sigmas_f[i] = reshape(asarray(fx(sigmas[i], dt, **fx_args), dtype=float64), (-1,))

        Wm = self._points.Wm
        Wc = self._points.Wc

        x_pred = einsum("i,ij->j", Wm, sigmas_f)

        P_pred = zeros((self.dim_x, self.dim_x))
        for i in range(n_sigmas):
            d = expand_dims(sigmas_f[i] - x_pred, -1)
            P_pred = P_pred + Wc[i] * (d @ transpose(d))
        P_pred = P_pred + asarray(self.Q, dtype=float64)
        P_pred = 0.5 * (P_pred + transpose(P_pred))

        self.x = x_pred
        self.P = P_pred
        self._sigmas_f = sigmas_f  # store for update

        self.x_prior = copy(self.x)
        self.P_prior = copy(self.P)

    def update(self, z, R=None, hx=None, **hx_args):
        """UKF update step.

        Parameters
        ----------
        z:
            Measurement vector, shape ``(dim_z,)``.
        R:
            Override measurement noise covariance.
        hx:
            Override the default measurement function.
        """
        if hx is None:
            hx = self._hx
        if R is None:
            R = self.R
        R = asarray(R, dtype=float64)

        z = reshape(asarray(z, dtype=float64), (-1,))

        # Re-generate sigma points if predict() was not called
        if not hasattr(self, "_sigmas_f"):
            self._sigmas_f = self._points.sigma_points(self.x, self.P)

        sigmas_f = self._sigmas_f
        n_sigmas = sigmas_f.shape[0]
        Wm = self._points.Wm
        Wc = self._points.Wc

        # Propagate sigma points through hx
        sigmas_h = empty((n_sigmas, self.dim_z))
        for i in range(n_sigmas):
            sigmas_h[i] = reshape(asarray(hx(sigmas_f[i], **hx_args), dtype=float64), (-1,))

        z_pred = einsum("i,ij->j", Wm, sigmas_h)

        # Innovation covariance
        Pz = zeros((self.dim_z, self.dim_z))
        for i in range(n_sigmas):
            dz = expand_dims(sigmas_h[i] - z_pred, -1)
            Pz = Pz + Wc[i] * (dz @ transpose(dz))
        Pz = Pz + R

        # Cross-covariance
        Pxz = zeros((self.dim_x, self.dim_z))
        for i in range(n_sigmas):
            dx = expand_dims(sigmas_f[i] - self.x, -1)
            dz = expand_dims(sigmas_h[i] - z_pred, -1)
            Pxz = Pxz + Wc[i] * (dx @ transpose(dz))

        # Kalman gain  (solve Pz K^T = Pxz^T  =>  K = (Pz^{-1} Pxz^T)^T)
        K = transpose(linalg.solve(Pz, transpose(Pxz)))

        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ Pz @ transpose(K)
        self.P = 0.5 * (self.P + transpose(self.P))

        # Clear cached sigma points
        del self._sigmas_f

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
