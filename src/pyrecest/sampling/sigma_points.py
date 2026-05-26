"""
Sigma-point sampling schemes for unscented transforms.
"""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray, float64, full, hstack, linalg, reshape, stack


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

    def _compute_weights(self):
        n = self.n
        lam = self.alpha**2 * (n + self.kappa) - n
        scale = n + lam

        self.Wm = hstack(
            [
                asarray([lam / scale], dtype=float64),
                full((2 * n,), 0.5 / scale, dtype=float64),
            ]
        )
        self.Wc = hstack(
            [
                asarray(
                    [lam / scale + (1.0 - self.alpha**2 + self.beta)],
                    dtype=float64,
                ),
                full((2 * n,), 0.5 / scale, dtype=float64),
            ]
        )

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

        positive = [x + U[:, i] for i in range(n)]
        negative = [x - U[:, i] for i in range(n)]
        return stack([x, *positive, *negative])


class JulierSigmaPoints:
    """Julier sigma points (Julier and Uhlmann, 1997).

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

    def _compute_weights(self):
        n = self.n
        k = n + self.kappa

        self.Wm = hstack(
            [
                asarray([self.kappa / k], dtype=float64),
                full((2 * n,), 0.5 / k, dtype=float64),
            ]
        )
        self.Wc = hstack(
            [
                asarray([self.kappa / k], dtype=float64),
                full((2 * n,), 0.5 / k, dtype=float64),
            ]
        )

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

        positive = [x + U[:, i] for i in range(n)]
        negative = [x - U[:, i] for i in range(n)]
        return stack([x, *positive, *negative])
