from .cart_prod.se2_pwn_distribution import SE2PWNDistribution


class SE2PartiallyWrappedNormalDistribution(SE2PWNDistribution):
    """Descriptive alias for ``SE2PWNDistribution``.

    This keeps the established implementation and legacy API while exposing a
    clearer class name at the package level.
    """

    def mean_4d(self):
        """Return the 4-D moment E[cos(x1), sin(x1), x2, x3]."""
        return self.mean4D()

    def covariance_4d(self):
        """Return the analytical 4-D covariance of [cos(x1), sin(x1), x2, x3]."""
        return self.covariance4D()

    def covariance_4d_numerical(self, n_samples=10000):
        """Estimate the 4-D covariance numerically."""
        return self.covariance4D_numerical(n_samples)

    @staticmethod
    def from_samples(samples):
        """Fit an ``SE2PartiallyWrappedNormalDistribution`` from samples."""
        fitted = SE2PWNDistribution.from_samples(samples)
        return SE2PartiallyWrappedNormalDistribution(fitted.mu, fitted.C)
