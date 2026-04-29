"""Tangent-space Gaussian distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    amax,
    array,
    diag,
    linalg,
    log,
    matmul,
    ndim,
    pi,
    random,
    sum,
    transpose,
    zeros,
)

from ._so3_helpers import (
    as_batch,
    exp_map_identity,
    geodesic_distance,
    log_map_identity,
    normalize_quaternions,
    quaternion_conjugate,
    quaternion_multiply,
    quaternions_to_rotation_matrices,
)
from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution
from .nonperiodic.gaussian_distribution import GaussianDistribution


class SO3TangentGaussianDistribution(AbstractBoundedDomainDistribution):
    """Gaussian distribution in a local tangent chart of SO(3).

    Rotations are represented as scalar-last unit quaternions ``(x, y, z, w)``.
    The density is evaluated by mapping rotations into the tangent space at the
    mean via ``log(mean^{-1} * rotation)`` and applying a 3-D Gaussian there.
    This is a local tangent approximation, not a globally wrapped density.
    """

    def __init__(self, mu, C, check_validity=True):
        super().__init__(dim=3)
        self.mu = self._normalize_quaternions(mu)[0]

        C = array(C, dtype=float)
        assert ndim(C) == 2 and C.shape == (3, 3), "C must have shape (3, 3)."
        if check_validity:
            linalg.cholesky(C)
        self.C = C

    @property
    def input_dim(self):
        return 4

    _normalize_quaternions = staticmethod(normalize_quaternions)

    @staticmethod
    def _as_tangent_batch(tangent_vectors):
        return as_batch(tangent_vectors, 3, "SO(3) tangent vectors")

    _quaternion_conjugate = staticmethod(quaternion_conjugate)
    _quaternion_multiply = staticmethod(quaternion_multiply)

    @staticmethod
    def exp_map(tangent_vectors, base=None):
        """Map tangent vectors to SO(3) quaternions.

        If ``base`` is given, the returned rotations are ``base * Exp(v)``.
        """
        tangent_vectors = SO3TangentGaussianDistribution._as_tangent_batch(
            tangent_vectors
        )
        delta_quaternions = exp_map_identity(tangent_vectors)

        if base is None:
            return delta_quaternions
        return SO3TangentGaussianDistribution._quaternion_multiply(
            base, delta_quaternions
        )

    @staticmethod
    def log_map(rotations, base=None):
        """Map SO(3) quaternions to tangent vectors.

        If ``base`` is given, this returns ``Log(base^{-1} * rotations)``.
        """
        rotations = SO3TangentGaussianDistribution._normalize_quaternions(rotations)
        if base is not None:
            rotations = SO3TangentGaussianDistribution._quaternion_multiply(
                SO3TangentGaussianDistribution._quaternion_conjugate(base), rotations
            )

        return log_map_identity(rotations)

    geodesic_distance = staticmethod(geodesic_distance)

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last quaternions to rotation matrices."""
        return quaternions_to_rotation_matrices(quaternions)

    def pdf(self, xs):
        """Evaluate the tangent Gaussian density at SO(3) quaternions."""
        tangent_vectors = self.log_map(xs, base=self.mu)
        gaussian = GaussianDistribution(zeros(3), self.C, check_validity=False)
        return gaussian.pdf(tangent_vectors)

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the tangent Gaussian density."""
        tangent_vectors = self.log_map(xs, base=self.mu)
        residual = tangent_vectors
        precision = linalg.inv(self.C)
        quadratic = sum(residual * matmul(residual, precision), axis=-1)
        log_det = 2.0 * sum(log(diag(linalg.cholesky(self.C))))
        return -0.5 * (3.0 * log(2.0 * pi) + log_det + quadratic)

    def tangent_vectors(self, rotations):
        """Return log-map coordinates of rotations around the distribution mean."""
        return self.log_map(rotations, base=self.mu)

    def sample_tangent(self, n):
        """Draw tangent-space Gaussian samples with shape ``(n, 3)``."""
        return random.multivariate_normal(mean=zeros(3), cov=self.C, size=n)

    def sample(self, n):
        """Draw ``n`` SO(3) samples as scalar-last unit quaternions."""
        return self.exp_map(self.sample_tangent(n), base=self.mu)

    def mean(self):
        """Return the mean rotation as a scalar-last unit quaternion."""
        return self.mu

    def mode(self):
        """Return the modal rotation as a scalar-last unit quaternion."""
        return self.mu

    def mean_rotation_matrix(self):
        """Return the mean rotation matrix."""
        return self.as_rotation_matrices(self.mu)[0]

    def covariance(self):
        """Return the 3-by-3 tangent covariance matrix."""
        return self.C

    def set_mean(self, new_mean):
        """Return a copy with a replaced mean rotation."""
        return self.set_mode(new_mean)

    def set_mode(self, new_mode):
        """Return a copy with a replaced modal rotation."""
        new_dist = self.__class__(new_mode, self.C, check_validity=False)
        return new_dist

    def get_manifold_size(self):
        """Return the embedding half-sphere volume used for unit quaternions."""
        return pi**2

    def is_valid(self, tolerance=1e-6):
        """Return whether the mean and covariance have valid SO(3) dimensions."""
        covariance_is_symmetric = amax(abs(self.C - transpose(self.C))) <= tolerance
        return bool(
            abs(linalg.norm(self.mu) - 1.0) <= tolerance
            and self.mu[-1] >= -tolerance
            and covariance_is_symmetric
        )

    @staticmethod
    def from_covariance_diagonal(mu, covariance_diagonal):
        """Create a tangent Gaussian from a diagonal covariance vector."""
        return SO3TangentGaussianDistribution(mu, diag(covariance_diagonal))
