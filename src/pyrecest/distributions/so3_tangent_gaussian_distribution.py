"""Tangent-space Gaussian distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    amax,
    arctan2,
    array,
    clip,
    concatenate,
    cos,
    diag,
    linalg,
    log,
    matmul,
    ndim,
    pi,
    random,
    reshape,
    sin,
    stack,
    sum,
    transpose,
    where,
    zeros,
)

from ._so3_helpers import (
    as_batch,
    geodesic_distance,
    normalize_quaternions,
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

    @staticmethod
    def _quaternion_conjugate(quaternions):
        quaternions = SO3TangentGaussianDistribution._normalize_quaternions(quaternions)
        return quaternions * array([-1.0, -1.0, -1.0, 1.0])

    @staticmethod
    def _quaternion_multiply(left, right):
        left = SO3TangentGaussianDistribution._normalize_quaternions(left)
        right = SO3TangentGaussianDistribution._normalize_quaternions(right)

        x1, y1, z1, w1 = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
        x2, y2, z2, w2 = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
        product = stack(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            axis=-1,
        )
        return SO3TangentGaussianDistribution._normalize_quaternions(product)

    @staticmethod
    def exp_map(tangent_vectors, base=None):
        """Map tangent vectors to SO(3) quaternions.

        If ``base`` is given, the returned rotations are ``base * Exp(v)``.
        """
        tangent_vectors = SO3TangentGaussianDistribution._as_tangent_batch(
            tangent_vectors
        )
        angles = linalg.norm(tangent_vectors, None, -1)
        angles_col = reshape(angles, (-1, 1))
        safe_angles = where(angles_col > 1e-12, angles_col, 1.0)
        vector_scale = where(
            angles_col > 1e-12,
            sin(0.5 * angles_col) / safe_angles,
            0.5 - angles_col**2 / 48.0,
        )
        delta_quaternions = concatenate(
            (tangent_vectors * vector_scale, cos(0.5 * angles_col)), axis=-1
        )
        delta_quaternions = SO3TangentGaussianDistribution._normalize_quaternions(
            delta_quaternions
        )

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

        vector_part = rotations[:, :3]
        scalar_part = clip(rotations[:, 3], -1.0, 1.0)
        vector_norm = linalg.norm(vector_part, None, -1)
        angles = 2.0 * arctan2(vector_norm, scalar_part)
        vector_norm_col = reshape(vector_norm, (-1, 1))
        safe_norm = where(vector_norm_col > 1e-12, vector_norm_col, 1.0)
        scale = where(
            vector_norm_col > 1e-12, reshape(angles, (-1, 1)) / safe_norm, 2.0
        )
        return vector_part * scale

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
