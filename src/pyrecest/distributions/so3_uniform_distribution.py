"""Uniform distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    arccos,
    clip,
    log,
    ones,
    pi,
    stack,
    sum,
)

from .hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)
from .so3_dirac_distribution import SO3DiracDistribution


class SO3UniformDistribution(HyperhemisphericalUniformDistribution):
    """Uniform distribution on SO(3) using scalar-last unit quaternions.

    The distribution represents rotations by canonical unit quaternions
    ``(x, y, z, w)`` with nonnegative scalar component. This identifies each
    SO(3) rotation with one point on the upper hemisphere of S^3, except for
    the measure-zero equator where ``q`` and ``-q`` both have zero scalar part.
    """

    def __init__(self):
        super().__init__(dim=3)

    @staticmethod
    def _normalize_quaternions(quaternions):
        return SO3DiracDistribution._normalize_quaternions(quaternions)

    def pdf(self, xs):
        """Evaluate the constant SO(3) density."""
        xs = self._normalize_quaternions(xs)
        return ones(xs.shape[0]) / self.get_manifold_size()

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the constant SO(3) density."""
        xs = self._normalize_quaternions(xs)
        return -log(self.get_manifold_size()) * ones(xs.shape[0])

    def sample(self, n):
        """Draw ``n`` canonical scalar-last unit quaternions."""
        return super().sample(n)

    def mode(self):
        """Mode is undefined for a uniform SO(3) distribution."""
        raise AttributeError("Mode not available for uniform distribution")

    def mean(self):
        """Mean rotation is undefined for a uniform SO(3) distribution."""
        raise AttributeError("Mean not available for uniform SO(3) distribution")

    def mean_axis(self):
        """Mean axis is undefined for a uniform SO(3) distribution."""
        raise AttributeError("Mean axis not available for uniform SO(3) distribution")

    @staticmethod
    def geodesic_distance(rotation_a, rotation_b):
        """Return the SO(3) geodesic distance between quaternions in radians."""
        quat_a = SO3UniformDistribution._normalize_quaternions(rotation_a)
        quat_b = SO3UniformDistribution._normalize_quaternions(rotation_b)
        inner = abs(sum(quat_a * quat_b, axis=-1))
        return 2.0 * arccos(clip(inner, 0.0, 1.0))

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last quaternions to rotation matrices."""
        quaternions = SO3UniformDistribution._normalize_quaternions(quaternions)
        x, y, z, w = (
            quaternions[:, 0],
            quaternions[:, 1],
            quaternions[:, 2],
            quaternions[:, 3],
        )
        row_0 = stack(
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            axis=-1,
        )
        row_1 = stack(
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            axis=-1,
        )
        row_2 = stack(
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
            axis=-1,
        )
        return stack((row_0, row_1, row_2), axis=-2)

    def is_valid(self, tolerance=1e-12):
        """Return whether this instance has the expected SO(3) dimensions."""
        return bool(
            self.dim == 3 and abs(self.get_manifold_size() - pi**2) <= tolerance
        )
