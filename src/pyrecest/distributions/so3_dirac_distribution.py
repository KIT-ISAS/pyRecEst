"""Dirac distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    amax,
    arccos,
    argmax,
    array,
    asarray,
    clip,
    column_stack,
    linalg,
    ndim,
    outer,
    reshape,
    spatial,
    sum,
    where,
    zeros,
)

from .abstract_dirac_distribution import AbstractDiracDistribution


class SO3DiracDistribution(AbstractDiracDistribution):
    """Weighted Dirac distribution on SO(3).

    The distribution stores rotations as unit quaternions with scalar-first
    coordinates ``(w, x, y, z)``. Since ``q`` and ``-q`` represent the same
    rotation, quaternions are canonicalized to a nonnegative scalar component
    during construction.
    """

    def __init__(self, d, w=None):
        quaternions = self._normalize_quaternions(d)
        self.dim = 3
        super().__init__(quaternions, w=w)

    @property
    def input_dim(self):
        return 4

    @staticmethod
    def _normalize_quaternions(quaternions):
        quaternions = array(quaternions, dtype=float)
        if ndim(quaternions) == 1:
            quaternions = reshape(quaternions, (1, 4))

        assert quaternions.shape[-1] == 4, "SO(3) quaternions must have length 4."
        norms = linalg.norm(quaternions, None, -1)
        assert all(norms > 0.0), "SO(3) quaternions must be nonzero."

        normalized = quaternions / reshape(norms, (-1, 1))
        sign = where(normalized[:, 0:1] < 0.0, -1.0, 1.0)
        return sign * normalized

    @staticmethod
    def _as_xyzw(quaternions):
        quaternions = SO3DiracDistribution._normalize_quaternions(quaternions)
        return column_stack(
            (quaternions[:, 1], quaternions[:, 2], quaternions[:, 3], quaternions[:, 0])
        )

    @staticmethod
    def _as_wxyz(quaternions_xyzw):
        quaternions_xyzw = asarray(quaternions_xyzw)
        if ndim(quaternions_xyzw) == 1:
            quaternions_xyzw = reshape(quaternions_xyzw, (1, 4))
        return column_stack(
            (
                quaternions_xyzw[:, 3],
                quaternions_xyzw[:, 0],
                quaternions_xyzw[:, 1],
                quaternions_xyzw[:, 2],
            )
        )

    @staticmethod
    def _require_rotation_method(method_name):
        if not hasattr(spatial.Rotation, method_name):
            raise NotImplementedError(
                f"Rotation.{method_name} is not supported by the active backend."
            )

    @classmethod
    def from_rotation_matrices(cls, rotation_matrices, w=None):
        """Create an SO(3) Dirac distribution from rotation matrices."""
        cls._require_rotation_method("from_matrix")
        rotation_matrices = asarray(rotation_matrices)
        if ndim(rotation_matrices) == 2:
            assert rotation_matrices.shape == (
                3,
                3,
            ), "A single rotation matrix must have shape (3, 3)."
        else:
            assert rotation_matrices.shape[-2:] == (
                3,
                3,
            ), "Rotation matrices must have shape (..., 3, 3)."

        quaternions_xyzw = spatial.Rotation.from_matrix(rotation_matrices).as_quat()
        return cls(cls._as_wxyz(quaternions_xyzw), w=w)

    def as_quaternions(self):
        """Return canonical scalar-first unit quaternions shaped ``(n, 4)``."""
        return self.d

    def as_rotation_matrices(self):
        """Return Dirac locations as rotation matrices shaped ``(n, 3, 3)``."""
        self._require_rotation_method("from_quat")
        return array(spatial.Rotation.from_quat(self._as_xyzw(self.d)).as_matrix())

    def moment(self):
        """Return the weighted quaternion second-moment matrix."""
        moment_matrix = zeros((self.input_dim, self.input_dim))
        for idx in range(self.d.shape[0]):
            moment_matrix = moment_matrix + self.w[idx] * outer(
                self.d[idx], self.d[idx]
            )
        return moment_matrix / sum(self.w)

    def mean_axis(self):
        """Return the principal quaternion axis of the Dirac mixture."""
        moment_matrix = self.moment()
        eigenvalues, eigenvectors = linalg.eigh(0.5 * (moment_matrix + moment_matrix.T))
        mean_quaternion = eigenvectors[:, argmax(eigenvalues)]
        return self._normalize_quaternions(mean_quaternion)[0]

    def mean(self):
        """Return the mean rotation as a canonical scalar-first quaternion."""
        return self.mean_axis()

    def mean_rotation_matrix(self):
        """Return the mean rotation as a 3-by-3 rotation matrix."""
        self._require_rotation_method("from_quat")
        return array(
            spatial.Rotation.from_quat(self._as_xyzw(self.mean())).as_matrix()
        )[0]

    @staticmethod
    def geodesic_distance(rotation_a, rotation_b):
        """Return the SO(3) geodesic distance between quaternions in radians."""
        quat_a = SO3DiracDistribution._normalize_quaternions(rotation_a)
        quat_b = SO3DiracDistribution._normalize_quaternions(rotation_b)
        inner = abs(sum(quat_a * quat_b, axis=-1))
        return 2.0 * arccos(clip(inner, 0.0, 1.0))

    def distance_to(self, rotation):
        """Return geodesic distances from all Dirac locations to ``rotation``."""
        return self.geodesic_distance(self.d, rotation)

    @staticmethod
    def from_distribution(distribution, n_particles):
        """Create an SO(3) Dirac distribution by sampling another distribution."""
        assert (
            isinstance(n_particles, int) and n_particles > 0
        ), "n_particles must be a positive integer"
        return SO3DiracDistribution(distribution.sample(n_particles))

    def mode(self, rel_tol=0.001):
        """Return the highest-weight Dirac location as a canonical quaternion."""
        _ = rel_tol
        return self.d[int(argmax(self.w))]

    def angular_error_mean(self, rotation):
        """Return the weighted mean angular error to ``rotation`` in radians."""
        return sum(self.w * self.distance_to(rotation))

    def is_valid(self, tolerance=1e-6):
        """Return whether all stored quaternions are normalized and canonical."""
        norms = linalg.norm(self.d, None, -1)
        return bool(
            amax(abs(norms - 1.0)) <= tolerance and all(self.d[:, 0] >= -tolerance)
        )
