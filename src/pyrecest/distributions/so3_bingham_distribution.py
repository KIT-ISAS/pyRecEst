"""Bingham distribution on SO(3)."""

import numpy as _np
from scipy.integrate import quad
from scipy.special import iv

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    amax,
    arccos,
    array,
    clip,
    diag,
    eye,
    linalg,
    log,
    ndim,
    reshape,
    stack,
    sum,
    to_numpy,
    transpose,
    where,
    zeros,
)

from .hypersphere_subset.bingham_distribution import BinghamDistribution
from .hypersphere_subset.hyperhemispherical_bingham_distribution import (
    HyperhemisphericalBinghamDistribution,
)


class SO3BinghamDistribution(HyperhemisphericalBinghamDistribution):
    """Bingham distribution over rotations represented by unit quaternions.

    The distribution uses scalar-last quaternion coordinates ``(x, y, z, w)``.
    Because ``q`` and ``-q`` encode the same rotation, this class models SO(3)
    through the upper quaternion hemisphere and evaluates twice the full
    antipodally symmetric S^3 Bingham density.
    """

    def __init__(self, Z, M):
        Z = array(Z, dtype=float)
        M = array(M, dtype=float)
        assert ndim(Z) == 1 and Z.shape[0] == 4, "Z must have shape (4,)."
        assert M.shape == (4, 4), "M must have shape (4, 4)."

        super().__init__(Z, M)
        self.distFullSphere.F = array(self.calculate_normalization_constant(Z))

    @staticmethod
    def calculate_normalization_constant(Z):
        """Return the 4-D Bingham normalizing constant."""
        Z_np = _np.asarray(to_numpy(Z), dtype=float)
        assert Z_np.shape == (4,), "Z must have shape (4,)."

        def integrand(u):
            first_pair = iv(0, 0.5 * _np.abs(float(Z_np[0] - Z_np[1])) * u)
            second_pair = iv(0, 0.5 * _np.abs(float(Z_np[2] - Z_np[3])) * (1.0 - u))
            exponent = 0.5 * (Z_np[0] + Z_np[1]) * u + 0.5 * (Z_np[2] + Z_np[3]) * (
                1.0 - u
            )
            return first_pair * second_pair * _np.exp(exponent)

        return float(2.0 * _np.pi**2 * quad(integrand, 0.0, 1.0)[0])

    @staticmethod
    def _as_quaternion_batch(quaternions):
        quaternions = array(quaternions, dtype=float)
        if ndim(quaternions) == 1:
            assert quaternions.shape[0] == 4, "SO(3) quaternions must have length 4."
            quaternions = reshape(quaternions, (1, 4))
        else:
            assert quaternions.shape[-1] == 4, "SO(3) quaternions must have length 4."
            quaternions = reshape(quaternions, (-1, 4))
        return quaternions

    @staticmethod
    def _normalize_quaternions(quaternions):
        quaternions = SO3BinghamDistribution._as_quaternion_batch(quaternions)
        norms = linalg.norm(quaternions, None, -1)
        assert all(norms > 0.0), "SO(3) quaternions must be nonzero."

        normalized = quaternions / reshape(norms, (-1, 1))
        sign = where(normalized[:, -1:] < 0.0, -1.0, 1.0)
        return sign * normalized

    @staticmethod
    def _quaternion_right_multiply_matrix(quaternion):
        """Return the scalar-last matrix mapping ``p`` to ``p * quaternion``."""
        quaternion = SO3BinghamDistribution._normalize_quaternions(quaternion)[0]
        x, y, z, w = quaternion
        return array(
            [
                [w, z, -y, x],
                [-z, w, x, y],
                [y, -x, w, z],
                [-x, -y, -z, w],
            ]
        )

    @staticmethod
    def _canonicalize_quaternion(quaternion):
        return SO3BinghamDistribution._normalize_quaternions(quaternion)[0]

    @staticmethod
    def _orthogonal_completion(mode):
        mode_np = _np.asarray(
            to_numpy(SO3BinghamDistribution._canonicalize_quaternion(mode)), dtype=float
        )
        basis: list[_np.ndarray] = []
        for column in _np.eye(4):
            vector = column - _np.dot(column, mode_np) * mode_np
            for existing in basis:
                vector = vector - _np.dot(vector, existing) * existing

            norm = _np.linalg.norm(vector)
            if norm > 1e-10:
                basis.append(vector / norm)
            if len(basis) == 3:
                break

        return array(_np.column_stack((*basis, mode_np)))

    @classmethod
    def from_mode_and_concentration(cls, mode, concentration):
        """Create an isotropic Bingham distribution around ``mode``."""
        assert concentration >= 0.0, "concentration must be nonnegative."
        Z = array([-concentration, -concentration, -concentration, 0.0])
        M = cls._orthogonal_completion(mode)
        return cls(Z, M)

    @classmethod
    def from_concentration_matrix(cls, concentration_matrix):
        """Create from a symmetric 4-by-4 exponent matrix."""
        matrix_np = _np.asarray(to_numpy(concentration_matrix), dtype=float)
        assert matrix_np.shape == (4, 4), "concentration_matrix must have shape (4, 4)."
        matrix_np = 0.5 * (matrix_np + matrix_np.T)

        eigenvalues, eigenvectors = _np.linalg.eigh(matrix_np)
        order = _np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        if eigenvectors[-1, -1] < 0.0:
            eigenvectors[:, -1] = -eigenvectors[:, -1]

        Z = eigenvalues - eigenvalues[-1]
        Z[-1] = 0.0
        return cls(array(Z), array(eigenvectors))

    @classmethod
    def from_bingham_distribution(cls, distribution):
        """Create from a 4-D hyperspherical Bingham distribution."""
        assert isinstance(
            distribution, BinghamDistribution
        ), "distribution must be a BinghamDistribution."
        assert distribution.input_dim == 4, "distribution must be 4-D."
        return cls(distribution.Z, distribution.M)

    def pdf(self, xs):
        """Evaluate the SO(3) density at scalar-last quaternions."""
        return super().pdf(self._normalize_quaternions(xs))

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the SO(3) density."""
        return log(self.pdf(xs))

    def mode(self):
        """Return the modal rotation as a canonical scalar-last quaternion."""
        return self._canonicalize_quaternion(self.distFullSphere.mode())

    def mean(self):
        """Return the principal quaternion axis as the SO(3) mean proxy."""
        return self.mean_axis()

    def mean_axis(self):
        """Return the principal axis as a canonical scalar-last quaternion."""
        return self.mode()

    def concentration_matrix(self):
        """Return the symmetric 4-by-4 exponent matrix."""
        return (
            self.distFullSphere.M
            @ diag(self.distFullSphere.Z)
            @ self.distFullSphere.M.T
        )

    def moment_weights(self):
        """Return normalized Bingham moment weights without backend mutation."""
        Z_np = _np.asarray(to_numpy(self.distFullSphere.Z), dtype=float)
        normalizer = self.calculate_normalization_constant(Z_np)
        epsilon = 0.001
        derivatives = _np.zeros(4)
        for idx in range(4):
            delta = _np.zeros(4)
            delta[idx] = epsilon
            derivatives[idx] = (
                self.calculate_normalization_constant(Z_np + delta)
                - self.calculate_normalization_constant(Z_np - delta)
            ) / (2.0 * epsilon)

        weights = derivatives / normalizer
        weights = weights / _np.sum(weights)
        return array(weights)

    def moment(self):
        """Return the quaternion scatter matrix."""
        weights = self.moment_weights()
        moment_matrix = (
            self.distFullSphere.M @ diag(weights) @ transpose(self.distFullSphere.M)
        )
        return 0.5 * (moment_matrix + transpose(moment_matrix))

    def multiply(self, B2):
        """Multiply two SO(3) Bingham densities."""
        assert isinstance(
            B2, SO3BinghamDistribution
        ), "B2 must be an SO3BinghamDistribution."
        product = self.distFullSphere.multiply(B2.distFullSphere)
        return SO3BinghamDistribution(product.Z, product.M)

    def compose(self, B2):
        """Approximate the distribution of the composed rotation ``self * other``."""
        assert isinstance(
            B2, SO3BinghamDistribution
        ), "B2 must be an SO3BinghamDistribution."

        weights = B2.moment_weights()
        first_moment = self.moment()

        composed_moment = zeros((4, 4))
        for idx in range(4):
            right_matrix = self._quaternion_right_multiply_matrix(
                B2.distFullSphere.M[:, idx]
            )
            composed_moment = composed_moment + weights[
                idx
            ] * right_matrix @ first_moment @ transpose(right_matrix)

        composed_moment = 0.5 * (composed_moment + transpose(composed_moment))
        return SO3BinghamDistribution.from_bingham_distribution(
            BinghamDistribution.fit_to_moment(composed_moment)
        )

    def sample(self, n):
        """Draw ``n`` canonical scalar-last unit quaternion samples."""
        return self._normalize_quaternions(super().sample(n))

    @staticmethod
    def geodesic_distance(rotation_a, rotation_b):
        """Return the SO(3) geodesic distance between quaternions in radians."""
        quat_a = SO3BinghamDistribution._normalize_quaternions(rotation_a)
        quat_b = SO3BinghamDistribution._normalize_quaternions(rotation_b)
        inner = abs(sum(quat_a * quat_b, axis=-1))
        return 2.0 * arccos(clip(inner, 0.0, 1.0))

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last quaternions to rotation matrices."""
        quaternions = SO3BinghamDistribution._normalize_quaternions(quaternions)
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

    def mean_rotation_matrix(self):
        """Return the modal/principal rotation matrix."""
        return self.as_rotation_matrices(self.mean())[0]

    def is_valid(self, tolerance=1e-6):
        """Return whether the parameters define a valid SO(3) Bingham model."""
        Z = self.distFullSphere.Z
        M = self.distFullSphere.M
        orthogonal = amax(abs(M @ transpose(M) - eye(4))) <= tolerance
        sorted_z = all(Z[:-1] <= Z[1:] + tolerance)
        zero_last = abs(Z[-1]) <= tolerance
        symmetric = (
            amax(
                abs(
                    self.concentration_matrix() - transpose(self.concentration_matrix())
                )
            )
            <= tolerance
        )
        return bool(orthogonal and sorted_z and zero_last and symmetric)
