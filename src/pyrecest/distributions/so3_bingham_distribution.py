"""Bingham distribution on SO(3)."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    amax,
    argsort,
    array,
    column_stack,
    diag,
    dot,
    eye,
    linalg,
    log,
    ndim,
    one_hot,
    stack,
    sum,
    transpose,
    zeros,
)

from ._so3_helpers import geodesic_distance, normalize_quaternions
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
        Z = array(Z, dtype=float)
        assert Z.shape == (4,), "Z must have shape (4,)."
        return BinghamDistribution.calculate_F(Z)

    @staticmethod
    def _normalize_quaternions(quaternions):
        return normalize_quaternions(quaternions)

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
        mode = SO3BinghamDistribution._canonicalize_quaternion(mode)
        basis = []
        for column in eye(4):
            vector = column - dot(column, mode) * mode
            for existing in basis:
                vector = vector - dot(vector, existing) * existing

            norm = linalg.norm(vector)
            if norm > 1e-10:
                basis.append(vector / norm)
            if len(basis) == 3:
                break

        return column_stack((*basis, mode))

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
        concentration_matrix = array(concentration_matrix, dtype=float)
        assert concentration_matrix.shape == (
            4,
            4,
        ), "concentration_matrix must have shape (4, 4)."
        concentration_matrix = 0.5 * (
            concentration_matrix + transpose(concentration_matrix)
        )

        eigenvalues, eigenvectors = linalg.eigh(concentration_matrix)
        order = argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        Z = eigenvalues - eigenvalues[-1]
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
        Z = array(self.distFullSphere.Z, dtype=float)
        normalizer = self.calculate_normalization_constant(Z)
        epsilon = 0.001
        derivatives = []
        for idx in range(4):
            delta = epsilon * one_hot(idx, 4)
            derivatives.append(
                (
                    self.calculate_normalization_constant(Z + delta)
                    - self.calculate_normalization_constant(Z - delta)
                )
                / (2.0 * epsilon)
            )

        weights = array(derivatives, dtype=float) / normalizer
        weights = weights / sum(weights)
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

    geodesic_distance = staticmethod(geodesic_distance)

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
