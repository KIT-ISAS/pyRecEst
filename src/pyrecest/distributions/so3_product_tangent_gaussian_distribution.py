"""Tangent-space Gaussian distribution on Cartesian products of SO(3)."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    amax,
    arccos,
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
    take,
    transpose,
    where,
    zeros,
)

from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution
from .nonperiodic.gaussian_distribution import GaussianDistribution


class SO3ProductTangentGaussianDistribution(AbstractBoundedDomainDistribution):
    """Gaussian distribution in a tangent chart of SO(3)^K.

    Rotations are represented as scalar-last unit quaternions ``(x, y, z, w)``.
    The mean is stored as a product point with shape ``(K, 4)`` and covariance is
    a full matrix on the flattened tangent vector in ``R^(3K)``.
    """

    def __init__(self, mu, C, num_rotations=None, check_validity=True):
        mean, inferred_num_rotations = self._as_product_point(
            mu, num_rotations=num_rotations
        )
        super().__init__(dim=3 * inferred_num_rotations)
        self.num_rotations = inferred_num_rotations
        self.mu = mean

        C = array(C, dtype=float)
        expected_shape = (self.dim, self.dim)
        assert C.shape == expected_shape, f"C must have shape {expected_shape}."
        if check_validity:
            linalg.cholesky(C)
        self.C = C

    @property
    def input_dim(self):
        return 4 * self.num_rotations

    def get_manifold_size(self):
        return pi ** (2 * self.num_rotations)

    @staticmethod
    def _normalize_quaternions(quaternions):
        quaternions = array(quaternions, dtype=float)
        assert quaternions.shape[-1] == 4, "SO(3) quaternions must have length 4."

        norms = linalg.norm(quaternions, axis=-1)
        assert all(norms > 0.0), "SO(3) quaternions must be nonzero."

        normalized = quaternions / reshape(norms, tuple(norms.shape) + (1,))
        return where(normalized[..., -1:] < 0.0, -normalized, normalized)

    @staticmethod
    def _as_product_point(rotations, num_rotations=None):
        rotations = array(rotations, dtype=float)

        if ndim(rotations) == 1:
            assert (
                rotations.shape[0] % 4 == 0
            ), "Flattened SO(3)^K rotations need 4 entries per component."
            inferred_num_rotations = rotations.shape[0] // 4
            rotations = reshape(rotations, (inferred_num_rotations, 4))
        elif ndim(rotations) == 2:
            assert rotations.shape[-1] == 4, "SO(3) quaternions must have length 4."
            inferred_num_rotations = rotations.shape[0]
        else:
            raise ValueError("A product point must have shape (K, 4) or (4 * K,).")

        if num_rotations is not None and inferred_num_rotations != num_rotations:
            raise ValueError("num_rotations does not match input shape.")

        return (
            SO3ProductTangentGaussianDistribution._normalize_quaternions(rotations),
            inferred_num_rotations,
        )

    @staticmethod
    def _as_product_batch(rotations, num_rotations=None):
        rotations = array(rotations, dtype=float)

        if ndim(rotations) == 1:
            assert (
                rotations.shape[0] % 4 == 0
            ), "Flattened SO(3)^K rotations need 4 entries per component."
            inferred_num_rotations = rotations.shape[0] // 4
            rotations = reshape(rotations, (1, inferred_num_rotations, 4))
        elif ndim(rotations) == 2:
            if rotations.shape[-1] == 4 and (
                num_rotations is None or rotations.shape[0] == num_rotations
            ):
                inferred_num_rotations = rotations.shape[0]
                rotations = reshape(rotations, (1, inferred_num_rotations, 4))
            else:
                assert (
                    rotations.shape[-1] % 4 == 0
                ), "Flattened SO(3)^K rotations need 4 entries per component."
                inferred_num_rotations = rotations.shape[-1] // 4
                rotations = reshape(
                    rotations, (rotations.shape[0], inferred_num_rotations, 4)
                )
        elif ndim(rotations) == 3:
            assert rotations.shape[-1] == 4, "SO(3) quaternions must have length 4."
            inferred_num_rotations = rotations.shape[1]
        else:
            raise ValueError(
                "SO(3)^K rotations must have shape (K, 4), (4 * K,), "
                "(n, K, 4), or (n, 4 * K)."
            )

        if num_rotations is not None and inferred_num_rotations != num_rotations:
            raise ValueError("num_rotations does not match input shape.")

        return (
            SO3ProductTangentGaussianDistribution._normalize_quaternions(rotations),
            inferred_num_rotations,
        )

    @staticmethod
    def _as_tangent_batch(tangent_vectors, num_rotations=None):
        tangent_vectors = array(tangent_vectors, dtype=float)

        if ndim(tangent_vectors) == 1:
            assert (
                tangent_vectors.shape[0] % 3 == 0
            ), "Flattened SO(3)^K tangent vectors need 3 entries per component."
            inferred_num_rotations = tangent_vectors.shape[0] // 3
            tangent_vectors = reshape(tangent_vectors, (1, inferred_num_rotations, 3))
        elif ndim(tangent_vectors) == 2:
            if tangent_vectors.shape[-1] == 3 and (
                num_rotations is None or tangent_vectors.shape[0] == num_rotations
            ):
                inferred_num_rotations = tangent_vectors.shape[0]
                tangent_vectors = reshape(
                    tangent_vectors, (1, inferred_num_rotations, 3)
                )
            else:
                assert (
                    tangent_vectors.shape[-1] % 3 == 0
                ), "Flattened SO(3)^K tangent vectors need 3 entries per component."
                inferred_num_rotations = tangent_vectors.shape[-1] // 3
                tangent_vectors = reshape(
                    tangent_vectors,
                    (tangent_vectors.shape[0], inferred_num_rotations, 3),
                )
        elif ndim(tangent_vectors) == 3:
            assert (
                tangent_vectors.shape[-1] == 3
            ), "SO(3) tangent vectors must have length 3."
            inferred_num_rotations = tangent_vectors.shape[1]
        else:
            raise ValueError(
                "SO(3)^K tangent vectors must have shape (K, 3), (3 * K,), "
                "(n, K, 3), or (n, 3 * K)."
            )

        if num_rotations is not None and inferred_num_rotations != num_rotations:
            raise ValueError("num_rotations does not match input shape.")

        return tangent_vectors, inferred_num_rotations

    @staticmethod
    def _quaternion_conjugate(quaternions):
        quaternions = SO3ProductTangentGaussianDistribution._normalize_quaternions(
            quaternions
        )
        return quaternions * array([-1.0, -1.0, -1.0, 1.0])

    @staticmethod
    def _quaternion_multiply(left, right):
        left = SO3ProductTangentGaussianDistribution._normalize_quaternions(left)
        right = SO3ProductTangentGaussianDistribution._normalize_quaternions(right)

        x1, y1, z1, w1 = left[..., 0], left[..., 1], left[..., 2], left[..., 3]
        x2, y2, z2, w2 = right[..., 0], right[..., 1], right[..., 2], right[..., 3]
        product = stack(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            -1,
        )
        return SO3ProductTangentGaussianDistribution._normalize_quaternions(product)

    @staticmethod
    def _exp_map_so3_identity(tangent_vectors):
        angles = linalg.norm(tangent_vectors, axis=-1)
        angles_col = reshape(angles, tuple(angles.shape) + (1,))
        safe_angles = where(angles_col > 1e-12, angles_col, 1.0)
        vector_scale = where(
            angles_col > 1e-12,
            sin(0.5 * angles_col) / safe_angles,
            0.5 - angles_col**2 / 48.0,
        )
        return SO3ProductTangentGaussianDistribution._normalize_quaternions(
            concatenate(
                (tangent_vectors * vector_scale, cos(0.5 * angles_col)), axis=-1
            )
        )

    @staticmethod
    def _log_map_so3_identity(rotations):
        rotations = SO3ProductTangentGaussianDistribution._normalize_quaternions(
            rotations
        )
        vector_part = rotations[..., :3]
        scalar_part = clip(rotations[..., 3], -1.0, 1.0)
        vector_norm = linalg.norm(vector_part, axis=-1)
        angles = 2.0 * arctan2(vector_norm, scalar_part)
        vector_norm_col = reshape(vector_norm, tuple(vector_norm.shape) + (1,))
        safe_norm = where(vector_norm_col > 1e-12, vector_norm_col, 1.0)
        scale = where(
            vector_norm_col > 1e-12,
            reshape(angles, tuple(angles.shape) + (1,)) / safe_norm,
            2.0,
        )
        return vector_part * scale

    @staticmethod
    def exp_map(tangent_vectors, base=None, num_rotations=None):
        """Map flattened tangent vectors to SO(3)^K product quaternions."""
        tangent_vectors, inferred_num_rotations = (
            SO3ProductTangentGaussianDistribution._as_tangent_batch(
                tangent_vectors, num_rotations=num_rotations
            )
        )

        if base is None:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                stack(
                    [
                        array([0.0, 0.0, 0.0, 1.0])
                        for _ in range(inferred_num_rotations)
                    ],
                    0,
                ),
                num_rotations=inferred_num_rotations,
            )
        else:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                base, num_rotations=inferred_num_rotations
            )

        components = []
        for i in range(inferred_num_rotations):
            delta = SO3ProductTangentGaussianDistribution._exp_map_so3_identity(
                tangent_vectors[:, i, :]
            )
            components.append(
                SO3ProductTangentGaussianDistribution._quaternion_multiply(
                    base[i, :], delta
                )
            )
        return stack(components, 1)

    @staticmethod
    def log_map(rotations, base=None, num_rotations=None):
        """Map SO(3)^K product quaternions to flattened tangent vectors."""
        rotations, inferred_num_rotations = (
            SO3ProductTangentGaussianDistribution._as_product_batch(
                rotations, num_rotations=num_rotations
            )
        )

        if base is None:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                stack(
                    [
                        array([0.0, 0.0, 0.0, 1.0])
                        for _ in range(inferred_num_rotations)
                    ],
                    0,
                ),
                num_rotations=inferred_num_rotations,
            )
        else:
            base, _ = SO3ProductTangentGaussianDistribution._as_product_point(
                base, num_rotations=inferred_num_rotations
            )

        tangent_components = []
        for i in range(inferred_num_rotations):
            relative_rotation = (
                SO3ProductTangentGaussianDistribution._quaternion_multiply(
                    SO3ProductTangentGaussianDistribution._quaternion_conjugate(
                        base[i, :]
                    ),
                    rotations[:, i, :],
                )
            )
            tangent_components.append(
                SO3ProductTangentGaussianDistribution._log_map_so3_identity(
                    relative_rotation
                )
            )

        tangent_vectors = stack(tangent_components, 1)
        return reshape(
            tangent_vectors, (tangent_vectors.shape[0], 3 * inferred_num_rotations)
        )

    @staticmethod
    def geodesic_distance(rotation_a, rotation_b, reduce=True, num_rotations=None):
        """Return component-wise or summed SO(3)^K geodesic distances."""
        rotation_a, inferred_num_rotations = (
            SO3ProductTangentGaussianDistribution._as_product_batch(
                rotation_a, num_rotations=num_rotations
            )
        )
        rotation_b, _ = SO3ProductTangentGaussianDistribution._as_product_batch(
            rotation_b, num_rotations=inferred_num_rotations
        )

        inner = abs(sum(rotation_a * rotation_b, axis=-1))
        distances = 2.0 * arccos(clip(inner, 0.0, 1.0))
        if reduce:
            return sum(distances, axis=-1)
        return distances

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last quaternions to rotation matrices."""
        quaternions = SO3ProductTangentGaussianDistribution._normalize_quaternions(
            quaternions
        )
        x = quaternions[..., 0]
        y = quaternions[..., 1]
        z = quaternions[..., 2]
        w = quaternions[..., 3]

        row_0 = stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            -1,
        )
        row_1 = stack(
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            -1,
        )
        row_2 = stack(
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
            -1,
        )
        return stack((row_0, row_1, row_2), -2)

    def pdf(self, xs):
        """Evaluate the tangent Gaussian density at SO(3)^K rotations."""
        tangent_vectors = self.tangent_vectors(xs)
        gaussian = GaussianDistribution(zeros(self.dim), self.C, check_validity=False)
        return gaussian.pdf(tangent_vectors)

    def ln_pdf(self, xs):
        """Evaluate the natural logarithm of the tangent Gaussian density."""
        residual = self.tangent_vectors(xs)
        precision = linalg.inv(self.C)
        quadratic = sum(residual * matmul(residual, precision), axis=-1)
        log_det = 2.0 * sum(log(diag(linalg.cholesky(self.C))))
        return -0.5 * (self.dim * log(2.0 * pi) + log_det + quadratic)

    def tangent_vectors(self, rotations):
        """Return flattened log-map coordinates around the distribution mean."""
        return self.log_map(rotations, base=self.mu, num_rotations=self.num_rotations)

    def tangent_vectors_product(self, rotations):
        """Return log-map coordinates with shape ``(n, K, 3)``."""
        tangent_vectors = self.tangent_vectors(rotations)
        return reshape(
            tangent_vectors, (tangent_vectors.shape[0], self.num_rotations, 3)
        )

    def sample_tangent(self, n):
        """Draw tangent-space Gaussian samples with shape ``(n, 3 * K)``."""
        samples = random.multivariate_normal(mean=zeros(self.dim), cov=self.C, size=n)
        if ndim(samples) == 1:
            return reshape(samples, (1, self.dim))
        return samples

    def sample(self, n):
        """Draw ``n`` SO(3)^K samples as scalar-last unit quaternions."""
        return self.exp_map(
            self.sample_tangent(n), base=self.mu, num_rotations=self.num_rotations
        )

    def mean(self):
        """Return the mean product rotation with shape ``(K, 4)``."""
        return self.mu

    def mode(self):
        """Return the modal product rotation with shape ``(K, 4)``."""
        return self.mu

    def mean_rotation_matrices(self):
        """Return rotation matrices of the mean product rotation."""
        return self.as_rotation_matrices(self.mu)

    def covariance(self):
        """Return the full ``(3K, 3K)`` tangent covariance matrix."""
        return self.C

    def set_mean(self, new_mean):
        """Return a copy with a replaced mean product rotation."""
        return self.set_mode(new_mean)

    def set_mode(self, new_mode):
        """Return a copy with a replaced modal product rotation."""
        return self.__class__(new_mode, self.C, check_validity=False)

    def marginalize_rotation(self, rotation_index):
        """Return the one-component SO(3) tangent Gaussian marginal."""
        return self.marginalize_rotations([rotation_index])

    def marginalize_rotations(self, rotation_indices):
        """Return the marginal over selected SO(3) components."""
        if isinstance(rotation_indices, int):
            rotation_indices = [rotation_indices]
        rotation_indices_array = array(rotation_indices)
        tangent_indices = [
            3 * rotation_index + offset
            for rotation_index in rotation_indices
            for offset in range(3)
        ]
        tangent_indices = array(tangent_indices)
        new_covariance = take(
            take(self.C, tangent_indices, axis=0), tangent_indices, axis=1
        )
        new_mean = reshape(
            take(self.mu, rotation_indices_array, axis=0),
            (len(rotation_indices), 4),
        )
        return SO3ProductTangentGaussianDistribution(
            new_mean,
            new_covariance,
            num_rotations=len(rotation_indices),
            check_validity=False,
        )

    def distance_to(self, rotations, reduce=True):
        """Return geodesic distances from the mean to ``rotations``."""
        return self.geodesic_distance(
            self.mu, rotations, reduce=reduce, num_rotations=self.num_rotations
        )

    def is_valid(self, tolerance=1e-6):
        """Return whether the mean and covariance have valid SO(3)^K dimensions."""
        mean_norms = linalg.norm(self.mu, axis=-1)
        covariance_is_symmetric = amax(abs(self.C - transpose(self.C))) <= tolerance
        return bool(
            amax(abs(mean_norms - 1.0)) <= tolerance
            and all(self.mu[:, -1] >= -tolerance)
            and covariance_is_symmetric
        )

    @staticmethod
    def from_covariance_diagonal(mu, covariance_diagonal):
        """Create a product tangent Gaussian from a diagonal covariance vector."""
        return SO3ProductTangentGaussianDistribution(mu, diag(covariance_diagonal))
