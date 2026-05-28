import unittest

import numpy as np
import numpy.testing as npt
import scipy.linalg

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class TestPartiallyWrappedNormalDistribution(unittest.TestCase):
    def setUp(self) -> None:
        self.mu = array([5.0, 1.0])
        self.C = array([[2.0, 1.0], [1.0, 1.0]])
        self.dist_2d = PartiallyWrappedNormalDistribution(self.mu, self.C, 1)

    def test_pdf(self):
        # Use distinct rows so that a tile/repeat swap in the implementation would
        # mix contributions between different input points and produce wrong values.
        xs = array([[0.5, 1.0], [1.5, 0.5], [2.0, 2.0]])
        result = self.dist_2d.pdf(xs)
        self.assertEqual(result.shape, (3,))
        # Each row in the batch must match its individually evaluated value.
        for i in range(xs.shape[0]):
            npt.assert_allclose(
                result[i],
                self.dist_2d.pdf(xs[i : i + 1])[0],  # noqa: E203
                rtol=1e-10,
            )

    def test_accepts_python_sequence_parameters_and_query_points(self):
        dist = PartiallyWrappedNormalDistribution(
            [0.0, 1.0], [[1.0, 0.1], [0.1, 2.0]], 0
        )
        query_points = [[0.0, 1.0], [1.0, 2.0]]

        npt.assert_allclose(dist.pdf(query_points), dist.pdf(array(query_points)))

    def test_pdf_interprets_one_dimensional_sequences_as_multiple_scalar_points(self):
        dist = PartiallyWrappedNormalDistribution([0.0], [[1.0]], 1)
        query_points = [0.1, 0.2, 0.3]
        expected_points = array([[0.1], [0.2], [0.3]])

        values = dist.pdf(query_points)

        self.assertEqual(values.shape, (3,))
        npt.assert_allclose(values, dist.pdf(expected_points))

    def test_set_mean_accepts_python_sequence_and_wraps_periodic_part(self):
        dist = PartiallyWrappedNormalDistribution(
            [5.0, 1.0], [[2.0, 1.0], [1.0, 1.0]], 1
        )

        shifted = dist.set_mean([7.0, 3.0])

        npt.assert_allclose(shifted.mu, [7.0 - 2.0 * pi, 3.0])
        npt.assert_allclose(dist.mu, [5.0, 1.0])

    def test_set_mean_rejects_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "new_mean"):
            self.dist_2d.set_mean([1.0])

    def test_linear_mean_is_empty_for_fully_periodic_distribution(self):
        dist = PartiallyWrappedNormalDistribution(
            [0.0, 1.0], [[1.0, 0.1], [0.1, 2.0]], np.int64(2)
        )

        self.assertEqual(dist.linear_mean().shape, (0,))
        self.assertEqual(dist.linear_covariance().shape, (0, 0))

    def test_constructor_rejects_invalid_parameters(self):
        invalid_parameters = [
            ("bound_dim", ([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], True)),
            ("bound_dim", ([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], 1.5)),
            ("bound_dim", ([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], -1)),
            ("bound_dim", ([0.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], 3)),
            ("mu", ([[0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], 1)),
            ("C must match", ([0.0, 1.0], [[1.0]], 1)),
            ("C must be symmetric", ([0.0, 1.0], [[1.0, 2.0], [0.0, 1.0]], 1)),
            (
                "positive definite",
                ([0.0, 1.0], [[1.0, 2.0], [2.0, 1.0]], 1),
            ),
        ]

        for message, args in invalid_parameters:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                PartiallyWrappedNormalDistribution(*args)

    def test_hybrid_mean_2d(self):
        npt.assert_allclose(self.dist_2d.hybrid_mean(), self.mu)

    def test_hybrid_mean_4d(self):
        mu = array([5.0, 1.0, 3.0, 4.0])
        C = array(
            scipy.linalg.block_diag(
                [[2.0, 1.0], [1.0, 1.0]],
                [[2.0, 1.0], [1.0, 1.0]],
            )
        )
        dist = PartiallyWrappedNormalDistribution(mu, C, 2)
        npt.assert_allclose(dist.hybrid_mean(), mu)

    def test_hybrid_moment_2d(self):
        # Validate against precalculated values
        npt.assert_allclose(
            self.dist_2d.hybrid_moment(),
            [0.10435348, -0.35276852, self.mu[-1]],
            rtol=5e-7,
        )

    def test_sample_accepts_integer_like_count(self):
        samples = self.dist_2d.sample(array(4))

        self.assertEqual(samples.shape, (4, 2))

    def test_sample_rejects_invalid_count(self):
        for n in (0, -1, 1.5, True, [3]):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    self.dist_2d.sample(n)


if __name__ == "__main__":
    unittest.main()
