import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, exp, mod, pi
from pyrecest.distributions import HypertoroidalWNDistribution


class TestHypertoroidalWNDistribution(unittest.TestCase):
    def test_pdf(self):
        mu = array([1, 2])
        C = array([[0.5, 0.1], [0.1, 0.3]])

        hwn = HypertoroidalWNDistribution(mu, C)

        xa = array([[0, 1, 2], [1, 2, 3]]).T
        pdf_values = hwn.pdf(xa)

        expected_values = array(
            [0.0499028191873498, 0.425359477472412, 0.0499028191873498]
        )
        npt.assert_allclose(pdf_values, expected_values, rtol=2e-6)

    def test_pdf_accepts_python_sequence_parameters_and_query_points(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.3]])

        query_points = [[1.0, 2.0], [1.1, 2.1]]

        npt.assert_allclose(dist.pdf(query_points), dist.pdf(array(query_points)))

    def test_pdf_interprets_one_dimensional_sequences_as_multiple_scalar_points(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        query_points = [0.1, 0.2, 0.3]
        expected_points = array([[0.1], [0.2], [0.3]])

        values = dist.pdf(query_points)

        self.assertEqual(values.shape, (3,))
        npt.assert_allclose(values, dist.pdf(expected_points))

    def test_scalar_parameters_are_stored_as_vector_and_matrix(self):
        dist = HypertoroidalWNDistribution(array(0.3), array(0.7))

        self.assertEqual(dist.dim, 1)
        self.assertEqual(dist.mu.shape, (1,))
        self.assertEqual(dist.C.shape, (1, 1))
        npt.assert_allclose(dist.mu, array([0.3]))
        npt.assert_allclose(dist.C, array([[0.7]]))
        npt.assert_allclose(
            dist.trigonometric_moment(1), exp(1j * array([0.3]) - 0.7 / 2)
        )

    def test_shift_returns_copy_without_mutating_original(self):
        mu = array([1.0, 2.0])
        C = array([[0.5, 0.1], [0.1, 0.6]])
        shift_by = array([0.25, 2.0 * pi - 0.5])
        dist = HypertoroidalWNDistribution(mu, C)

        shifted = dist.shift(shift_by)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, mod(mu + shift_by, 2.0 * pi))
        npt.assert_allclose(shifted.C, dist.C)

    def test_shift_accepts_plain_python_sequences(self):
        mu = array([1.0, 2.0])
        C = array([[0.5, 0.1], [0.1, 0.6]])
        dist = HypertoroidalWNDistribution(mu, C)

        shifted = dist.shift([0.25, -0.5])

        npt.assert_allclose(shifted.mu, mod(mu + array([0.25, -0.5]), 2.0 * pi))
        npt.assert_allclose(dist.mu, mu)


if __name__ == "__main__":
    unittest.main()
