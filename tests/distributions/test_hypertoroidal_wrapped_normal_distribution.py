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

    def test_pdf_accepts_scalar_and_1d_batch_for_one_dimensional_distribution(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        scalar_value = dist.pdf(0.3)
        batch_values = dist.pdf([0.3, 0.4])

        self.assertEqual(scalar_value.shape, (1,))
        self.assertEqual(batch_values.shape, (2,))
        npt.assert_allclose(scalar_value, dist.pdf(array([[0.3]])))
        npt.assert_allclose(batch_values, dist.pdf(array([[0.3], [0.4]])))

    def test_pdf_accepts_list_single_point_for_multidimensional_distribution(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.3]])

        from_list = dist.pdf([1.0, 2.0])
        from_matrix = dist.pdf([[1.0, 2.0]])

        self.assertEqual(from_list.shape, (1,))
        npt.assert_allclose(from_list, from_matrix)

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

    def test_shift_accepts_scalar_for_one_dimensional_distribution(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        shifted = dist.shift(0.25)

        self.assertEqual(shifted.dim, 1)
        npt.assert_allclose(shifted.mu, array([0.55]))
        npt.assert_allclose(dist.mu, array([0.3]))

    def test_shift_accepts_list_for_multidimensional_distribution(self):
        dist = HypertoroidalWNDistribution([1.0, 2.0], [[0.5, 0.1], [0.1, 0.6]])

        shifted = dist.shift([0.25, -0.5])

        npt.assert_allclose(shifted.mu, mod(array([1.25, 1.5]), 2.0 * pi))
        npt.assert_allclose(dist.mu, array([1.0, 2.0]))

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


if __name__ == "__main__":
    unittest.main()
