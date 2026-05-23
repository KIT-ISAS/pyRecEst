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

    def test_scalar_pdf_accepts_scalar_and_plain_sequence_inputs(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        scalar_pdf = dist.pdf(0.2)
        list_pdf = dist.pdf([0.2, 0.4])
        matrix_pdf = dist.pdf(array([[0.2], [0.4]]))

        self.assertEqual(scalar_pdf.shape, (1,))
        self.assertEqual(list_pdf.shape, (2,))
        npt.assert_allclose(list_pdf, matrix_pdf)
        npt.assert_allclose(scalar_pdf, matrix_pdf[:1])

    def test_vector_pdf_accepts_single_point_sequence(self):
        dist = HypertoroidalWNDistribution(array([0.3, 0.4]), array([[0.7, 0.0], [0.0, 0.5]]))

        one_point_pdf = dist.pdf([0.2, 0.5])
        matrix_pdf = dist.pdf(array([[0.2, 0.5]]))

        self.assertEqual(one_point_pdf.shape, (1,))
        npt.assert_allclose(one_point_pdf, matrix_pdf)

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

    def test_scalar_shift_accepts_python_scalar(self):
        dist = HypertoroidalWNDistribution(0.3, 0.7)

        shifted = dist.shift(0.5)

        npt.assert_allclose(shifted.mu, array([0.8]))
        npt.assert_allclose(dist.mu, array([0.3]))

    def test_set_mode_wraps_to_fundamental_domain(self):
        dist = HypertoroidalWNDistribution(array([0.3, 0.4]), array([[0.7, 0.0], [0.0, 0.5]]))

        updated = dist.set_mode(array([2.0 * pi + 0.1, -0.2]))

        npt.assert_allclose(updated.mu, mod(array([0.1, -0.2]), 2.0 * pi))
        npt.assert_allclose(dist.mu, array([0.3, 0.4]))


if __name__ == "__main__":
    unittest.main()
