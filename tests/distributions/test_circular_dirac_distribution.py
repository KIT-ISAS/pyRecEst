import unittest

import numpy.testing as npt
from pyrecest.backend import array, ones, pi
from pyrecest.distributions import CircularDiracDistribution, VonMisesDistribution


class TestCircularDiracDistribution(unittest.TestCase):
    def test_column_vector_locations_are_stored_flat(self):
        d = array([[0.0], [pi / 2.0], [pi]])
        w = ones(3) / 3.0

        wd = CircularDiracDistribution(d, w)

        self.assertEqual(wd.d.shape, (3,))
        npt.assert_allclose(wd.d, d[:, 0])
        npt.assert_allclose(wd.w, w)

    def test_column_vector_locations_with_uniform_weights_are_stored_flat(self):
        d = array([[0.0], [pi / 2.0], [pi]])

        wd = CircularDiracDistribution(d)

        self.assertEqual(wd.d.shape, (3,))
        self.assertEqual(wd.w.shape, (3,))
        npt.assert_allclose(wd.d, d[:, 0])

    def test_rejects_multidimensional_locations_for_circular_dirac(self):
        with self.assertRaisesRegex(ValueError, "shapes of d and w"):
            CircularDiracDistribution(array([[0.0, pi / 2.0], [pi, 3.0 * pi / 2.0]]))

    def test_from_distribution_preserves_circular_dirac_type(self):
        n_particles = 5
        vm = VonMisesDistribution(array(0.2), array(1.5))

        wd = CircularDiracDistribution.from_distribution(vm, n_particles)

        self.assertIsInstance(wd, CircularDiracDistribution)
        self.assertEqual(wd.d.shape, (n_particles,))
        self.assertEqual(wd.w.shape, (n_particles,))
        npt.assert_allclose(wd.w, ones(n_particles) / n_particles)


if __name__ == "__main__":
    unittest.main()
