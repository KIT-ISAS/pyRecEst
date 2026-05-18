import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy.testing as npt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import abs, allclose, array, exp, linspace
from pyrecest.distributions import VonMisesDistribution


class TestVonMisesDistribution(unittest.TestCase):
    def test_vm_init(self):
        dist1 = VonMisesDistribution(0, 1)
        dist2 = VonMisesDistribution(2, 1)
        self.assertEqual(dist1.kappa, dist2.kappa)
        self.assertNotEqual(dist1.mu, dist2.mu)

    def test_pdf(self):
        dist = VonMisesDistribution(2, 1)
        xs = linspace(1, 7, 7)
        npt.assert_array_almost_equal(
            dist.pdf(xs),
            array(
                [
                    0.215781465110296,
                    0.341710488623463,
                    0.215781465110296,
                    0.0829150854731715,
                    0.0467106111086458,
                    0.0653867888824553,
                    0.166938593220285,
                ],
            ),
        )

    def test_uniform_trigonometric_moments(self):
        dist = VonMisesDistribution(2, 0)

        npt.assert_allclose(dist.trigonometric_moment(0), array(1.0 + 0.0j))
        npt.assert_allclose(dist.trigonometric_moment(1), array(0.0 + 0.0j))
        npt.assert_allclose(dist.trigonometric_moment(2), array(0.0 + 0.0j))

    def test_from_moment_handles_zero_moment_as_uniform(self):
        dist = VonMisesDistribution.from_moment(array(0.0 + 0.0j))

        self.assertTrue(allclose(dist.kappa, array(0.0), atol=1e-14))
        self.assertTrue(allclose(dist.mu, array(0.0), atol=1e-14))
        self.assertTrue(allclose(dist.trigonometric_moment(1), array(0.0 + 0.0j)))

    def test_from_moment_round_trips_valid_moments(self):
        expected = VonMisesDistribution(array(1.3), array(4.2))
        moment = expected.trigonometric_moment(1)

        actual = VonMisesDistribution.from_moment(moment)

        self.assertTrue(allclose(actual.mu, expected.mu, atol=1e-12))
        self.assertTrue(allclose(actual.kappa, expected.kappa, rtol=1e-12, atol=1e-12))
        self.assertTrue(allclose(actual.trigonometric_moment(1), moment, atol=1e-12))

    def test_besselratio_inverse_handles_small_targets(self):
        kappa = VonMisesDistribution.besselratio_inverse(0, array(1e-14))

        self.assertTrue(allclose(kappa, array(0.0), atol=1e-14))

    def test_from_moment_rejects_infeasible_moment_magnitudes(self):
        for moment in (array(1.0 + 0.0j), array(1.01 * exp(1j * 0.4))):
            with self.subTest(moment=moment):
                with self.assertRaises(ValueError):
                    VonMisesDistribution.from_moment(moment)

    def test_besselratio_inverse_rejects_negative_targets(self):
        with self.assertRaises(ValueError):
            VonMisesDistribution.besselratio_inverse(0, array(-0.1))

    def test_convolution_of_uniform_components_remains_uniform(self):
        first = VonMisesDistribution(array(0.3), array(0.0))
        second = VonMisesDistribution(array(0.4), array(2.0))

        convolved = first.convolve(second)

        self.assertTrue(allclose(convolved.kappa, array(0.0), atol=1e-14))
        self.assertTrue(allclose(abs(convolved.trigonometric_moment(1)), array(0.0)))

    def test_plot(self):
        matplotlib.pyplot.close("all")
        matplotlib.use("Agg")
        vm = VonMisesDistribution(0, 1)
        vm.plot()
        plt.close()


if __name__ == "__main__":
    unittest.main()
