from math import pi
from pyrecest.backend import tile
from pyrecest.backend import ones
from pyrecest.backend import array
from pyrecest.backend import allclose
from pyrecest.backend import all
from pyrecest.backend import zeros
import unittest


from pyrecest.distributions.hypertorus.toroidal_uniform_distribution import (
    ToroidalUniformDistribution,
)


class TestToroidalUniformDistribution(unittest.TestCase):
    def setUp(self):
        self.tud = ToroidalUniformDistribution()
        self.x = tile(array([[1, 2, 3, 4, 5, 6]]), (2, 1))

    def test_pdf(self):
        self.assertTrue(
            allclose(
                self.tud.pdf(self.x), (1 / (2 * pi) ** 2) * ones(self.x.shape[1])
            )
        )

    def test_shift(self):
        tud_shifted = self.tud.shift(array([1, 2]))
        self.assertTrue(
            allclose(
                tud_shifted.pdf(self.x),
                (1 / (2 * pi) ** 2) * ones(self.x.shape[1]),
            )
        )

    def test_trigonometric_moments(self):
        for k in range(4):
            self.assertTrue(
                allclose(
                    self.tud.trigonometric_moment(k),
                    self.tud.trigonometric_moment_numerical(k),
                    atol=1e-10,
                )
            )
            if k == 0:
                self.assertTrue(
                    allclose(
                        self.tud.trigonometric_moment(k), ones(2), rtol=1e-10
                    )
                )
            else:
                self.assertTrue(
                    allclose(
                        self.tud.trigonometric_moment(k), zeros(2), rtol=1e-10
                    )
                )

    def test_mean_direction(self):
        with self.assertRaises(ValueError) as cm:
            self.tud.mean_direction()

        self.assertEqual(
            str(cm.exception),
            "Hypertoroidal uniform distributions do not have a unique mean",
        )

    def test_entropy(self):
        self.assertAlmostEqual(
            self.tud.entropy(), self.tud.entropy_numerical(), delta=1e-10
        )

    def test_sampling(self):
        n = 10
        s = self.tud.sample(n)
        self.assertEqual(s.shape, (n, 2))
        self.assertTrue(all(s >= 0))
        self.assertTrue(all(s < 2 * pi))


if __name__ == "__main__":
    unittest.main()