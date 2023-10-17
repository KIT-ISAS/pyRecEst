from pyrecest.backend import meshgrid
from pyrecest.backend import linspace
from pyrecest.backend import array
import unittest
from warnings import catch_warnings, simplefilter

import numpy as np
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture


class LinearMixtureTest(unittest.TestCase):
    def test_constructor_warning(self):
        with catch_warnings(record=True) as w:
            simplefilter("always")
            LinearMixture(
                [
                    GaussianDistribution(array(1), array(1)),
                    GaussianDistribution(array(50), array(1)),
                ],
                array([0.3, 0.7]),
            )
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertIn(
                "For mixtures of Gaussians, consider using GaussianMixture.",
                str(w[-1].message),
            )

    def test_pdf(self):
        gm1 = GaussianDistribution(array([1, 1]), np.diag([2, 3]))
        gm2 = GaussianDistribution(-array([3, 1]), np.diag([2, 3]))

        with catch_warnings():
            simplefilter("ignore", category=UserWarning)
            lm = LinearMixture([gm1, gm2], array([0.3, 0.7]))

        x, y = meshgrid(linspace(-2, 2, 100), linspace(-2, 2, 100))
        points = np.column_stack((x.ravel(), y.ravel()))

        np.testing.assert_allclose(
            lm.pdf(points), 0.3 * gm1.pdf(points) + 0.7 * gm2.pdf(points), atol=1e-20
        )


if __name__ == "__main__":
    unittest.main()