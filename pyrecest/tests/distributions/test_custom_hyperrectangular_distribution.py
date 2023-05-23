import unittest

import numpy as np
from pyrecest.distributions.custom_hyperrectangular_distribution import (
    CustomHyperrectangularDistribution,
)
from pyrecest.distributions.nonperiodic.hyperrectangular_uniform_distribution import (
    HyperrectangularUniformDistribution,
)


class TestCustomHyperrectangularDistribution(unittest.TestCase):
    def test_basic(self):
        hud = HyperrectangularUniformDistribution(np.array([[1, 3], [2, 5]]))
        cd = CustomHyperrectangularDistribution(hud.pdf, hud.bounds)
        x_mesh, y_mesh = np.meshgrid(np.linspace(1, 3, 50), np.linspace(2, 5, 50))
        self.assertTrue(
            np.allclose(
                cd.pdf(np.column_stack((x_mesh.ravel(), y_mesh.ravel()))),
                1 / 6 * np.ones(50**2),
            )
        )
