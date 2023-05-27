import unittest

import numpy as np
from pyrecest.distributions.custom_hyperrectangular_distribution import (
    CustomHyperrectangularDistribution,
)
from pyrecest.distributions.nonperiodic.hyperrectangular_uniform_distribution import (
    HyperrectangularUniformDistribution,
)


class TestCustomHyperrectangularDistribution(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array([[1, 3], [2, 5]])
        self.hud = HyperrectangularUniformDistribution(self.bounds)
        self.cd = CustomHyperrectangularDistribution(self.hud.pdf, self.hud.bounds)

    def test_object_creation(self):
        """Test that a CustomHyperrectangularDistribution object is successfully created."""
        self.assertIsInstance(
            self.cd,
            CustomHyperrectangularDistribution,
            "CustomHyperrectangularDistribution object creation failed.",
        )

    def test_pdf_method(self):
        """Test that the pdf method returns correct values."""
        x_mesh, y_mesh = np.meshgrid(np.linspace(1, 3, 50), np.linspace(2, 5, 50))
        expected_pdf = 1 / 6 * np.ones(50**2)
        calculated_pdf = self.cd.pdf(np.column_stack((x_mesh.ravel(), y_mesh.ravel())))
        self.assertTrue(
            np.allclose(calculated_pdf, expected_pdf),
            "PDF calculated values do not match the expected values.",
        )


if __name__ == "__main__":
    unittest.main()
