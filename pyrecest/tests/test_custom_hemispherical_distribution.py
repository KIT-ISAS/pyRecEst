import unittest
import warnings
import numpy as np
from bingham_distribution import BinghamDistribution
from custom_hemispherical_distribution import CustomHemisphericalDistribution
from vmf_distribution import VMFDistribution


class CustomHemisphericalDistributionTest(unittest.TestCase):

    def test_simple_distribution_2D(self):
        M = np.eye(3)
        Z = np.array([-2, -0.5, 0])
        bd = BinghamDistribution(Z, M)
        chhd = CustomHemisphericalDistribution.from_distribution(bd)
        
        p = chhd.pdf(np.asarray([1,0,0]))
        self.assertIs(p.size, 1)
        
        np.random.seed(10)
        points = np.random.randn(100, 3)
        points = points[points[:, 2] >= 0, :]
        points /= np.linalg.norm(points, axis=1, keepdims = True)
        
        self.assertAlmostEqual(np.allclose(chhd.pdf(points), 2 * bd.pdf(points), atol=1e-5), True)
    
    def test_integral_bingham_s2(self):
        M = np.eye(3)
        Z = np.array([-2, -0.5, 0])
        bd = BinghamDistribution(Z, M)
        chhd = CustomHemisphericalDistribution.from_distribution(bd)
        chhd.pdf(np.asarray([1,0,0]))
        self.assertAlmostEqual(chhd.integral_numerical(), 1, delta=1e-4)

    def test_warning_asymmetric(self):
        vmf = VMFDistribution(np.array([0, 0, 1]), 10)
        with warnings.catch_warnings(record=True) as warning_list:
            CustomHemisphericalDistribution.from_distribution(vmf)
            # Check the warning message
            warning_message = 'You are creating a CustomHyperhemispherical distribution based on a distribution on the full hypersphere. ' + \
                            'Using numerical integration to calculate the normalization constant.'
            assert str(warning_list[-1].message) == warning_message
    
if __name__ == "__main__":
    unittest.main()
