import unittest

import numpy.testing as npt

# pylint: disable=no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, diag, exp, eye, sin
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_bingham_distribution import (
    HyperhemisphericalBinghamDistribution,
)


class HyperhemisphericalBinghamDistributionTest(unittest.TestCase):
    @parameterized.expand(
        [
            # Test case 1: Identity matrix M and Z with one negative value
            (eye(2), array([-3.0, 0.0])),
            # Test case 2: Rotated matrix M and Z with one large negative value
            (array([[cos(0.7), -sin(0.7)], [sin(0.7), cos(0.7)]]), array([-5.0, 0.0])),
            # Test case 3: 4D identity matrix and Z with multiple negative values
            (eye(4), array([-10.0, -2.0, -1.0, 0.0])),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.jax",
        reason="Not supported on this backend",
    )
    def test_pdf(self, M, Z):
        B = HyperhemisphericalBinghamDistribution(Z, M)
        if B.dim == 3:
            B.distFullSphere.F = 2.98312  # Precomputed normalization constant
        testpoints = pyrecest.backend.random.uniform(size=(20, B.dim + 1))
        testpoints /= pyrecest.backend.linalg.norm(testpoints, axis=0)
        for i in range(testpoints.shape[0]):
            expected = (
                2.0 / B.F * exp(testpoints[i] @ M @ diag(Z) @ M.T @ testpoints[i].T)
            )
            npt.assert_allclose(B.pdf(testpoints[i]), expected)


if __name__ == "__main__":
    unittest.main()
