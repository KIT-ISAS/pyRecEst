import unittest

import numpy as np
from pyrecest.distributions import VMFDistribution

vectors_to_test_2d = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0] / np.sqrt(2),
        [1, 1, 2] / np.linalg.norm([1, 1, 2]),
        -np.array([1, 1, 2]) / np.linalg.norm([1, 1, 2]),
    ]
)


class TestVMFDistribution(unittest.TestCase):
    def test_init_2d(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        dist = VMFDistribution(mu, kappa)
        np.testing.assert_array_almost_equal(dist.C, 7.22562325261744e-05)

    def test_init_3d(self):
        mu = np.array([1, 1, 2, -3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        dist = VMFDistribution(mu, kappa)
        np.testing.assert_array_almost_equal(dist.C, 0.0318492506152322)

    def test_pdf_2d(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        dist = VMFDistribution(mu, kappa)

        np.testing.assert_array_almost_equal(
            dist.pdf(vectors_to_test_2d),
            np.array(
                [
                    0.00428425301914546,
                    0.00428425301914546,
                    0.254024093013817,
                    0.0232421165060131,
                    1.59154943419939,
                    3.28042788159008e-09,
                ],
            ),
        )

    def test_pdf_3d(self):
        mu = np.array([1, 1, 2, -3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        dist = VMFDistribution(mu, kappa)

        xs_unnorm = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [1, -1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
            ]
        )
        xs = xs_unnorm / np.linalg.norm(xs_unnorm, axis=1, keepdims=True)

        np.testing.assert_array_almost_equal(
            dist.pdf(xs),
            np.array(
                [
                    0.0533786916025838,
                    0.0533786916025838,
                    0.0894615936690536,
                    0.00676539409350726,
                    0.0661093769275549,
                    0.0318492506152322,
                    0.0952456142366906,
                    0.0221063629087443,
                    0.0153438863274034,
                    0.13722300001807,
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
