import unittest

import numpy.testing as npt
from pyrecest.backend import array, linalg, ones, pi, to_numpy
from pyrecest.distributions import VonMisesFisherDistribution


def _as_float(value) -> float:
    try:
        value_np = to_numpy(value)
    except AttributeError:
        value_np = value
    if hasattr(value_np, "item"):
        return float(value_np.item())
    return float(value_np)


class TestVonMisesFisherUniformConvolution(unittest.TestCase):
    def test_convolve_with_uniform_rhs_is_uniform_without_zonal_assertion(self):
        mu = array([1.0, 2.0, 3.0])
        mu = mu / linalg.norm(mu)
        concentrated = VonMisesFisherDistribution(mu, 4.0)

        # The stored direction of a uniform vMF is arbitrary and is deliberately
        # not zonal around the final coordinate axis. It is still zonal around
        # every axis mathematically, so convolution must not reject it.
        uniform = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 0.0)

        convolved = concentrated.convolve(uniform)

        self.assertAlmostEqual(_as_float(convolved.kappa), 0.0, places=12)
        expected_density = 1.0 / (4.0 * pi)
        points = array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        npt.assert_allclose(
            to_numpy(convolved.pdf(points)),
            to_numpy(expected_density * ones(3)),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_convolve_with_uniform_lhs_is_uniform(self):
        uniform = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 0.0)
        zonal = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 2.0)

        convolved = uniform.convolve(zonal)

        self.assertAlmostEqual(_as_float(convolved.kappa), 0.0, places=12)
        self.assertAlmostEqual(
            _as_float(convolved.pdf(array([0.0, 1.0, 0.0]))),
            1.0 / (4.0 * pi),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
