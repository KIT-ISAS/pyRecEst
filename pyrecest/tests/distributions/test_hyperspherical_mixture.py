import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyrecest.distributions import (
    AbstractHypersphereSubsetDistribution,
    HypersphericalMixture,
    VMFDistribution,
    WatsonDistribution,
)


class HypersphericalMixtureTest(unittest.TestCase):
    def test_pdf_3d(self):
        wad = WatsonDistribution(np.array([0, 0, 1]), -10)
        vmf = VMFDistribution(np.array([0, 0, 1]), 1)
        w = [0.3, 0.7]
        smix = HypersphericalMixture([wad, vmf], w)

        phi, theta = np.meshgrid(
            np.linspace(0, 2 * np.pi, 10), np.linspace(-np.pi / 2, np.pi / 2, 10)
        )
        points = AbstractHypersphereSubsetDistribution.polar_to_cart(
            np.stack([phi.ravel(), theta.ravel()], axis=-1)
        )

        assert_allclose(
            smix.pdf(points),
            w[0] * wad.pdf(points) + w[1] * vmf.pdf(points),
            atol=1e-10,
        )

    def test_pdf_4d(self):
        wad = WatsonDistribution(np.array([0, 0, 0, 1]), -10)
        vmf = VMFDistribution(np.array([0, 1, 0, 0]), 1)
        w = [0.3, 0.7]
        smix = HypersphericalMixture([wad, vmf], w)

        a, b, c, d = np.mgrid[-1:1:4j, -1:1:4j, -1:1:4j, -1:1:4j]
        points = np.array([a.ravel(), b.ravel(), c.ravel(), d.ravel()]).T
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))

        assert_allclose(
            smix.pdf(points),
            w[0] * wad.pdf(points) + w[1] * vmf.pdf(points),
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
