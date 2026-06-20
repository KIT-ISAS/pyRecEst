import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)


class GaussianMixtureConstructorTest(unittest.TestCase):
    def test_rejects_empty_component_list(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            GaussianMixture([], array([]))

    def test_rejects_non_gaussian_components(self):
        component = LinearDiracDistribution(array([0.0]), array([1.0]))

        with self.assertRaisesRegex(ValueError, "GaussianDistribution"):
            GaussianMixture([component], array([1.0]))


if __name__ == "__main__":
    unittest.main()
