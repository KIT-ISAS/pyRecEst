import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class VectorLogDensityDistribution(AbstractManifoldSpecificDistribution):
    def __init__(self):
        super().__init__(dim=1)

    @property
    def input_dim(self):
        return 1

    def get_manifold_size(self):
        return 1.0

    def pdf(self, xs):
        return array(1.0)

    def ln_pdf(self, xs):
        return array([0.0, -1.0])

    def mean(self):
        return array([0.0])


class MetropolisHastingsScalarValidationTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="This regression test targets the non-JAX MH implementation.",
    )
    def test_sample_metropolis_hastings_rejects_vector_log_density(self):
        distribution = VectorLogDensityDistribution()

        def proposal(x):
            return x

        with self.assertRaisesRegex(ValueError, "scalar"):
            distribution.sample_metropolis_hastings(
                n=1,
                burn_in=0,
                skipping=1,
                proposal=proposal,
                start_point=array([0.0]),
            )


if __name__ == "__main__":
    unittest.main()
