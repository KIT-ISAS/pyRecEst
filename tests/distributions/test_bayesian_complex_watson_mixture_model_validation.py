import unittest

from pyrecest.backend import zeros
from pyrecest.distributions.hypersphere_subset.bayesian_complex_watson_mixture_model import (
    BayesianComplexWatsonMixtureModel,
)


class TestBayesianComplexWatsonMixtureModelFitDefaultValidation(unittest.TestCase):
    def test_fit_default_rejects_large_feature_dimension_with_value_error(self):
        observations = zeros((100, 1), dtype=complex)

        with self.assertRaisesRegex(ValueError, "D < 100"):
            BayesianComplexWatsonMixtureModel.fit_default(observations, 1)

    def test_fit_default_rejects_non_matrix_observations(self):
        observations = zeros((3,), dtype=complex)

        with self.assertRaisesRegex(ValueError, "shape"):
            BayesianComplexWatsonMixtureModel.fit_default(observations, 1)


if __name__ == "__main__":
    unittest.main()
