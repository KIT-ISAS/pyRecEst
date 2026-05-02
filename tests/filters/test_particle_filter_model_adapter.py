import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter
from pyrecest.models import LikelihoodMeasurementModel, SampleableTransitionModel


class ParticleFilterModelAdapterTest(unittest.TestCase):
    def setUp(self):
        self.particle_locations = array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        self.filter = EuclideanParticleFilter(n_particles=4, dim=2)
        self.filter.filter_state = LinearDiracDistribution(self.particle_locations)

    def test_predict_model_with_vectorized_sampleable_transition(self):
        direct_filter = copy.deepcopy(self.filter)
        model_filter = copy.deepcopy(self.filter)
        shift = array([1.0, -2.0])

        direct_filter.predict_nonlinear(
            lambda particles: particles + shift,
            noise_distribution=None,
            function_is_vectorized=True,
        )
        model_filter.predict_model(
            SampleableTransitionModel(lambda particles: particles + shift)
        )

        npt.assert_allclose(model_filter.filter_state.d, direct_filter.filter_state.d)
        npt.assert_allclose(model_filter.filter_state.w, direct_filter.filter_state.w)

    def test_predict_model_with_nonvectorized_sampleable_transition(self):
        direct_filter = copy.deepcopy(self.filter)
        model_filter = copy.deepcopy(self.filter)
        shift = array([1.0, -2.0])

        direct_filter.predict_nonlinear(
            lambda particle: particle + shift,
            noise_distribution=None,
            function_is_vectorized=False,
        )
        model_filter.predict_model(
            SampleableTransitionModel(
                lambda particle: particle + shift,
                function_is_vectorized=False,
            )
        )

        npt.assert_allclose(model_filter.filter_state.d, direct_filter.filter_state.d)
        npt.assert_allclose(model_filter.filter_state.w, direct_filter.filter_state.w)

    def test_update_model_with_likelihood_measurement_model(self):
        measurement = array([2.0])

        def likelihood(meas, particles):
            residual = particles[:, 0] - meas[0]
            return 1.0 / (1.0 + residual * residual)

        direct_filter = copy.deepcopy(self.filter)
        model_filter = copy.deepcopy(self.filter)

        random.seed(7)
        direct_filter.update_nonlinear_using_likelihood(likelihood, measurement)
        random.seed(7)
        model_filter.update_model(LikelihoodMeasurementModel(likelihood), measurement)

        npt.assert_allclose(model_filter.filter_state.d, direct_filter.filter_state.d)
        npt.assert_allclose(model_filter.filter_state.w, direct_filter.filter_state.w)


if __name__ == "__main__":
    unittest.main()
