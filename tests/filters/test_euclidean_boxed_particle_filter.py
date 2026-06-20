import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones, random, to_numpy
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.filters.euclidean_boxed_particle_filter import (
    BoxedParticleFilter,
    EuclideanBoxedParticleFilter,
)


class DeterministicProposal:
    dim = 1

    def __init__(self):
        self.calls = 0

    def sample(self, n):
        self.calls += 1
        base = array([[-1.0], [0.25], [0.75]])
        repeats = int(np.ceil(int(n) / 3))
        return array(np.tile(to_numpy(base), (repeats, 1))[: int(n)])


class EuclideanBoxedParticleFilterTest(unittest.TestCase):
    def test_filter_state_remains_point_particles(self):
        pf = EuclideanBoxedParticleFilter(4, 2)
        self.assertIsInstance(pf.filter_state, LinearDiracDistribution)
        self.assertIs(BoxedParticleFilter, EuclideanBoxedParticleFilter)

    def test_rejects_bool_and_nonintegral_particle_counts(self):
        invalid_arguments = (
            (True, 1),
            (1.5, 1),
            (2, True),
            (2, 1.5),
        )

        for n_particles, dim in invalid_arguments:
            with self.subTest(n_particles=n_particles, dim=dim):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    EuclideanBoxedParticleFilter(n_particles, dim)

    def test_sampling_controls_reject_bool_and_nonintegral_values(self):
        invalid_values = (
            ("batch_size", True),
            ("batch_size", 1.5),
            ("max_sampling_iterations", True),
            ("max_sampling_iterations", 1.5),
            ("max_tries_per_particle", True),
            ("max_tries_per_particle", 1.5),
        )

        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{name} must be a positive integer",
                ):
                    EuclideanBoxedParticleFilter._validate_positive_int(value, name)

        self.assertEqual(
            EuclideanBoxedParticleFilter._validate_positive_int(
                np.int64(3),
                "batch_size",
            ),
            3,
        )

    def test_uniform_generation_places_point_particles_in_box(self):
        random.seed(1)
        pf = EuclideanBoxedParticleFilter(50, 2)

        pf.generate_boxed_particles(array([-1.0, 2.0]), array([1.0, 3.0]))

        particles = np.asarray(to_numpy(pf.filter_state.d))
        weights = np.asarray(to_numpy(pf.filter_state.w))
        self.assertEqual(particles.shape, (50, 2))
        self.assertTrue(np.all(particles[:, 0] >= -1.0))
        self.assertTrue(np.all(particles[:, 0] <= 1.0))
        self.assertTrue(np.all(particles[:, 1] >= 2.0))
        self.assertTrue(np.all(particles[:, 1] <= 3.0))
        npt.assert_allclose(weights, np.ones(50) / 50)

    def test_accept_generation_resamples_current_in_box_particles(self):
        random.seed(1)
        np.random.seed(1)
        pf = EuclideanBoxedParticleFilter(4, 1)
        pf.filter_state = LinearDiracDistribution(
            array([[-2.0], [-0.5], [0.25], [3.0]]),
            array([0.1, 0.2, 0.7, 0.0]),
        )

        pf.generate_boxed_particles(
            array([-1.0]), array([1.0]), boxed_generation_method="accept"
        )

        self.assertEqual(pf.filter_state.d.shape, (4, 1))
        self.assertTrue(np.all(to_numpy(pf.filter_state.d) >= -1.0))
        self.assertTrue(np.all(to_numpy(pf.filter_state.d) <= 1.0))
        npt.assert_allclose(pf.filter_state.w, ones(4) / 4)

    def test_reweight_constraint_zeros_outside_particles(self):
        pf = EuclideanBoxedParticleFilter(3, 1)
        pf.filter_state = LinearDiracDistribution(
            array([[0.0], [1.0], [3.0]]),
            array([0.2, 0.3, 0.5]),
        )
        pf.set_resampling_criterion(lambda _state: False)

        pf.generate_boxed_particles(
            array([0.0]),
            array([1.0]),
            boxed_generation_method="reweight",
        )

        npt.assert_allclose(to_numpy(pf.filter_state.d), [[0.0], [1.0], [3.0]])
        npt.assert_allclose(to_numpy(pf.filter_state.w), [0.4, 0.6, 0.0])

    def test_reweight_constraint_rejects_invalid_likelihood_weights(self):
        invalid_likelihoods = (
            ("nan", lambda _particles: array([1.0, np.nan, 1.0]), "finite"),
            ("inf", lambda _particles: array([1.0, np.inf, 1.0]), "finite"),
            (
                "negative",
                lambda _particles: array([1.0, -0.5, 1.0]),
                "nonnegative",
            ),
            (
                "zero-mass",
                lambda _particles: array([0.0, 0.0, 0.0]),
                "positive finite total mass",
            ),
        )

        for name, likelihood, message in invalid_likelihoods:
            with self.subTest(name=name):
                pf = EuclideanBoxedParticleFilter(3, 1)
                pf.filter_state = LinearDiracDistribution(
                    array([[0.0], [1.0], [2.0]]),
                    array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
                )

                with self.assertRaisesRegex(ValueError, message):
                    pf.reweight_by_box(
                        array([0.0]), array([2.0]), likelihood=likelihood
                    )

                npt.assert_allclose(
                    to_numpy(pf.filter_state.w),
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                )

    def test_upsample_generation_accepts_proposal_samples_in_box(self):
        proposal = DeterministicProposal()
        pf = EuclideanBoxedParticleFilter(3, 1)

        pf.generate_boxed_particles(
            array([0.0]),
            array([1.0]),
            boxed_generation_method="upsample",
            proposal_distribution=proposal,
            max_sampling_iterations=3,
        )

        particles = np.asarray(to_numpy(pf.filter_state.d)).reshape(-1)
        self.assertTrue(np.all(particles >= 0.0))
        self.assertTrue(np.all(particles <= 1.0))
        self.assertGreaterEqual(proposal.calls, 2)

    def test_inscribed_gaussian_generation_returns_particles_inside_box(self):
        np.random.seed(2)
        pf = EuclideanBoxedParticleFilter(50, 2)

        pf.generate_boxed_particles(
            array([-1.0, 2.0]),
            array([1.0, 4.0]),
            boxed_generation_method="inscribed_gaussian",
        )

        particles = np.asarray(to_numpy(pf.filter_state.d))
        self.assertEqual(particles.shape, (50, 2))
        self.assertTrue(np.all(particles[:, 0] >= -1.0))
        self.assertTrue(np.all(particles[:, 0] <= 1.0))
        self.assertTrue(np.all(particles[:, 1] >= 2.0))
        self.assertTrue(np.all(particles[:, 1] <= 4.0))
        npt.assert_allclose(pf.filter_state.w, ones(50) / 50)

    def test_identity_box_update_gates_particles(self):
        pf = EuclideanBoxedParticleFilter(4, 1)
        pf.set_resampling_criterion(lambda _state: False)
        pf.filter_state = LinearDiracDistribution(
            array([[-2.0], [-0.5], [0.25], [3.0]]),
            array([0.25, 0.25, 0.25, 0.25]),
        )

        pf.update_identity_box(array([-1.0]), array([1.0]))

        npt.assert_allclose(pf.filter_state.w, array([0.0, 0.5, 0.5, 0.0]))

    def test_predict_nonlinear_boxed_applies_prediction_then_constraint(self):
        pf = EuclideanBoxedParticleFilter(2, 1)
        pf.filter_state = LinearDiracDistribution(array([[0.0], [1.0]]))
        pf.set_resampling_criterion(lambda _state: False)

        pf.predict_nonlinear_boxed(
            lambda particles: particles + 10.0,
            box_lower=array([10.0]),
            box_upper=array([11.0]),
            boxed_generation_method="reweight",
        )

        npt.assert_allclose(to_numpy(pf.filter_state.d), [[10.0], [11.0]])
        npt.assert_allclose(to_numpy(pf.filter_state.w), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
