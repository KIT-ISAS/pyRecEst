import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.distributions.hypersphere_subset.complex_watson_distribution import (
    ComplexWatsonDistribution,
)
from pyrecest.distributions.hypersphere_subset.bayesian_complex_watson_mixture_model import (
    BayesianComplexWatsonMixtureModel,
    _simplex_integral,
    _complex_bingham_first_order_moments,
)


def _make_unit_vectors(D, N, rng):
    Z = rng.standard_normal((D, N)) + 1j * rng.standard_normal((D, N))
    Z /= np.linalg.norm(Z, axis=0)
    return Z


class TestSimplexIntegralMixture(unittest.TestCase):
    def test_D2(self):
        a, b = 2.0, 1.0
        expected = (np.exp(a) - np.exp(b)) / (a - b)
        self.assertAlmostEqual(_simplex_integral(np.array([a, b])), expected, places=8)

    def test_D3(self):
        Lambda = np.array([2.0, 1.0, 0.0])
        expected = np.exp(2) / 2 - np.exp(1) + 0.5
        self.assertAlmostEqual(_simplex_integral(Lambda), expected, places=5)


class TestComplexBinghamFirstOrderMoments(unittest.TestCase):
    def test_uniform_zero_eigenvalues(self):
        D = 3
        Lambda = np.zeros(D)
        moments = _complex_bingham_first_order_moments(Lambda, D)
        npt.assert_allclose(moments, np.ones(D) / D, atol=1e-2)

    def test_sum_to_one(self):
        D = 4
        Lambda = np.array([-1.0, -2.0, -3.0, -4.0])
        Lambda -= Lambda.max()
        moments = _complex_bingham_first_order_moments(Lambda, D)
        self.assertAlmostEqual(moments.sum(), 1.0, places=5)

    def test_largest_eigenvalue_largest_moment(self):
        D = 3
        Lambda = np.array([0.0, -5.0, -10.0])
        moments = _complex_bingham_first_order_moments(Lambda, D)
        self.assertGreater(moments[0], moments[1])
        self.assertGreater(moments[1], moments[2])


class TestQuadraticExpectation(unittest.TestCase):
    def test_identity_input_equals_moments_sum(self):
        """E[z^H I z] = E[|z|^2] = 1 by definition."""
        D = 3
        K = 2
        rng = np.random.default_rng(7)
        B = np.zeros((D, D, K), dtype=complex)
        I_3d = np.eye(D, dtype=complex)[:, :, np.newaxis]
        E = BayesianComplexWatsonMixtureModel.quadratic_expectation(I_3d, B)
        npt.assert_allclose(E, np.ones((1, K)), atol=1e-8)

    def test_shape(self):
        D = 4
        N = 10
        K = 3
        rng = np.random.default_rng(1)
        Z = _make_unit_vectors(D, N, rng)
        dp = Z[:, np.newaxis, :] * Z.conj()[np.newaxis, :, :]
        dp = dp.reshape(D, D, N)
        B = np.zeros((D, D, K), dtype=complex)
        E = BayesianComplexWatsonMixtureModel.quadratic_expectation(dp, B)
        self.assertEqual(E.shape, (N, K))

    def test_real_output(self):
        D = 3
        N = 5
        K = 2
        rng = np.random.default_rng(2)
        Z = _make_unit_vectors(D, N, rng)
        dp = (Z[:, np.newaxis, :] * Z.conj()[np.newaxis, :, :]).reshape(D, D, N)
        B = np.zeros((D, D, K), dtype=complex)
        E = BayesianComplexWatsonMixtureModel.quadratic_expectation(dp, B)
        self.assertTrue(np.all(np.isreal(E)))


class TestBayesianComplexWatsonMixtureModelConstructor(unittest.TestCase):
    def test_basic_construction(self):
        D, K = 3, 2
        B = np.zeros((D, D, K), dtype=complex)
        concentrations = np.array([5.0, 10.0])
        alpha = np.array([1.0, 1.0])
        model = BayesianComplexWatsonMixtureModel(B, concentrations, alpha)
        self.assertEqual(model.K, K)
        self.assertEqual(model.dim, D)

    def test_non_hermitian_B_raises(self):
        D, K = 2, 1
        B = np.ones((D, D, K), dtype=complex) * (1 + 1j)
        with self.assertRaises(AssertionError):
            BayesianComplexWatsonMixtureModel(B, np.array([1.0]), np.array([1.0]))


class TestParametersDefault(unittest.TestCase):
    def test_keys_present(self):
        params = BayesianComplexWatsonMixtureModel.parameters_default(4, 3)
        self.assertIn("initial", params)
        self.assertIn("prior", params)
        self.assertIn("I", params)
        self.assertIn("B", params["initial"])
        self.assertIn("kappa", params["initial"])
        self.assertIn("alpha", params["initial"])

    def test_shapes(self):
        D, K = 5, 4
        params = BayesianComplexWatsonMixtureModel.parameters_default(D, K)
        self.assertEqual(params["initial"]["B"].shape, (D, D, K))
        self.assertEqual(len(params["initial"]["alpha"]), K)


class TestFitDefault(unittest.TestCase):
    def test_fit_returns_model(self):
        rng = np.random.default_rng(42)
        D, K, N = 3, 2, 50
        Z = _make_unit_vectors(D, N, rng)
        model, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        self.assertIsInstance(model, BayesianComplexWatsonMixtureModel)
        self.assertEqual(model.K, K)
        self.assertEqual(model.dim, D)

    def test_posterior_keys(self):
        rng = np.random.default_rng(0)
        D, K, N = 3, 2, 30
        Z = _make_unit_vectors(D, N, rng)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        for key in ("B", "kappa", "alpha", "gamma"):
            self.assertIn(key, posterior)

    def test_gamma_sums_to_one(self):
        rng = np.random.default_rng(1)
        D, K, N = 3, 2, 40
        Z = _make_unit_vectors(D, N, rng)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        gamma = posterior["gamma"]
        npt.assert_allclose(gamma.sum(axis=1), np.ones(N), atol=1e-8)

    def test_alpha_positive(self):
        rng = np.random.default_rng(2)
        D, K, N = 3, 2, 40
        Z = _make_unit_vectors(D, N, rng)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        self.assertTrue(np.all(posterior["alpha"] > 0))

    def test_kappa_nonnegative(self):
        rng = np.random.default_rng(3)
        D, K, N = 3, 3, 60
        Z = _make_unit_vectors(D, N, rng)
        _, posterior = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        self.assertTrue(np.all(posterior["kappa"] >= 0))

    def test_B_hermitian_after_fit(self):
        rng = np.random.default_rng(4)
        D, K, N = 3, 2, 40
        Z = _make_unit_vectors(D, N, rng)
        model, _ = BayesianComplexWatsonMixtureModel.fit_default(Z, K)
        for k in range(K):
            npt.assert_allclose(
                model.B[:, :, k],
                model.B[:, :, k].conj().T,
                atol=1e-8,
                err_msg=f"B[:,:,{k}] is not Hermitian",
            )

    def test_fit_two_cluster_recovery(self):
        """Fit on data from two distinct clusters should assign high weight to each."""
        rng = np.random.default_rng(99)
        D = 3
        # Two orthogonal modes
        mu1 = np.array([1.0, 0.0, 0.0], dtype=complex)
        mu2 = np.array([0.0, 1.0, 0.0], dtype=complex)
        dist1 = ComplexWatsonDistribution(mu1, 20.0)
        dist2 = ComplexWatsonDistribution(mu2, 20.0)
        Z = np.concatenate([dist1.sample(60), dist2.sample(60)], axis=1)
        K = 2
        params = BayesianComplexWatsonMixtureModel.parameters_default(D, K)
        params["I"] = 50
        _, posterior = BayesianComplexWatsonMixtureModel.fit(Z, params)
        # Both components should have non-trivial assignment
        N_k = posterior["gamma"].sum(axis=0)
        self.assertGreater(min(N_k), 10.0)


if __name__ == "__main__":
    unittest.main()
