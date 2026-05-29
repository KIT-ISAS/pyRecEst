import unittest

from pyrecest.backend import (
    allclose,
    array,
    cov,
    eye,
    linalg,
    linspace,
    mean,
    ones,
    pi,
    zeros,
)
from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import (
    StateSpaceSubdivisionGaussianDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.filters.relaxed_s3f_circular import (
    circular_weighted_mean,
    grid_probability_masses,
    predict_circular_relaxed,
    rotate_body_increment,
    uniform_circular_cell_statistics,
)
from pyrecest.filters.state_space_subdivision_filter import StateSpaceSubdivisionFilter


class RelaxedS3FCircularTest(unittest.TestCase):
    def test_covariance_inflation_is_positive_semidefinite(self):
        stats = uniform_circular_cell_statistics(12, array([0.7, -0.2]))

        for cov_matrix in stats.covariance_inflations:
            eigvals = linalg.eigvalsh(cov_matrix)
            self.assertGreaterEqual(float(eigvals[0]), -1e-12)

    def test_closed_form_statistics_match_deterministic_quadrature(self):
        n_cells = 9
        cell_idx = 3
        body_increment = array([0.6, 0.15])
        stats = uniform_circular_cell_statistics(n_cells, body_increment)
        center = stats.grid[cell_idx]
        half_width = 0.5 * stats.cell_width
        samples = linspace(center - half_width, center + half_width, 20001)
        rotated = rotate_body_increment(samples, body_increment)

        mean_quad = mean(rotated, axis=0)
        cov_quad = cov(rotated.T, bias=True)

        self.assertTrue(
            bool(allclose(stats.mean_displacements[cell_idx], mean_quad, atol=1e-5))
        )
        self.assertTrue(
            bool(allclose(stats.covariance_inflations[cell_idx], cov_quad, atol=1e-5))
        )

    def test_prediction_and_update_conserve_grid_mass(self):
        filter_ = _make_filter(10)

        predict_circular_relaxed(
            filter_,
            array([0.4, 0.1]),
            variant="r1_r2",
            process_noise_cov=eye(2) * 0.01,
        )
        self.assertTrue(
            bool(allclose(float(filter_.filter_state.gd.integrate()), 1.0, atol=1e-12))
        )

        filter_.update(
            likelihoods_linear=[
                GaussianDistribution(
                    array([0.2, 0.1]),
                    eye(2) * 0.05,
                    check_validity=False,
                )
            ]
        )
        self.assertTrue(
            bool(allclose(float(filter_.filter_state.gd.integrate()), 1.0, atol=1e-12))
        )

    def test_relaxations_vanish_as_grid_resolution_increases(self):
        body_increment = array([0.8, -0.1])
        coarse = uniform_circular_cell_statistics(8, body_increment)
        fine = uniform_circular_cell_statistics(64, body_increment)

        coarse_mean_gap = linalg.norm(
            coarse.mean_displacements - coarse.representative_displacements,
            axis=1,
        ).max()
        fine_mean_gap = linalg.norm(
            fine.mean_displacements - fine.representative_displacements,
            axis=1,
        ).max()
        coarse_cov_norm = linalg.norm(coarse.covariance_inflations, axis=(1, 2)).max()
        fine_cov_norm = linalg.norm(fine.covariance_inflations, axis=(1, 2)).max()

        self.assertLess(fine_mean_gap, coarse_mean_gap / 20.0)
        self.assertLess(fine_cov_norm, coarse_cov_norm / 20.0)

    def test_validation_errors_are_explicit(self):
        for n_cells in (0, 2.5, True, [8], float("nan")):
            with self.subTest(n_cells=n_cells):
                with self.assertRaisesRegex(ValueError, "n_cells"):
                    uniform_circular_cell_statistics(n_cells, array([0.4, 0.1]))

        with self.assertRaisesRegex(ValueError, "body_increment"):
            uniform_circular_cell_statistics(8, array([0.4]))
        with self.assertRaisesRegex(ValueError, "body_increment"):
            uniform_circular_cell_statistics(8, array([0.4, float("nan")]))
        with self.assertRaisesRegex(ValueError, "grid"):
            uniform_circular_cell_statistics(8, array([0.4, 0.1]), grid=array([0.0]))
        with self.assertRaisesRegex(ValueError, "grid"):
            uniform_circular_cell_statistics(
                2, array([0.4, 0.1]), grid=array([0.0, float("inf")])
            )
        with self.assertRaisesRegex(ValueError, "process_noise_cov"):
            predict_circular_relaxed(
                _make_filter(8),
                array([0.4, 0.1]),
                process_noise_cov=array([[float("nan"), 0.0], [0.0, 1.0]]),
            )
        with self.assertRaisesRegex(ValueError, "process_noise_cov"):
            predict_circular_relaxed(
                _make_filter(8),
                array([0.4, 0.1]),
                process_noise_cov=array([[float("inf"), 0.0], [0.0, 1.0]]),
            )

    def test_grid_probability_masses_reject_invalid_values(self):
        invalid_cases = [
            (array([1.0, float("nan")]), "finite"),
            (array([1.0, -0.5]), "nonnegative"),
            (array([0.0, 0.0]), "positive total mass"),
        ]

        for values, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    grid_probability_masses(values)
                with self.assertRaisesRegex(ValueError, message):
                    circular_weighted_mean(array([0.0, 1.0]), values)


def _make_filter(n_cells: int) -> StateSpaceSubdivisionFilter:
    grid = linspace(0.0, 2.0 * pi, n_cells, endpoint=False).reshape(-1, 1)
    gd = HypertoroidalGridDistribution(
        ones(n_cells) / (2.0 * pi),
        grid_type="custom",
        grid=grid,
    )
    gd.normalize_in_place(warn_unnorm=False)
    gaussians = [
        GaussianDistribution(
            zeros(2),
            eye(2) * 0.1,
            check_validity=False,
        )
        for _ in range(n_cells)
    ]
    state = StateSpaceSubdivisionGaussianDistribution(gd, gaussians)
    return StateSpaceSubdivisionFilter(state)


if __name__ == "__main__":
    unittest.main()
