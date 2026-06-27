"""Regression tests for score_to_cost output validation."""

import unittest

import numpy as np
from pyrecest.backend import __backend_name__, array
from pyrecest.utils import solve_multisession_assignment_from_similarity


class TestMultiSessionAssignmentSimilarityScoreToCost(unittest.TestCase):
    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_accepts_numeric_cost_matrix_from_score_to_cost(self):
        result = solve_multisession_assignment_from_similarity(
            [array([[0.25]], dtype=float)],
            start_cost=1.0,
            end_cost=1.0,
            score_to_cost=lambda scores: 1.0 - np.asarray(scores),
        )

        self.assertEqual(result.tracks, [{0: 0, 1: 0}])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_boolean_cost_matrix_from_score_to_cost(self):
        pairwise_scores = [array([[0.25]], dtype=float)]

        with self.assertRaisesRegex(
            ValueError,
            "score_to_cost must return real numeric cost matrices",
        ):
            solve_multisession_assignment_from_similarity(
                pairwise_scores,
                start_cost=1.0,
                end_cost=1.0,
                score_to_cost=lambda scores: np.asarray(scores) > 0.0,
            )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_text_cost_matrix_from_score_to_cost(self):
        pairwise_scores = [array([[0.25]], dtype=float)]

        with self.assertRaisesRegex(
            ValueError,
            "score_to_cost must return real numeric cost matrices",
        ):
            solve_multisession_assignment_from_similarity(
                pairwise_scores,
                start_cost=1.0,
                end_cost=1.0,
                score_to_cost=lambda scores: np.asarray([["bad"]]),
            )


if __name__ == "__main__":
    unittest.main()
