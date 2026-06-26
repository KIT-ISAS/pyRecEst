import unittest

import numpy as np
from pyrecest.utils import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
    prune_pairwise_cost_matrix,
)


class TestCandidatePruningMatrixValidation(unittest.TestCase):
    def test_cost_matrix_rejects_boolean_entries(self):
        invalid_matrices = (
            [[True, False]],
            np.array([[True, False]]),
            np.array([[1.0, True]], dtype=object),
            np.array([[False, 2.0]], dtype=object),
        )

        for function in (candidate_mask_from_costs, prune_pairwise_cost_matrix):
            for matrix in invalid_matrices:
                with self.subTest(function=function.__name__, matrix=matrix):
                    with self.assertRaisesRegex(
                        ValueError,
                        "cost_matrix must be numeric",
                    ):
                        function(matrix)

    def test_cost_matrix_rejects_complex_entries(self):
        invalid_matrices = (
            np.array([[1.0 + 2.0j]]),
            np.array([[1.0 + 2.0j]], dtype=object),
        )

        for function in (candidate_mask_from_costs, prune_pairwise_cost_matrix):
            for matrix in invalid_matrices:
                with self.subTest(function=function.__name__, dtype=str(matrix.dtype)):
                    with self.assertRaisesRegex(
                        ValueError,
                        "cost_matrix must be real-valued numeric",
                    ):
                        function(matrix)

    def test_cost_matrix_rejects_temporal_entries(self):
        invalid_matrices = (
            np.array([["2026-06-25"]], dtype="datetime64[D]"),
            np.array([[np.timedelta64(5, "s")]]),
            np.array([[np.datetime64("2026-06-25")]], dtype=object),
            np.array([[np.timedelta64(5, "s")]], dtype=object),
        )

        for function in (candidate_mask_from_costs, prune_pairwise_cost_matrix):
            for matrix in invalid_matrices:
                with self.subTest(function=function.__name__, dtype=str(matrix.dtype)):
                    with self.assertRaisesRegex(
                        ValueError,
                        "cost_matrix must be numeric",
                    ):
                        function(matrix)

    def test_cost_matrix_rejects_negative_infinity(self):
        invalid_matrices = (
            [[0.0, -float("inf")]],
            np.array([[0.0, -np.inf]]),
        )

        for function in (candidate_mask_from_costs, prune_pairwise_cost_matrix):
            for matrix in invalid_matrices:
                with self.subTest(function=function.__name__, matrix=matrix):
                    with self.assertRaisesRegex(
                        ValueError,
                        "positive infinity",
                    ):
                        function(matrix)

    def test_cost_matrix_keeps_positive_infinity_as_missing_candidate(self):
        costs = np.array([[1.0, np.inf]])

        np.testing.assert_array_equal(
            candidate_mask_from_costs(costs),
            np.array([[True, False]]),
        )
        np.testing.assert_array_equal(prune_pairwise_cost_matrix(costs), costs)

    def test_cost_matrix_rejects_text_entries(self):
        invalid_matrices = (
            np.array([["1.0", "2.0"]]),
            np.array([[b"1.0", b"2.0"]], dtype=object),
        )

        for function in (candidate_mask_from_costs, prune_pairwise_cost_matrix):
            for matrix in invalid_matrices:
                with self.subTest(function=function.__name__, dtype=str(matrix.dtype)):
                    with self.assertRaisesRegex(
                        ValueError,
                        "cost_matrix must be numeric",
                    ):
                        function(matrix)

    def test_probability_matrix_rejects_boolean_and_text_entries(self):
        config = CandidatePruningConfig(probability_threshold=0.5)
        invalid_probability_matrices = (
            np.array([[True]]),
            np.array([[np.bool_(False)]], dtype=object),
            np.array([["0.75"]]),
            np.array([[b"0.75"]], dtype=object),
        )

        for probabilities in invalid_probability_matrices:
            with self.subTest(dtype=str(probabilities.dtype)):
                with self.assertRaisesRegex(
                    ValueError,
                    "probability_matrix must be numeric",
                ):
                    candidate_mask_from_costs(
                        np.array([[1.0]]),
                        probability_matrix=probabilities,
                        config=config,
                    )

    def test_probability_matrix_rejects_complex_entries(self):
        config = CandidatePruningConfig(probability_threshold=0.5)
        invalid_probability_matrices = (
            np.array([[0.75 + 1.0j]]),
            np.array([[0.75 + 1.0j]], dtype=object),
        )

        for probabilities in invalid_probability_matrices:
            with self.subTest(dtype=str(probabilities.dtype)):
                with self.assertRaisesRegex(
                    ValueError,
                    "probability_matrix must be real-valued numeric",
                ):
                    candidate_mask_from_costs(
                        np.array([[1.0]]),
                        probability_matrix=probabilities,
                        config=config,
                    )

    def test_probability_matrix_rejects_temporal_entries(self):
        config = CandidatePruningConfig(probability_threshold=0.5)
        invalid_probability_matrices = (
            np.array([["2026-06-25"]], dtype="datetime64[D]"),
            np.array([[np.timedelta64(1, "s")]]),
            np.array([[np.datetime64("2026-06-25")]], dtype=object),
            np.array([[np.timedelta64(1, "s")]], dtype=object),
        )

        for probabilities in invalid_probability_matrices:
            with self.subTest(dtype=str(probabilities.dtype)):
                with self.assertRaisesRegex(
                    ValueError,
                    "probability_matrix must be numeric",
                ):
                    candidate_mask_from_costs(
                        np.array([[1.0]]),
                        probability_matrix=probabilities,
                        config=config,
                    )


if __name__ == "__main__":
    unittest.main()
