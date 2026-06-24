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
        )

        for function in (candidate_mask_from_costs, prune_pairwise_cost_matrix):
            for matrix in invalid_matrices:
                with self.subTest(function=function.__name__, matrix=matrix):
                    with self.assertRaisesRegex(
                        ValueError,
                        "cost_matrix must be numeric",
                    ):
                        function(matrix)

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


if __name__ == "__main__":
    unittest.main()
