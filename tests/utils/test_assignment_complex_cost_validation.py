import unittest

import numpy as np
import pyrecest.backend
from pyrecest.utils import (
    min_cost_max_cardinality_assignment,
    murty_k_best_assignments,
)


class AssignmentComplexCostValidationTest(unittest.TestCase):
    @staticmethod
    def _solvers():
        return (
            ("murty", lambda matrix: murty_k_best_assignments(matrix, k=1)),
            ("max_cardinality", min_cost_max_cardinality_assignment),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_complex_cost_matrix_entries_are_rejected(self):
        complex_matrices = (
            np.array([[1.0 + 2.0j, 2.0 + 0.0j]]),
            np.array([[1.0, 2.0 + 0.0j]], dtype=object),
        )
        for solver_name, solver in self._solvers():
            for matrix in complex_matrices:
                with self.subTest(solver=solver_name, dtype=str(matrix.dtype)):
                    with self.assertRaisesRegex(ValueError, "real-valued"):
                        solver(matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_complex_non_assignment_costs_are_rejected(self):
        matrix = np.array([[1.0]])
        invalid_costs = (
            {"row_non_assignment_costs": np.array([1.0 + 2.0j])},
            {"col_non_assignment_costs": np.array([1.0 + 0.0j], dtype=object)},
        )
        for kwargs in invalid_costs:
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaisesRegex(ValueError, "real-valued"):
                    murty_k_best_assignments(matrix, k=1, **kwargs)


if __name__ == "__main__":
    unittest.main()
