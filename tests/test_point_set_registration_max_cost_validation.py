import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import solve_gated_assignment


class TestRegistrationMaxCostValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_solve_gated_assignment_rejects_nonnumeric_max_cost(self):
        cost_matrix = array([[0.5]])

        for invalid_max_cost in (True, False, "1.0", b"1.0"):
            with self.subTest(invalid_max_cost=invalid_max_cost):
                with self.assertRaisesRegex(ValueError, "max_cost"):
                    solve_gated_assignment(cost_matrix, max_cost=invalid_max_cost)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_solve_gated_assignment_preserves_numeric_scalar_max_cost(self):
        cost_matrix = array([[0.5], [2.0]])

        assignment = solve_gated_assignment(cost_matrix, max_cost=array(1.0))

        npt.assert_array_equal(assignment, array([0, -1]))


if __name__ == "__main__":
    unittest.main()
