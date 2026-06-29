import unittest

from pyrecest import backend
from pyrecest.backend import array
from pyrecest.utils.roi_assignment import assign_by_similarity_matrix


class TestRoiAssignmentUnmatchedValue(unittest.TestCase):
    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_rejects_unmatched_value_colliding_with_columns(self):
        similarity_matrix = array([[1.0, 0.5]])

        for unmatched_value in (0, 1):
            with self.subTest(unmatched_value=unmatched_value):
                with self.assertRaisesRegex(ValueError, "unmatched_value"):
                    assign_by_similarity_matrix(
                        similarity_matrix,
                        unmatched_value=unmatched_value,
                    )

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_rejects_noninteger_unmatched_value(self):
        similarity_matrix = array([[1.0]])

        for unmatched_value in (True, 1.5, float("nan"), [1]):
            with self.subTest(unmatched_value=unmatched_value):
                with self.assertRaisesRegex(ValueError, "unmatched_value"):
                    assign_by_similarity_matrix(
                        similarity_matrix,
                        unmatched_value=unmatched_value,
                    )


if __name__ == "__main__":
    unittest.main()
