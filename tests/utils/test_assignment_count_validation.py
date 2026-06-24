import unittest

import numpy as np
from pyrecest.utils import murty_k_best_assignments


class AssignmentCountValidationTest(unittest.TestCase):
    def test_text_scalar_k_is_rejected(self):
        for invalid_k in (
            "1",
            b"1",
            np.str_("1"),
            np.bytes_(b"1"),
            np.array("1"),
            np.array(b"1"),
        ):
            with self.subTest(invalid_k=invalid_k):
                with self.assertRaisesRegex(ValueError, "k must be an integer"):
                    murty_k_best_assignments(np.eye(2), k=invalid_k)


if __name__ == "__main__":
    unittest.main()
