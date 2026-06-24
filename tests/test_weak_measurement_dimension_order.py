import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.models.weak_measurement import block_diag_measurement_covariance


class TestWeakMeasurementDimensionOrder(unittest.TestCase):
    def test_explicit_dimension_order_must_cover_all_mapping_keys(self):
        with self.assertRaisesRegex(KeyError, "omits std entries"):
            block_diag_measurement_covariance(
                trusted_std={"x": 1.0, "y": 2.0},
                weak_std={"range": 10.0},
                dimension_order=["x", "range"],
            )

    def test_explicit_dimension_order_preserves_complete_mapping_order(self):
        covariance = block_diag_measurement_covariance(
            trusted_std={"x": 1.0, "y": 2.0},
            weak_std={"range": 10.0},
            dimension_order=["range", "x", "y"],
        )

        npt.assert_allclose(covariance, np.diag([100.0, 1.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
