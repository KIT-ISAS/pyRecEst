import unittest

import numpy.testing as npt

from pyrecest.backend import array, ones
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.filters.abstract_grid_filter import AbstractGridFilter


class AbstractGridFilterTest(unittest.TestCase):
    def test_filter_state_conversion_preserves_cartesian_product_grid(self):
        initial_state = HypertoroidalGridDistribution.from_function(
            lambda xs: ones(xs.shape[0]),
            (3, 4),
            enforce_pdf_nonnegative=False,
        )
        filt = AbstractGridFilter(initial_state)

        new_state = HypertoroidalWrappedNormalDistribution(
            array([0.2, 0.3]),
            array([[0.5, 0.1], [0.1, 0.7]]),
        )

        with self.assertWarnsRegex(RuntimeWarning, "new_state is not a GridDistribution"):
            filt.filter_state = new_state

        self.assertIsInstance(filt.filter_state, HypertoroidalGridDistribution)
        self.assertEqual(filt.filter_state.grid_type, "cartesian_prod")
        self.assertFalse(filt.filter_state.enforce_pdf_nonnegative)
        self.assertEqual(filt.filter_state.grid_values.shape, (3, 4))
        npt.assert_allclose(filt.filter_state.get_grid().shape, (12, 2))


if __name__ == "__main__":
    unittest.main()
