import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import (
    amin,
    array,
    isscalar,
    prod,
    random,
    ravel_tril_indices,
    scatter_add,
    to_numpy,
)


class TestBackendInterface(unittest.TestCase):
    @staticmethod
    def _scalar(value):
        converted = to_numpy(value)
        if hasattr(converted, "item"):
            return converted.item()
        return converted

    def test_isscalar_matches_numpy_semantics(self):
        self.assertTrue(isscalar(1))
        self.assertTrue(isscalar(1.5))
        self.assertFalse(isscalar(array(1)))
        self.assertFalse(isscalar(array([1])))
        self.assertFalse(isscalar([1]))

    def test_prod_without_axis_reduces_all_elements(self):
        values = array([[2, 3], [4, 5]])

        self.assertEqual(self._scalar(prod(values)), 120)

    def test_amin_without_axis_reduces_all_elements(self):
        values = array([[2, 3], [4, 5]])

        self.assertEqual(self._scalar(amin(values)), 2)

    def test_ravel_tril_indices_returns_flat_indices(self):
        npt.assert_array_equal(to_numpy(ravel_tril_indices(3)), [0, 3, 4, 6, 7, 8])

    def test_scatter_add_preserves_existing_values(self):
        result = scatter_add(array([10, 20, 30]), 0, array([0, 2]), array([1, 2]))

        npt.assert_array_equal(to_numpy(result), [11, 20, 32])

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "numpy",  # pylint: disable=no-member
        reason="NumPy-specific backend behavior",
    )
    def test_numpy_choice_uses_seeded_global_rng_state(self):
        values = array([[0, 1], [2, 3], [4, 5]])
        weights = array([0.1, 0.2, 0.7])

        random.seed(12345)
        state = random.get_state()
        first = random.choice(values, size=25, replace=True, p=weights)
        random.set_state(state)
        second = random.choice(values, size=25, replace=True, p=weights)

        self.assertEqual(first.shape, (25, 2))
        npt.assert_array_equal(first, second)

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "pytorch",  # pylint: disable=no-member
        reason="PyTorch-specific backend behavior",
    )
    def test_pytorch_choice_without_replacement_returns_unique_values(self):
        values = array([0, 1, 2, 3])
        random.seed(0)

        samples = random.choice(values, size=values.shape[0], replace=False)

        self.assertEqual(samples.shape, values.shape)
        self.assertEqual(len(set(samples.tolist())), values.shape[0])
