import unittest

import pyrecest.backend
from pyrecest.backend import array, isscalar, random


class TestBackendInterface(unittest.TestCase):
    def test_isscalar_matches_numpy_semantics(self):
        self.assertTrue(isscalar(1))
        self.assertTrue(isscalar(1.5))
        self.assertFalse(isscalar(array(1)))
        self.assertFalse(isscalar(array([1])))
        self.assertFalse(isscalar([1]))

    @unittest.skipUnless(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="PyTorch-specific backend behavior",
    )
    def test_pytorch_choice_without_replacement_returns_unique_values(self):
        values = array([0, 1, 2, 3])
        random.seed(0)

        samples = random.choice(values, size=values.shape[0], replace=False)

        self.assertEqual(samples.shape, values.shape)
        self.assertEqual(len(set(samples.tolist())), values.shape[0])
