import unittest

from pyrecest.backend import array, isscalar


class TestBackendInterface(unittest.TestCase):
    def test_isscalar_matches_numpy_semantics(self):
        self.assertTrue(isscalar(1))
        self.assertTrue(isscalar(1.5))
        self.assertFalse(isscalar(array(1)))
        self.assertFalse(isscalar(array([1])))
        self.assertFalse(isscalar([1]))
