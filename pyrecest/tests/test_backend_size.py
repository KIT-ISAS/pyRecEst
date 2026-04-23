import unittest

from pyrecest.backend import array, size


class TestBackendSize(unittest.TestCase):
    def test_size_returns_total_number_of_elements(self):
        self.assertEqual(size(array([[1, 2, 3], [4, 5, 6]])), 6)

    def test_size_returns_axis_length(self):
        self.assertEqual(size(array([[1, 2, 3], [4, 5, 6]]), axis=1), 3)


if __name__ == "__main__":
    unittest.main()
