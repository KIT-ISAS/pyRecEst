import unittest

from pyrecest.backend import array, comb, size


class TestBackendSize(unittest.TestCase):
    def test_size_returns_total_number_of_elements(self):
        self.assertEqual(size(array([[1, 2, 3], [4, 5, 6]])), 6)

    def test_size_returns_axis_length(self):
        self.assertEqual(size(array([[1, 2, 3], [4, 5, 6]]), axis=1), 3)

    def test_size_handles_python_array_like_inputs(self):
        values = [[1, 2, 3], [4, 5, 6]]

        self.assertEqual(size(values), 6)
        self.assertEqual(size(values, axis=0), 2)
        self.assertEqual(size(values, axis=1), 3)

    def test_size_handles_python_scalars(self):
        self.assertEqual(size(3.0), 1)


class TestBackendComb(unittest.TestCase):
    def test_comb_returns_standard_binomial_values(self):
        self.assertEqual(comb(5, 2), 10)
        self.assertEqual(comb(3, 4), 0)


if __name__ == "__main__":
    unittest.main()
