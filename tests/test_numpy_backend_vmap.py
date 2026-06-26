import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
from pyrecest.backend import array, to_numpy


def _add_scaled_offset(value, offset):
    return value + 2.0 * offset


class NumpyBackendVmapTest(unittest.TestCase):
    def setUp(self):
        if backend.__backend_name__ != "numpy":
            self.skipTest("NumPy backend vmap contract")

    def test_vmap_maps_over_first_axis(self):
        values = array([[1.0, 2.0], [3.0, 4.0]])
        offsets = array([[0.5, 1.0], [1.5, 2.0]])

        mapped = backend.vmap(_add_scaled_offset)
        result = mapped(values, offsets)

        npt.assert_allclose(to_numpy(result), np.array([[2.0, 4.0], [6.0, 8.0]]))


if __name__ == "__main__":
    unittest.main()
