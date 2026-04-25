import unittest

import numpy as _np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import random


class TestBackendRandom(unittest.TestCase):
    def test_randint_returns_integer_samples_in_bounds(self):
        if pyrecest.backend.__backend_name__ == "jax":
            samples = random.randint((64,), minval=0, maxval=5)
        elif pyrecest.backend.__backend_name__ == "pytorch":
            samples = random.randint(0, 5, (64,))
        else:
            samples = random.randint(0, 5, size=(64,))

        samples_np = _np.asarray(samples)
        self.assertEqual(samples_np.shape, (64,))
        self.assertTrue(_np.issubdtype(samples_np.dtype, _np.integer))
        self.assertTrue(_np.all(samples_np >= 0))
        self.assertTrue(_np.all(samples_np < 5))


if __name__ == "__main__":
    unittest.main()
