import unittest

import numpy as np

pytorch_backend = None
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pass


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchLinalgNormAxis(unittest.TestCase):
    def test_norm_accepts_numpy_scalar_axis(self):
        values = pytorch_backend.array([[3.0, 4.0], [0.0, 5.0]])

        result = pytorch_backend.linalg.norm(values, axis=np.array(1))

        expected = pytorch_backend.array([5.0, 5.0])
        self.assertTrue(pytorch_backend.allclose(result, expected))

    def test_norm_accepts_singleton_numpy_array_axis(self):
        values = pytorch_backend.array([[3.0, 4.0], [0.0, 5.0]])

        result = pytorch_backend.linalg.norm(values, axis=np.array([0]))

        expected = pytorch_backend.array([3.0, 41.0**0.5])
        self.assertTrue(pytorch_backend.allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
