import unittest
from typing import Any

import numpy as np

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchLinalgQrContract(unittest.TestCase):
    def test_mode_r_returns_numpy_style_r_factor(self):
        matrices_np = np.stack(
            [
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]),
                np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 4.0]]),
            ]
        )
        matrices = pytorch_backend.array(matrices_np, dtype=pytorch_backend.float64)

        result = pytorch_backend.linalg.qr(matrices, mode="r")
        expected = np.linalg.qr(matrices_np, mode="r")

        self.assertEqual(tuple(result.shape), expected.shape)
        self.assertTrue(
            pytorch_backend.allclose(
                result, pytorch_backend.array(expected, dtype=pytorch_backend.float64)
            )
        )

    def test_mode_raw_returns_numpy_style_h_tau_tuple(self):
        matrix_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
        matrix = pytorch_backend.array(matrix_np, dtype=pytorch_backend.float64)

        h, tau = pytorch_backend.linalg.qr(matrix, mode="raw")
        expected_h, expected_tau = np.linalg.qr(matrix_np, mode="raw")

        self.assertEqual(tuple(h.shape), expected_h.shape)
        self.assertEqual(tuple(tau.shape), expected_tau.shape)
        self.assertTrue(
            pytorch_backend.allclose(
                h, pytorch_backend.array(expected_h, dtype=pytorch_backend.float64)
            )
        )
        self.assertTrue(
            pytorch_backend.allclose(
                tau, pytorch_backend.array(expected_tau, dtype=pytorch_backend.float64)
            )
        )


if __name__ == "__main__":
    unittest.main()
