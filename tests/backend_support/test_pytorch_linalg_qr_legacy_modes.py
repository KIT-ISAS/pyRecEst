import unittest
from typing import Any

import numpy as np

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchLinalgQrLegacyModes(unittest.TestCase):
    def test_mode_full_alias_matches_reduced_numpy_result(self):
        matrix_np = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]])
        matrix = pytorch_backend.array(matrix_np, dtype=pytorch_backend.float64)

        q, r = pytorch_backend.linalg.qr(matrix, mode="full")
        expected_q, expected_r = np.linalg.qr(matrix_np, mode="reduced")

        self.assertEqual(tuple(q.shape), expected_q.shape)
        self.assertEqual(tuple(r.shape), expected_r.shape)
        self.assertTrue(
            pytorch_backend.allclose(
                q, pytorch_backend.array(expected_q, dtype=pytorch_backend.float64)
            )
        )
        self.assertTrue(
            pytorch_backend.allclose(
                r, pytorch_backend.array(expected_r, dtype=pytorch_backend.float64)
            )
        )

    def test_mode_economic_returns_numpy_style_geqrf_storage(self):
        matrices_np = np.stack(
            [
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]),
                np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 4.0]]),
            ]
        )
        matrices = pytorch_backend.array(matrices_np, dtype=pytorch_backend.float64)

        result = pytorch_backend.linalg.qr(matrices, mode="economic")
        expected_h, _ = np.linalg.qr(matrices_np, mode="raw")
        expected = np.swapaxes(expected_h, -1, -2)

        self.assertEqual(tuple(result.shape), expected.shape)
        self.assertTrue(
            pytorch_backend.allclose(
                result, pytorch_backend.array(expected, dtype=pytorch_backend.float64)
            )
        )


if __name__ == "__main__":
    unittest.main()
