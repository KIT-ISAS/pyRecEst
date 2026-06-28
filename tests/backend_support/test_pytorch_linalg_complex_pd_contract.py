import unittest
from typing import Any

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchLinalgComplexPdContract(unittest.TestCase):
    def test_complex_positive_definite_result_is_python_bool(self):
        matrix = pytorch_backend.array(
            [[2.0 + 0.0j, 0.0 + 0.25j], [0.0 - 0.25j, 1.5 + 0.0j]],
            dtype=pytorch_backend.complex128,
        )

        self.assertIs(pytorch_backend.linalg.is_single_matrix_pd(matrix), True)

    def test_complex_non_positive_definite_result_is_python_bool(self):
        matrix = pytorch_backend.array(
            [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            dtype=pytorch_backend.complex128,
        )

        self.assertIs(pytorch_backend.linalg.is_single_matrix_pd(matrix), False)


if __name__ == "__main__":
    unittest.main()
