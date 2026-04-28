import unittest
from typing import Any

pytorch_backend: Any
try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchBackendStd(unittest.TestCase):
    def test_std_accepts_numpy_style_axis_and_ddof(self):
        values = pytorch_backend.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        population_std = pytorch_backend.std(values, axis=0)
        sample_std = pytorch_backend.std(values, axis=0, ddof=1)

        self.assertTrue(
            pytorch_backend.allclose(
                population_std,
                pytorch_backend.array([1.632993161855452, 1.632993161855452]),
            )
        )
        self.assertTrue(
            pytorch_backend.allclose(sample_std, pytorch_backend.array([2.0, 2.0]))
        )

    def test_std_accepts_keepdims_and_dtype(self):
        values = pytorch_backend.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=pytorch_backend.float32
        )

        result = pytorch_backend.std(
            values, axis=0, keepdims=True, dtype=pytorch_backend.float64
        )

        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.dtype, pytorch_backend.float64)

    def test_std_rejects_conflicting_ddof_and_correction(self):
        values = pytorch_backend.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            pytorch_backend.std(values, ddof=1, correction=1)


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestPytorchBackendCov(unittest.TestCase):
    def test_cov_bias_true_defaults_to_equal_weights(self):
        values = pytorch_backend.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])

        result = pytorch_backend.cov(values, bias=True)

        expected = pytorch_backend.array([[2.0 / 3.0, 2.0], [2.0, 56.0 / 9.0]])
        self.assertTrue(pytorch_backend.allclose(result, expected))

    def test_cov_bias_true_normalizes_aweights_without_mutating(self):
        values = pytorch_backend.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])
        aweights = pytorch_backend.array([1.0, 2.0, 3.0])
        original_aweights = pytorch_backend.copy(aweights)

        result = pytorch_backend.cov(values, aweights=aweights, bias=True)

        expected = pytorch_backend.array(
            [[5.0 / 9.0, 16.0 / 9.0], [16.0 / 9.0, 53.0 / 9.0]]
        )
        self.assertTrue(pytorch_backend.allclose(result, expected))
        self.assertTrue(pytorch_backend.allclose(aweights, original_aweights))


if __name__ == "__main__":
    unittest.main()
