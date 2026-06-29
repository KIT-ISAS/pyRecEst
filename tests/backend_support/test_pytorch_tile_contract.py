import unittest

import numpy as np
import numpy.testing as npt

try:
    from pyrecest._backend import pytorch as pytorch_backend
except ModuleNotFoundError:
    pytorch_backend = None


@unittest.skipIf(pytorch_backend is None, "PyTorch is not installed")
class TestRawPytorchTileContract(unittest.TestCase):
    def test_matches_numpy_repetition_semantics(self):
        cases = [
            (np.arange(6).reshape(2, 3), 2),
            (np.arange(6).reshape(2, 3), (2,)),
            (np.arange(6).reshape(2, 3), (2, 1)),
            (np.arange(6).reshape(2, 3), (2, 1, 2)),
            (np.arange(3), (2, 2)),
        ]

        for values, reps in cases:
            with self.subTest(reps=reps):
                expected = np.tile(values, reps)
                result = pytorch_backend.tile(values, reps)

                npt.assert_array_equal(pytorch_backend.to_numpy(result), expected)
                self.assertEqual(tuple(result.shape), expected.shape)

    def test_accepts_tensor_repetitions(self):
        result = pytorch_backend.tile(
            [[1, 2], [3, 4]], pytorch_backend.array([2, 1])
        )
        expected = np.tile(np.array([[1, 2], [3, 4]]), (2, 1))

        npt.assert_array_equal(pytorch_backend.to_numpy(result), expected)
        self.assertEqual(tuple(result.shape), expected.shape)

    def test_rejects_negative_repetitions(self):
        with self.assertRaises(ValueError):
            pytorch_backend.tile([1, 2], (-1,))
