import unittest
import warnings

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import all, array, isinf, linalg
from pyrecest.evaluation import determine_all_deviations


class _ObjectMatrix:
    """Small 2-D object container matching the function's indexing contract."""

    ndim = 2

    def __init__(self, rows):
        self._rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self._rows[row][col]
        return self._rows[key]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestDetermineAllDeviations(unittest.TestCase):
    def test_missing_estimates_do_not_require_filter_metadata(self):
        groundtruths = _ObjectMatrix(
            [
                [array([1.0, 2.0])],
                [array([3.0, 4.0])],
            ]
        )
        results = _ObjectMatrix([[None, None]])

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            deviations = determine_all_deviations(
                results,
                lambda filter_state: filter_state,
                lambda estimate, truth: linalg.norm(estimate - truth),
                groundtruths,
            )

        self.assertTrue(all(isinf(deviations)))
        self.assertTrue(
            any(
                "Filter result 0 apparently failed 2 times" in str(warning.message)
                for warning in caught_warnings
            )
        )


if __name__ == "__main__":
    unittest.main()
