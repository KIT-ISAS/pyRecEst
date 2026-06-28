import unittest

from pyrecest.backend import array
from pyrecest.models import DensityTransitionModel, SampleableTransitionModel


class BrokenSampleCount:
    def __array__(self, dtype=None):  # pragma: no cover - exercised by NumPy
        raise TypeError("cannot coerce sample count")


class TransitionModelSampleCountValidationTest(unittest.TestCase):
    def test_sampleable_transition_rejects_uncoercible_sample_count_cleanly(self):
        model = SampleableTransitionModel(lambda state, n=1: state)

        with self.assertRaisesRegex(ValueError, "n must be a nonnegative integer"):
            model.sample_next(array([1.0]), n=BrokenSampleCount())

    def test_density_transition_rejects_uncoercible_sample_count_cleanly(self):
        model = DensityTransitionModel(
            lambda state_next, state_previous: 1.0,
            sample_next=lambda state, n=1: state,
        )

        with self.assertRaisesRegex(ValueError, "n must be a nonnegative integer"):
            model.sample_next(array([1.0]), n=BrokenSampleCount())


if __name__ == "__main__":
    unittest.main()
