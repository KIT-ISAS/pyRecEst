"""Regression tests for transition samplers without sample-count support."""

import unittest

from pyrecest.backend import array
from pyrecest.models import SampleableTransitionModel, sample_next_state


class UnsupportedTransitionSampleCountTest(unittest.TestCase):
    def test_placeholder(self):
        model = SampleableTransitionModel(lambda state: state)
        self.assertIsNotNone(sample_next_state(model, array([0.0]), n=1))


if __name__ == "__main__":
    unittest.main()
