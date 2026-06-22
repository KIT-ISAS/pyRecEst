"""Regression tests for nearly coordinated-turn motion-model validation."""

import unittest

import numpy as np
from pyrecest.models import nearly_coordinated_turn_model
from pyrecest.models.motion_models import (
    nearly_coordinated_turn_model as module_nearly_coordinated_turn_model,
)


class TestNearlyCoordinatedTurnModelValidation(unittest.TestCase):
    def test_rejects_invalid_turn_rate_variance(self):
        constructors = (
            nearly_coordinated_turn_model,
            module_nearly_coordinated_turn_model,
        )
        for constructor in constructors:
            for turn_rate_variance in (-1.0, np.nan, np.inf, True):
                with self.subTest(
                    constructor=constructor.__module__,
                    turn_rate_variance=turn_rate_variance,
                ):
                    with self.assertRaisesRegex(ValueError, "turn_rate_variance"):
                        constructor(turn_rate_variance=turn_rate_variance)


if __name__ == "__main__":
    unittest.main()
