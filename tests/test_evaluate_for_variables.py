# pylint: disable=protected-access
import unittest

from pyrecest.evaluation.evaluate_for_variables import _expand_filter_configs


class TestEvaluateForVariables(unittest.TestCase):
    def test_expand_filter_configs_accepts_scalar_parameters(self):
        filter_configs = [
            {"name": "kf", "parameter": None},
            {"name": "pf", "parameter": 100},
            {"name": "pf", "parameter": [51, 81]},
        ]

        expanded_filter_configs = _expand_filter_configs(filter_configs)

        self.assertEqual(
            expanded_filter_configs,
            [
                {"name": "kf", "parameter": None},
                {"name": "pf", "parameter": 100},
                {"name": "pf", "parameter": 51},
                {"name": "pf", "parameter": 81},
            ],
        )


if __name__ == "__main__":
    unittest.main()
